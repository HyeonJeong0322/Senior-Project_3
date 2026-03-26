"""
model_builder.py  (Person A — Step 2 아키텍처)
===============================================
전체 DL 모델 파이프라인:

  [기존] DILIAttentionModel : FP(Global) + BRICS BoF(Local) → CrossedDiffAttn → MLP
  [신규] DILIGCNModel       : FP(Global) + 1-Layer GCN(Local) → GraphAttnReadout → Concat → MLP(aux)

최종 산출 목표: final_embeddings.npy (Person B 파트 XGBoost 입력용)
MLP(Auxiliary Head)는 학습 중 역전파를 위한 보조 도구이며, 추론/임베딩 추출 시 제거됨.

  ① End-to-End 학습: CrossEntropyLoss로 Attention 포함 전체 모델 역전파
  ② 임베딩 추출 모드: Auxiliary Head 직전 Concat 벡터를 .npy로 저장
  ③ Label Smoothing: 소규모 데이터셋의 과적합 완화
"""

import torch
import torch.nn as nn

from attention_module import (
    CrossedDiffAttentionStack,
    OneLayerGCN,
    GraphAttentionReadout,
    NodeFPCrossAttention,
    ProjectionBlock,
)
import config


# ─────────────────────────────────────────────────────────────
# MLP Auxiliary Head (학습 전용, 임베딩 추출 시 Bypass 가능)
# ─────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Concat 벡터 → 이진 분류 (학습 Phase Auxiliary Head).
    return_embedding=True 시 MLP 내부 표현(hidden_dim) 반환.
    """
    def __init__(self, input_dim: int,
                 hidden_dim: int = config.MLP_HIDDEN_DIM,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = config.FF_DROPOUT):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor,
                return_embedding: bool = False):
        h = self.drop(self.act(self.norm(self.fc1(x))))
        if return_embedding:
            return h   # (B, hidden_dim) → Person B용
        return self.fc2(h)   # (B, num_classes)


# ─────────────────────────────────────────────────────────────
# [기존] BoF 기반 모델 (하위 호환 유지)
# ─────────────────────────────────────────────────────────────

class DILIAttentionModel(nn.Module):
    """
    기존 파이프라인 (BRICS BoF Local Feature 사용).
    하위 호환을 위해 유지. 신규 학습은 DILIGCNModel 권장.

    fp_input (B, fp_dim) + sub_input (B, sub_dim)
    → CrossedDiffAttentionStack → MLPClassifier
    → logits (B, 2) or embedding (B, hidden_dim)
    """
    def __init__(self, fp_dim: int, sub_dim: int):
        super().__init__()
        self.attention_stack = CrossedDiffAttentionStack(
            fp_input_dim  = fp_dim,
            sub_input_dim = sub_dim,
            proj_dim   = config.PROJECT_DIM,
            num_heads  = config.NUM_HEADS,
            num_stacks = config.NUM_STACKS,
            attn_drop  = config.ATTN_DROPOUT,
            ff_drop    = config.FF_DROPOUT,
        )
        self.classifier = MLPClassifier(
            input_dim   = self.attention_stack.output_dim,
            hidden_dim  = config.MLP_HIDDEN_DIM,
            num_classes = config.NUM_CLASSES,
            dropout     = config.FF_DROPOUT,
        )

    def forward(self, fp: torch.Tensor, sub: torch.Tensor,
                return_embedding: bool = False):
        fused  = self.attention_stack(fp, sub)
        return self.classifier(fused, return_embedding)


# ─────────────────────────────────────────────────────────────
# [신규] GCN + Attention Readout 기반 통합 모델
# ─────────────────────────────────────────────────────────────

class DILIGCNModel(nn.Module):
    """
    신규 파이프라인 (1-Layer GCN + 진짜 Crossed Attention):

    fp_input (B, fp_dim)                  → ProjectionBlock   → fp_vec   (B, PROJECT_DIM)
    graph (B,N,feat), adj (B,N,N), mask   → OneLayerGCN       → node_mat (B, N, GCN_OUT_DIM)

    [Crossed Attention — 양방향 독립 스트림]
      Pass 2: Node_i(Q) → fp_vec(K,V) → node_mat_updated  (각 원자가 전역 FP 참조)
      Pass 1: fp_vec(Q) → node_mat_updated(K,V) → z_readout (FP가 FP-informed 원자 참조)

    Concat([fp_vec, z_readout]) → (B, 2*PROJECT_DIM)
    └─ [학습 시] → MLPClassifier → logits (B, 2)
    └─ [추출 시] → return_embedding=True → (B, 2*PROJECT_DIM) [= final_embeddings.npy]

    설계 철학 (Crossed Attention 재설계):
      - 기존 단방향: FP → GCN (FP만 GCN을 봄)
      - 신규 양방향:
          Pass 2 (NodeFPCrossAttention): 각 원자_i가 전역 FP를 gate로 흡수
            → 독성기 원자: FP와 강하게 공명 → 높은 gate → self 강화
            → 노이즈 원자: FP와 무관 → gate ≈ 0 → 자동 억제
          Pass 1 (GraphAttentionReadout): FP가 이미 FP-정보를 흡수한 원자들 참조
            → 순환 편향 없음: Pass 2 입력은 순수 GCN node_mat (fp_vec 개입 전)
    """
    def __init__(self, fp_dim: int):
        super().__init__()

        # ① FP Projection Block (Global feature 차원 통일)
        self.fp_proj = ProjectionBlock(fp_dim, config.PROJECT_DIM, config.FF_DROPOUT)

        # ② 1-Layer GCN (Local topology encoder)
        self.gcn = OneLayerGCN(
            in_dim  = config.GCN_NODE_FEAT_DIM,
            out_dim = config.GCN_OUT_DIM,
            dropout = config.FF_DROPOUT,
        )

        # ③ Crossed Attention Pass 2: 각 원자가 전역 FP를 참조 (GCN → FP 방향)
        self.node_fp_cross = NodeFPCrossAttention(
            node_dim = config.GCN_OUT_DIM,
            fp_dim   = config.PROJECT_DIM,
            dropout  = config.ATTN_DROPOUT,
        )

        # ④ Crossed Attention Pass 1: FP가 FP-informed 원자들을 참조 (FP → GCN 방향)
        self.readout = GraphAttentionReadout(
            query_dim = config.PROJECT_DIM,
            node_dim  = config.GCN_OUT_DIM,
            out_dim   = config.PROJECT_DIM,
            dropout   = config.ATTN_DROPOUT,
        )

        # Concat 출력 차원: fp_vec + z_readout
        self.output_dim = config.PROJECT_DIM * 2   # (B, 2 * PROJECT_DIM)

        # ⑤ Auxiliary MLP Head (학습 전용)
        self.classifier = MLPClassifier(
            input_dim   = self.output_dim,
            hidden_dim  = config.MLP_HIDDEN_DIM,
            num_classes = config.NUM_CLASSES,
            dropout     = config.FF_DROPOUT,
        )

    def forward(self,
                fp:   torch.Tensor,
                x:    torch.Tensor,
                adj:  torch.Tensor,
                mask: torch.Tensor,
                return_embedding: bool = False):
        """
        fp   : (B, fp_dim)              — RF 선택된 Global FP
        x    : (B, max_N, node_feat)    — 패딩된 원자 피처
        adj  : (B, max_N, max_N)        — 패딩된 인접 행렬
        mask : (B, max_N)               — 유효 원자 마스크
        return_embedding=True  → (B, 2*PROJECT_DIM) 임베딩 [final_embeddings.npy]
        return_embedding=False → (B, 2) 로짓
        """
        # ① FP 투영 (순수 전역 특징)
        fp_vec   = self.fp_proj(fp)            # (B, PROJECT_DIM)

        # ② GCN: 위상 보존 로컬 원자 인코딩 (순수 구조 정보, fp_vec 개입 없음)
        node_mat = self.gcn(x, adj, mask)      # (B, max_N, GCN_OUT_DIM)

        # ③ Crossed Attention Pass 2: 각 원자가 전역 FP 맥락을 참조하여 자신을 업데이트
        #    입력: 순수 GCN node_mat  →  FP 편향 없이 독립 스트림 유지
        node_mat_fp = self.node_fp_cross(node_mat, fp_vec, mask)  # (B, max_N, GCN_OUT_DIM)

        # ④ Crossed Attention Pass 1: FP가 FP-정보를 흡수한 원자들을 참조
        #    node_mat_fp는 이미 전역 맥락을 반영했으므로 Readout 품질 향상
        z_readout = self.readout(fp_vec, node_mat_fp, mask)        # (B, PROJECT_DIM)

        # ⑤ Concat: 전역(fp_vec) + 교차 참조된 국소(z_readout) 융합 벡터
        fused = torch.cat([fp_vec, z_readout], dim=-1)             # (B, 2*PROJECT_DIM)

        if return_embedding:
            return fused

        return self.classifier(fused)


# ─────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def build_gcn_model(fp_dim: int,
                    device: torch.device = None) -> DILIGCNModel:
    """GCN 기반 통합 모델 인스턴스 생성 및 디바이스 이동."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DILIGCNModel(fp_dim=fp_dim)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GCN 모델 빌드] fp_dim={fp_dim}")
    print(f"               총 학습 파라미터: {total_params:,}")
    print(f"               임베딩 출력 차원: {model.output_dim}")
    return model


def build_model(fp_dim: int, sub_dim: int,
                device: torch.device = None) -> DILIAttentionModel:
    """[기존 호환] BoF 기반 Attention 모델 생성."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DILIAttentionModel(fp_dim=fp_dim, sub_dim=sub_dim)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[BoF 모델 빌드] fp_dim={fp_dim}, sub_dim={sub_dim}")
    print(f"               총 학습 파라미터: {total_params:,}")
    return model


if __name__ == "__main__":
    device = torch.device("cpu")
    B, N, fp_dim = 4, 30, 256

    model = build_gcn_model(fp_dim, device)

    dummy_fp   = torch.randn(B, fp_dim)
    dummy_x    = torch.randn(B, N, config.GCN_NODE_FEAT_DIM)
    dummy_adj  = torch.eye(N).unsqueeze(0).repeat(B, 1, 1)
    dummy_mask = torch.ones(B, N, dtype=torch.bool)

    logits = model(dummy_fp, dummy_x, dummy_adj, dummy_mask)
    emb    = model(dummy_fp, dummy_x, dummy_adj, dummy_mask, return_embedding=True)

    print("logits shape   :", logits.shape)   # (4, 2)
    print("embedding shape:", emb.shape)      # (4, 256) = (4, 2 * PROJECT_DIM)
