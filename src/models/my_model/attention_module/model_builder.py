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
    신규 파이프라인 (1-Layer GCN + Graph Attention Readout):

    fp_input (B, fp_dim)                  → ProjectionBlock → fp_proj (B, PROJECT_DIM)
    graph (B,N,feat), adj (B,N,N), mask   → OneLayerGCN    → node_mat (B, N, GCN_OUT_DIM)

    fp_proj + node_mat → GraphAttentionReadout → z_readout (B, PROJECT_DIM)

    Concat([fp_proj, z_readout]) → (B, 2*PROJECT_DIM)
    └─ [학습 시] → MLPClassifier → logits (B, 2)
    └─ [추출 시] → return_embedding=True → (B, 2*PROJECT_DIM) [= final_embeddings.npy]

    설계 철학:
      - FP(Q)가 분자의 전역 형태를 파악한 채, GCN 노드(K,V)를 순회하며
        독성 패턴(Toxicophore)이 있는 정확한 원자 이웃에 집중.
      - Differential Attention이 노이즈 원자 가중치를 차감(-), 핵심 원자를 증폭.
      - Z_readout: 농축된 독성 시그널 → Concat 후 Person B(XGBoost)에 전달.
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

        # ③ Graph Attention Readout (Differential weighted-sum over atoms)
        self.readout = GraphAttentionReadout(
            query_dim = config.PROJECT_DIM,
            node_dim  = config.GCN_OUT_DIM,
            out_dim   = config.PROJECT_DIM,
            dropout   = config.ATTN_DROPOUT,
        )

        # Concat 출력 차원: FP proj + Z_readout
        self.output_dim = config.PROJECT_DIM * 2   # (B, 2 * PROJECT_DIM)

        # ④ Auxiliary MLP Head (학습 전용)
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
        # ① FP 투영
        fp_vec = self.fp_proj(fp)             # (B, PROJECT_DIM)

        # ② GCN: 위상 보존 로컬 원자 인코딩
        node_mat = self.gcn(x, adj, mask)     # (B, max_N, GCN_OUT_DIM)

        # ③ Attention Readout: 독성 원자에 집중, 노이즈 상쇄
        z_readout = self.readout(fp_vec, node_mat, mask)   # (B, PROJECT_DIM)

        # ④ Concat: 전역 + 국소 융합 벡터 (최종 임베딩)
        fused = torch.cat([fp_vec, z_readout], dim=-1)    # (B, 2*PROJECT_DIM)

        if return_embedding:
            return fused   # → final_embeddings.npy (Person B용)

        return self.classifier(fused)   # (B, 2) 학습용 로짓


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
