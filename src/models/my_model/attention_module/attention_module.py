"""
attention_module.py  (Person A — Step 2 핵심 모듈)
====================================================
Crossed Differential Attention Layer 및 3-Stack 블록 구현

  ① Linear Projection: 서로 다른 차원의 FP/Sub 벡터 → PROJECT_DIM 통일
  ② [NEW] 1-Layer GCN Local Encoder:
       SMILES 그래프(원자 노드 + 결합 엣지)를 위상(Topology) 손실 없이 인코딩.
       Oversmoothing 방지를 위해 딱 1번의 Message Passing만 수행.
  ③ [NEW] Graph Attention Readout:
       GCN Node 행렬(N개 원자)을 Differential Attention 스코어로 가중합하여
       단일 분자 표현 벡터로 압축. 노이즈 원자는 자동 상쇄(near-zero weight).
  ④ Crossed Diff Attention:
       Pass 1 : Q=FP,  K/V=Sub  →  FP가 주목하는 Substructure 탐색
       Pass 2 : Q=Sub, K/V=FP   →  Sub가 주목하는 전역 문맥 파악
  ⑤ Differential Attention (Microsoft 2024):
       두 개의 독립적인 Softmax Attention 값의 차(diff)를 활용하여
       관련 없는 Noise Attention을 상쇄시키고 신호 집중도를 높임
  ⑥ NUM_STACKS회 반복으로 정보 밀도를 점진적으로 증가

참고 논문: Ye et al. "Differential Transformer", arXiv:2410.05258 (2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ─────────────────────────────────────────────────────────────
# [NEW] 0. 원자 피처 추출 유틸리티 (data_preprocessing와 연동)
# ─────────────────────────────────────────────────────────────

# RDKit 원자 번호 → 허용 목록 (DILI 관련 주요 원소 44종)
_ALLOWED_ATOMS = [
    6, 7, 8, 9, 15, 16, 17, 35, 53,   # C,N,O,F,P,S,Cl,Br,I
    5, 14, 34, 52,                      # B,Si,Se,Te
    11, 12, 19, 20, 26, 29, 30, 33,    # Na,Mg,K,Ca,Fe,Cu,Zn,As
]
_ATOM_SET     = {a: i for i, a in enumerate(_ALLOWED_ATOMS)}
_ATOM_OHE_DIM = len(_ALLOWED_ATOMS) + 1   # +1 for 'other'
# 총 차원: 22 (OHE) + 1 (형식전하) + 1 (방향족) + 1 (수소 수) = 25 = GCN_NODE_FEAT_DIM


def atom_features(atom) -> list:
    """
    단일 RDKit Atom → 피처 벡터 리스트 (GCN_NODE_FEAT_DIM = 25 차원)
    구성: 원자번호 OHE (22) + 형식전하 (1) + 방향족 여부 (1) + 수소 수 (1)
    """
    anum = atom.GetAtomicNum()
    ohe  = [0.0] * _ATOM_OHE_DIM
    idx  = _ATOM_SET.get(anum, len(_ALLOWED_ATOMS))  # 미등록 → 마지막 칸
    ohe[idx] = 1.0

    return ohe + [
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
    ]


# ─────────────────────────────────────────────────────────────
# [NEW] 1-Layer GCN Local Encoder
# ─────────────────────────────────────────────────────────────

class OneLayerGCN(nn.Module):
    """
    위상(Topology) 보존 1-Layer Graph Convolutional Network.

    - 순수 PyTorch 구현 (외부 패키지 없음):
      배치 내 분자마다 원자 수가 다르므로, 최대 원자 수(max_atoms)로
      패딩된 [B, max_atoms, feat_dim] 텐서와
      마스킹된 인접 행렬 [B, max_atoms, max_atoms]을 입력받음.

    - Message Passing (1-hop, 단 1번):
        H_out = ReLU(LayerNorm( D^{-1} * A_self * H_in * W ))
        A_self: self-loop 포함 인접 행렬, D: 차수 정규화 행렬

    Oversmoothing을 방지하기 위해 레이어를 1개(1-hop)로 제한.
    DILI 독성기(Toxicophore)는 반경 1 이웃 내 국소 구조로 발현되므로
    1-hop 수렴이 최적의 정보 밀도를 제공함.
    """
    def __init__(self,
                 in_dim:  int   = config.GCN_NODE_FEAT_DIM,
                 out_dim: int   = config.GCN_OUT_DIM,
                 dropout: float = config.FF_DROPOUT):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.ReLU(inplace=True)

    def forward(self,
                x:    torch.Tensor,
                adj:  torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, max_atoms, in_dim)    — 패딩된 원자 피처 행렬
        adj  : (B, max_atoms, max_atoms) — self-loop 포함 인접 행렬
        mask : (B, max_atoms)            — 유효 원자 True, 패딩 False
        return: (B, max_atoms, out_dim)
        """
        # 1. 선형 투영: (B, N, in) → (B, N, out)
        h = self.W(x)
        # 2. 차수 정규화: D^{-1} * A  (행별 합의 역수로 나눔)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)   # (B, N, 1)
        # 3. 1-hop Message Passing: (B, N, N) @ (B, N, out) → (B, N, out)
        agg = torch.bmm(adj / degree, h)
        # 4. 활성화 + 정규화
        out = self.drop(self.act(self.norm(agg)))
        # 5. 패딩 위치 제로 마스킹
        out = out * mask.unsqueeze(-1).float()
        return out   # (B, N, out_dim)


# ─────────────────────────────────────────────────────────────
# [NEW] Graph Attention Readout
# ─────────────────────────────────────────────────────────────

class GraphAttentionReadout(nn.Module):
    """
    GCN 노드 행렬(N개 원자)을 전역 FP를 Query로 삼아
    Differential Attention 가중합으로 1개의 분자 벡터로 압축.

    핵심 역할 (프로젝트 목적 직결):
      - FP(Q): 전역적 독성 패턴 형태를 파악.
      - GCN Node K,V: 각 원자의 위상/이웃 환경 정보 보유.
      - A_diff = A1 - λ*A2: 독성 패턴과 무관한 원자는 near-zero로 상쇄,
        독성 Toxicophore 원자에만 높은 가중치 집중.
      - Z_readout = Σ(A_diff_i * V_i): 농축된 독성 시그널 벡터.
        → Projection Block에 전달.
    """
    def __init__(self,
                 query_dim: int   = config.PROJECT_DIM,
                 node_dim:  int   = config.GCN_OUT_DIM,
                 out_dim:   int   = config.PROJECT_DIM,
                 dropout:   float = config.ATTN_DROPOUT):
        super().__init__()
        self.W_q1 = nn.Linear(query_dim, node_dim, bias=False)
        self.W_q2 = nn.Linear(query_dim, node_dim, bias=False)
        self.W_k  = nn.Linear(node_dim,  node_dim, bias=False)
        self.W_v  = nn.Linear(node_dim,  out_dim,  bias=False)

        self.scale = math.sqrt(node_dim)
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(out_dim)

        # 학습 가능한 λ: 0.8 베이스 + sigmoid 범위 보정 (안정적 학습)
        self.lambda_init  = 0.8
        self.lambda_logit = nn.Parameter(torch.zeros(1))

    def forward(self,
                fp_vec:   torch.Tensor,
                node_mat: torch.Tensor,
                mask:     torch.Tensor) -> torch.Tensor:
        """
        fp_vec   : (B, query_dim) — 투영된 FP 벡터 (Query 역할)
        node_mat : (B, N, node_dim) — GCN 출력 노드 행렬 (Key, Value)
        mask     : (B, N)            — 유효 원자 True, 패딩 False
        return   : (B, out_dim)      — Readout된 단일 분자 표현 벡터
        """
        # ── Differential Query 투영 (두 갈래) ──
        q1 = self.W_q1(fp_vec).unsqueeze(1)   # (B, 1, node_dim)
        q2 = self.W_q2(fp_vec).unsqueeze(1)

        # ── Key / Value 투영 ──
        k = self.W_k(node_mat)                 # (B, N, node_dim)
        v = self.W_v(node_mat)                 # (B, N, out_dim)

        # ── 어텐션 스코어 계산 ──
        s1 = (q1 @ k.transpose(-2, -1)) / self.scale   # (B, 1, N)
        s2 = (q2 @ k.transpose(-2, -1)) / self.scale

        # 패딩 원자 마스킹 (-inf → softmax 후 0)
        pad_mask = (~mask).unsqueeze(1).float() * -1e9  # (B, 1, N)
        s1 = s1 + pad_mask
        s2 = s2 + pad_mask

        A1 = F.softmax(s1, dim=-1)   # (B, 1, N)
        A2 = F.softmax(s2, dim=-1)

        # λ: 0.8 ~ 1.0 범위 안정 유지
        lam = self.lambda_init + torch.sigmoid(self.lambda_logit) * 0.2

        # Differential: 노이즈 어텐션 상쇄, 핵심 원자 신호 증폭
        A_diff = self.drop(A1 - lam * A2)   # (B, 1, N)

        # ── 가중합 Readout ──
        z = (A_diff @ v).squeeze(1)   # (B, out_dim)
        return self.norm(z)




# ─────────────────────────────────────────────────────────────
# 1. Linear Projection Block (FP-Sub간 차원 통일)
# ─────────────────────────────────────────────────────────────

class ProjectionBlock(nn.Module):
    """
    FP(1191-dim) 또는 Sub(vocab-dim) → PROJECT_DIM(256)으로 선형 매핑.
    BatchNorm1d + ReLU + Dropout으로 학습 안정성 확보.
    """
    def __init__(self, input_dim: int, proj_dim: int = config.PROJECT_DIM,
                 dropout: float = config.FF_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) → (B, proj_dim)
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# 2. Differential Multi-Head Attention
# ─────────────────────────────────────────────────────────────

class DiffAttentionHead(nn.Module):
    """
    단일 Differential Attention Head.
        각 head를 절반 크기(head_dim/2)의 'Attention 쌍'으로 분리.
        Attention_1 - λ * Attention_2 로 차이를 계산하여
        Noise(irrelevant)를 상쇄하고 Signal에 집중.

    λ (lambda): 학습 가능한 스칼라. 초기값은 작은 값으로 시작.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = config.ATTN_DROPOUT):
        super().__init__()
        assert embed_dim % (num_heads * 2) == 0, \
            "embed_dim은 (num_heads * 2)의 배수여야 합니다."

        self.num_heads  = num_heads
        self.embed_dim  = embed_dim
        self.half_dim   = embed_dim // (num_heads * 2)  # 각 sub-head의 차원
        self.scale      = math.sqrt(self.half_dim)

        # Q는 2배 크기 (Q1, Q2 결합)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim // 2, bias=False)  # V는 절반
        self.W_o = nn.Linear(embed_dim // 2, embed_dim, bias=False)

        # 학습 가능한 λ (논문 권장: 0.8 + exp(λ_init) 형태, 단순화 사용)
        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.randn(self.half_dim) * 0.01)
        self.lambda_k1 = nn.Parameter(torch.randn(self.half_dim) * 0.01)
        self.lambda_q2 = nn.Parameter(torch.randn(self.half_dim) * 0.01)
        self.lambda_k2 = nn.Parameter(torch.randn(self.half_dim) * 0.01)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(embed_dim // 2)

    def forward(self, query: torch.Tensor,
                key:   torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        query: (B, d_model)  — 주체 벡터 (FP 또는 Sub)
        key  : (B, d_model)  — 참조 벡터 (각각의 반대)
        value: (B, d_model)  — 정보 제공 벡터
        return: (B, d_model)
        """
        B = query.shape[0]
        H = self.num_heads
        hd = self.half_dim   # sub-head 차원

        # ── Q/K/V 투영 ──
        Q = self.W_q(query)   # (B, d)
        K = self.W_k(key)     # (B, d)
        V = self.W_v(value)   # (B, d/2)

        # ── Multi-head reshape  ──
        # Q → 2H개의 sub-head (Q1, Q2로 분리)
        Q = Q.view(B, H, 2, hd)            # (B, H, 2, hd)
        K = K.view(B, H, 2, hd)
        V = V.view(B, H, hd)               # (B, H, hd)

        Q1, Q2 = Q[..., 0, :], Q[..., 1, :]   # (B, H, hd)
        K1, K2 = K[..., 0, :], K[..., 1, :]

        # ── Differential Attention ──
        # Q (B,H,hd) @ K (B,H,hd) → (B,H) — 분자 수가 1개(벡터)이므로 스칼라
        # 실제 아키텍처에서 [분자 1개 = 1 token]이므로 sequence=1
        # (B, H, 1, hd) 형태로 확장하여 표준 Attention 계산 수행
        Q1 = Q1.unsqueeze(2)   # (B, H, 1, hd)
        Q2 = Q2.unsqueeze(2)
        K1 = K1.unsqueeze(2)
        K2 = K2.unsqueeze(2)
        V  = V.unsqueeze(2)    # (B, H, 1, hd)

        # (B, H, 1, hd) @ (B, H, hd, 1) → (B, H, 1, 1)
        A1 = (Q1 @ K1.transpose(-2, -1)) / self.scale
        A2 = (Q2 @ K2.transpose(-2, -1)) / self.scale

        A1 = F.softmax(A1, dim=-1)
        A2 = F.softmax(A2, dim=-1)

        # λ 계산 (논문 수식 5: λ = exp(λq1·λk1) - exp(λq2·λk2) + λ_init)
        lam = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        )

        # Differential: A1 - λ * A2
        A_diff = self.dropout(A1 - lam * A2)   # (B, H, 1, 1)

        # 가중합
        out = A_diff @ V          # (B, H, 1, hd)
        out = out.squeeze(2)      # (B, H, hd)

        # 헤드 concat
        out = out.reshape(B, H * hd)            # (B, H*hd) = (B, d/2)
        out = self.norm(out)
        out = self.W_o(out)                     # (B, d)
        return out


# ─────────────────────────────────────────────────────────────
# 3. Crossed Diff Attention Layer (Pass 1 + Pass 2)
# ─────────────────────────────────────────────────────────────

class CrossedDiffAttentionLayer(nn.Module):
    """
    하나의 Crossed Attention 레이어:
      Pass 1: Q=fp,  K/V=sub  → fp_updated
      Pass 2: Q=sub, K/V=fp   → sub_updated
    두 결과 모두 Residual Connection + LayerNorm + FFN 적용.
    """
    def __init__(self, dim: int = config.PROJECT_DIM,
                 num_heads: int = config.NUM_HEADS,
                 attn_drop: float = config.ATTN_DROPOUT,
                 ff_drop:   float = config.FF_DROPOUT):
        super().__init__()

        # Pass 1: FP queries Sub
        self.attn_fp  = DiffAttentionHead(dim, num_heads, attn_drop)
        self.norm_fp1 = nn.LayerNorm(dim)
        self.ffn_fp   = self._make_ffn(dim, ff_drop)
        self.norm_fp2 = nn.LayerNorm(dim)

        # Pass 2: Sub queries FP
        self.attn_sub  = DiffAttentionHead(dim, num_heads, attn_drop)
        self.norm_sub1 = nn.LayerNorm(dim)
        self.ffn_sub   = self._make_ffn(dim, ff_drop)
        self.norm_sub2 = nn.LayerNorm(dim)

    @staticmethod
    def _make_ffn(dim: int, drop: float) -> nn.Module:
        """Position-wise Feed-Forward Network (FFN)."""
        return nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop),
        )

    def forward(self, fp: torch.Tensor, sub: torch.Tensor):
        """
        fp  : (B, dim)
        sub : (B, dim)
        returns: fp_out (B, dim), sub_out (B, dim)
        """
        # ── Pass 1: FP가 Sub를 참조 ──
        fp_attn   = self.attn_fp(query=fp, key=sub, value=sub)
        fp_res    = self.norm_fp1(fp + fp_attn)           # Residual
        fp_out    = self.norm_fp2(fp_res + self.ffn_fp(fp_res))

        # ── Pass 2: Sub가 FP를 참조 ──
        sub_attn  = self.attn_sub(query=sub, key=fp, value=fp)
        sub_res   = self.norm_sub1(sub + sub_attn)        # Residual
        sub_out   = self.norm_sub2(sub_res + self.ffn_sub(sub_res))

        return fp_out, sub_out


# ─────────────────────────────────────────────────────────────
# 4. 3-Stack Crossed Diff Attention Block
# ─────────────────────────────────────────────────────────────

class CrossedDiffAttentionStack(nn.Module):
    """
    CrossedDiffAttentionLayer를 num_stacks회 반복.
    각 Layer는 독립적인 가중치(Shared X)를 가짐.
    마지막 출력: fp_final과 sub_final을 Concatenate → (B, 2*dim)
    """
    def __init__(self, fp_input_dim:  int,
                       sub_input_dim: int,
                       proj_dim:    int = config.PROJECT_DIM,
                       num_heads:   int = config.NUM_HEADS,
                       num_stacks:  int = config.NUM_STACKS,
                       attn_drop:   float = config.ATTN_DROPOUT,
                       ff_drop:     float = config.FF_DROPOUT):
        super().__init__()

        # Linear Projection (차원 통일, 개선점 ①)
        self.fp_proj  = ProjectionBlock(fp_input_dim,  proj_dim, ff_drop)
        self.sub_proj = ProjectionBlock(sub_input_dim, proj_dim, ff_drop)

        # 3-Stack Attention Layers
        self.layers = nn.ModuleList([
            CrossedDiffAttentionLayer(proj_dim, num_heads, attn_drop, ff_drop)
            for _ in range(num_stacks)
        ])

        self.output_dim = proj_dim * 2   # fp_final ‖ sub_final

    def forward(self, fp: torch.Tensor, sub: torch.Tensor):
        """
        fp  : (B, fp_input_dim)
        sub : (B, sub_input_dim)
        return: (B, output_dim=proj_dim*2)
        """
        fp  = self.fp_proj(fp)     # (B, proj_dim)
        sub = self.sub_proj(sub)   # (B, proj_dim)

        for layer in self.layers:
            fp, sub = layer(fp, sub)

        fused = torch.cat([fp, sub], dim=-1)   # (B, proj_dim*2)
        return fused
