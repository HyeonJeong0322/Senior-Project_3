"""
model_builder.py  (Person A — Step 2 아키텍처)
===============================================
전체 DL 모델 = CrossedDiffAttentionStack + MLP Classifier

  ① End-to-End 학습: CrossEntropyLoss로 Attention 블록을 포함한
     전체 딥러닝 모델을 역전파로 먼저 사전 학습
  ② 임베딩 추출 모드: MLP 직전의 내부 표현(embedding)을 .npy로 저장
     → 이후 (XGBoost 등의 DILI 학습 모델이 임베딩을 입력값으로 받아 최종 분류 수행
  ③ Label Smoothing: 소규모 데이터셋(~1,000)의 과적합 완화
"""

import torch
import torch.nn as nn

from .attention_module import CrossedDiffAttentionStack
from . import config


# ─────────────────────────────────────────────────────────────
# MLP Classifier Head
# ─────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Attention Block 출력(2*PROJECT_DIM) → 이진 분류
    학습용으로 Attention과 함께 End-to-End 역전파.
    임베딩 추출 시에는 마지막 Linear 직전 텐서를 반환.
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
        self.fc2  = nn.Linear(hidden_dim, num_classes)   # 최종 분류 레이어

    def forward(self, x: torch.Tensor,
                return_embedding: bool = False):
        """
        return_embedding=True → MLP 내부 표현(hidden_dim) 반환
                         False → 로짓(num_classes) 반환
        """
        h = self.drop(self.act(self.norm(self.fc1(x))))
        if return_embedding:
            return h                        # (B, hidden_dim) → Person B용
        return self.fc2(h)                  # (B, num_classes)


# ─────────────────────────────────────────────────────────────
# 통합 모델
# ─────────────────────────────────────────────────────────────

class DILIAttentionModel(nn.Module):
    """
    전체 Pipeline:
      fp_input (B, 1191) ──────────────────────────────────┐
                                                            ├→ CrossedDiffAttentionStack
      sub_input (B, vocab_dim) ───────────────────────────┘
                     ↓ (B, 512)
              MLPClassifier
                     ↓
         logits (B, 2) or embedding (B, 128)
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
            input_dim   = self.attention_stack.output_dim,  # 512
            hidden_dim  = config.MLP_HIDDEN_DIM,
            num_classes = config.NUM_CLASSES,
            dropout     = config.FF_DROPOUT,
        )

    def forward(self, fp: torch.Tensor, sub: torch.Tensor,
                return_embedding: bool = False):
        """
        fp  : (B, fp_dim)
        sub : (B, sub_dim)
        return_embedding=True  → (B, MLP_HIDDEN_DIM) 추출 임베딩
        return_embedding=False → (B, NUM_CLASSES) 로짓
        """
        fused  = self.attention_stack(fp, sub)           # (B, 512)
        output = self.classifier(fused, return_embedding)
        return output


# ─────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def build_model(fp_dim: int, sub_dim: int,
                device: torch.device = None) -> DILIAttentionModel:
    """모델 인스턴스 생성 및 디바이스 이동."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DILIAttentionModel(fp_dim=fp_dim, sub_dim=sub_dim)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[모델 빌드] fp_dim={fp_dim}, sub_dim={sub_dim}")
    print(f"           총 학습 파라미터: {total_params:,}")
    return model


if __name__ == "__main__":
    # 빠른 shape 확인 테스트
    device = torch.device("cpu")
    fp_dim, sub_dim = config.FP_DIM, 256   # sub_dim은 vocab에 따라 변동
    model = build_model(fp_dim, sub_dim, device)

    dummy_fp  = torch.randn(4, fp_dim)
    dummy_sub = torch.randn(4, sub_dim)

    logits = model(dummy_fp, dummy_sub)
    emb    = model(dummy_fp, dummy_sub, return_embedding=True)
    print("logits shape   :", logits.shape)    # (4, 2)
    print("embedding shape:", emb.shape)       # (4, 128)
