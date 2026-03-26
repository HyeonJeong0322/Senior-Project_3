"""
config.py
=========
공통 하이퍼파라미터 및 경로 설정 파일.
"""

import os

# ─────────────────────────────────────────────
# 1. 경로 설정
# ─────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))          # attention_module/
_MODEL_DIR = os.path.dirname(_THIS_DIR)                           # my_model/
_REPO_ROOT = os.path.abspath(os.path.join(_MODEL_DIR, "..", "..", ".."))  # 프로젝트 루트

DATA_PATH  = os.path.join(_REPO_ROOT, "data", "dilirank.csv")   # 원본 데이터
OUTPUT_DIR = os.path.join(_MODEL_DIR, "outputs")                 # 출력 결과물들 폴더

# 모델 학습에 필요한 임베딩 파일들->지은이가 사용
EMBEDDING_TRAIN_PATH = os.path.join(OUTPUT_DIR, "train_embeddings.npy")
EMBEDDING_TEST_PATH  = os.path.join(OUTPUT_DIR, "test_embeddings.npy")
LABEL_TRAIN_PATH     = os.path.join(OUTPUT_DIR, "train_labels.npy")
LABEL_TEST_PATH      = os.path.join(OUTPUT_DIR, "test_labels.npy")

# CDA가 저장하는 학습된 DL 모델 가중치
MODEL_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "attention_model.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. 데이터 설정
# ─────────────────────────────────────────────
SMILES_COL  = "smiles"       # Dataset.csv 내 SMILES 열 이름 (StackDILI 형식)
LABEL_COL   = "binary_label"        # Dataset.csv 내 라벨 열 이름 (0: non-DILI, 1: DILI)
TEST_SIZE   = 0.2                   # 사용할 테스트 셋 비율, f.e) 0.2 = 20%
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 3. Feature 설정 (Global Path)
#기존 연구들처럼 하나의 분자 지문(Single molecular representation)만 사용할 경우, 
#분자의 부분 구조, 물리화학적 성질, 전역적 토폴로지를 모두 포괄하기에 정보량이 절대적으로 부족함
# 따라서 MACCS와 ECFP를 Concatenation했으나, 노이즈가 급증하여 모델 성능을 떨어뜨릴 수 있기에 나중에 문제 될 경우 확인할 것.

# ─────────────────────────────────────────────
MORGAN_DIM  = 512            # Morgan Fingerprint (ECFP6) 비트 수 (1024 → 512 경량화)
# Global FP 총 차원 = MACCS(167) + Morgan(512) = 679 → RF 후 FP_SELECT_DIM으로 축소

MORGAN_RADIUS = 3            # ECFP6 = radius 3
MORGAN_NBITS  = MORGAN_DIM   # Morgan Generator fpSize 파라미터에 직접 전달

# ─────────────────────────────────────────────
# 4. 피처 설정 (Local Path - Substructure)
# Vocabulary : Bag-of-Words 개념을 분자 구조에 응용한 것.
# data_preprocessing.py의 build_fragment_vocab()에서 참조.
# 레거시 DILIAttentionModel(BoF 기반) 사용 시 필요.
# ─────────────────────────────────────────────
SUB_VOCAB_INIT_DIM = 128     # Vocabulary 최대 상한 (OHE 희소성 제한)

# ─────────────────────────────────────────────
# 5. Attention 모델 하이퍼파라미터
# ─────────────────────────────────────────────
PROJECT_DIM   = 128          # Linear Projection 차원 (64 → 128, 표현력 향상)
NUM_HEADS     = 4            # Multi-head Attention 헤드 수
NUM_STACKS    = 1            # Crossed Diff Attention 반복 횟수
ATTN_DROPOUT  = 0.2          # Attention Dropout 완화 (0.4 → 0.2)
FF_DROPOUT    = 0.3          # Feed-Forward Dropout 완화 (0.5 → 0.3, underfitting 해소)

# ─────────────────────────────────────────────
# 6. 학습 하이퍼파라미터
# ─────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 80
WARMUP_EPOCHS = 5            # 스케줄러 Warm-up용 에폭
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-2         # L2 정규화 대폭 상향 (1e-3 -> 1e-2)
PATIENCE      = 15           # Early Stopping patience (val AUC 기준으로 변경됨)

# MLP Classifier (End-to-End 학습용)
MLP_HIDDEN_DIM = 128         # 64 → 128 (분류기 표현력 향상)
NUM_CLASSES    = 2           # binary: 0(non-DILI) / 1(DILI)

# Feature Selection
FP_SELECT_DIM = 128          # Random Forest로 선택할 최종 상위 FP 특징 수

# ─────────────────────────────────────────────
# 7. GCN 로컬 인코더 하이퍼파라미터
# ─────────────────────────────────────────────
# 원자 초기 피처 수 (원자 번호 one-hot 22 + 전하 + 방향족 + 수소 = 25)
GCN_NODE_FEAT_DIM = 25
# GCN 출력 차원 = Attention Readout 입력 차원 (PROJECT_DIM과 통일)
GCN_OUT_DIM       = 128     # PROJECT_DIM과 동일하게 맞춰 Attention에서 통일된 Key/Value 활용