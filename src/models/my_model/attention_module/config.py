"""
config.py
=========
공통 하이퍼파라미터 및 경로 설정 파일.
기본적으로 이 파일을 import하여 사용할 것.
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

# 모델 학습에 필요한 임베딩 파일들
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
SMILES_COL  = "smiles"       # CSV 내 SMILES 열 이름
LABEL_COL   = "binary_label"        # CSV 내 라벨 열 이름 (0: non-DILI, 1: DILI)
TEST_SIZE   = 0.2                   # 사용할 테스트 셋 비율, f.e) 0.2 = 20%
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 3. Feature 설정 (Global Path)
#기존 연구들처럼 하나의 분자 지문(Single molecular representation)만 사용할 경우, 
#분자의 부분 구조, 물리화학적 성질, 전역적 토폴로지를 모두 포괄하기에 정보량이 절대적으로 부족함
# 따라서 MACCS와 ECFP를 Concatenation했으나, 노이즈가 급증하여 모델 성능을 떨어뜨릴 수 있기에 나중에 문제 될 경우 확인할 것.

# ─────────────────────────────────────────────
MACCS_DIM   = 167            # MACCS Keys (RDKit) 비트 수
MORGAN_DIM  = 1024           # Morgan Fingerprint (ECFP6) 비트 수
FP_DIM      = MACCS_DIM + MORGAN_DIM   # 두 지문 Concatenation 차원 = 1191

MORGAN_RADIUS = 3            # ECFP6 = radius 3
MORGAN_NBITS  = MORGAN_DIM

# ─────────────────────────────────────────────
# 4. 피처 설정 (Local Path - Substructure)
# Vocabulary : Bag-of-Words 개념을 분자 구조에 응용한 것
# 분자마다 쪼개지는 조각의 개수나 종류가 제각각이므로, 공통 기준인 'Vocab'을 정해두고 무조건 그 사전의 크기(예: 1000차원)에 맞춰 숫자를 채워 넣는 것
# Train data 에서만 사용할 것을 유의.
# ─────────────────────────────────────────────
# Substructure 사전(dictionary) 크기: data_preprocessing.py에서 결정됨.
# 이 값은 train split을 기반으로 동적으로 설정되므로 기본값만 지정.
SUB_VOCAB_INIT_DIM = 512     # Vocabulary 최대 상한 (OHE 희소성 제한 용도)

# ─────────────────────────────────────────────
# 5. Attention 모델 하이퍼파라미터
# ─────────────────────────────────────────────
PROJECT_DIM   = 256          # Linear Projection 후 공통 차원
NUM_HEADS     = 8            # Multi-head Attention 헤드 수
NUM_STACKS    = 3            # Crossed Diff Attention 반복 횟수 (1~3 실험 해볼 것.)
ATTN_DROPOUT  = 0.2          # Attention Dropout 비율
FF_DROPOUT    = 0.3          # Feed-Forward Dropout 비율

# ─────────────────────────────────────────────
# 6. 학습 하이퍼파라미터
# ─────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-3         # L2 정규화 (AdamW : 기울기 업데이트 계산이 다 끝난 뒤 마지막에 독립적으로(Decoupled) 가중치를 직접 깎아버리는 방식 차용)
PATIENCE      = 15           # Early Stopping patience

# MLP Classifier (End-to-End 학습용)
MLP_HIDDEN_DIM = 128
NUM_CLASSES    = 2           # binary: 0(non-DILI) / 1(DILI)