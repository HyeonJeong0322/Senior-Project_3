# """
# train_baseline.py (GCN Embedding -> LR & SVM)
# =============================================================
# 개선 사항:
#   ① GCN 임베딩에 적합하지 않은 Tree 기반 모델(RF, XGB 등) 제거
#   ② 연속형 고차원 데이터에 강한 Logistic Regression과 SVC(SVM) 적용
#   ③ 데이터 스케일링(StandardScaler) 필수 적용
#   ④ 불필요한 스태킹 구조를 제거하고 빠르고 직관적인 투표(Voting) 앙상블 사용
# """

# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     roc_auc_score, accuracy_score, matthews_corrcoef, f1_score
# )

# # ── 경로 설정 ────────────────────────────────────────────────
# EMBEDDING_TRAIN_PATH = "/app/src/models/my_model/outputs/train_embeddings.npy"
# EMBEDDING_TEST_PATH  = "/app/src/models/my_model/outputs/test_embeddings.npy"
# LABEL_TRAIN_PATH     = "/app/src/models/my_model/outputs/train_labels.npy"
# LABEL_TEST_PATH      = "/app/src/models/my_model/outputs/test_labels.npy"

# RANDOM_SEED = 42

# def _best_threshold_mcc(y_true, y_prob):
#     """MCC 최대화 threshold 탐색"""
#     thresholds = np.linspace(0.1, 0.9, 81)
#     best_mcc, best_thr = -1.0, 0.5
#     for thr in thresholds:
#         mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
#         if mcc > best_mcc:
#             best_mcc, best_thr = mcc, thr
#     return best_thr, best_mcc

# def _print_metrics(label, y_true, y_prob, threshold=0.5):
#     y_pred = (y_prob >= threshold).astype(int)
#     print(f"  {label:<22} "
#           f"AUC={roc_auc_score(y_true, y_prob):.4f}  "
#           f"ACC={accuracy_score(y_true, y_pred):.4f}  "
#           f"MCC={matthews_corrcoef(y_true, y_pred):.4f}  "
#           f"F1={f1_score(y_true, y_pred):.4f}  "
#           f"thr={threshold:.2f}")

# def main():
#     np.random.seed(RANDOM_SEED)

#     # 1. 임베딩 로드
#     print("[1/4] 임베딩 로드 및 스케일링")
#     X_train_raw = np.load(EMBEDDING_TRAIN_PATH)
#     y_train = np.load(LABEL_TRAIN_PATH).astype(int)
#     X_test_raw  = np.load(EMBEDDING_TEST_PATH)
#     y_test  = np.load(LABEL_TEST_PATH).astype(int)
    
#     # LR과 SVM은 스케일에 매우 민감하므로 필수
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train_raw)
#     X_test  = scaler.transform(X_test_raw)
    
#     print(f"   Train: {X_train.shape}  Test: {X_test.shape}")

#     # Threshold 튜닝을 위한 Validation 분할
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
#     )

#     # 2. 모델 학습
#     print("\n[2/4] 선형 기반 분류기 학습 (Logistic Regression & SVM)")
    
#     # 모델 1: Logistic Regression
#     lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=0.1, solver="saga")
#     lr.fit(X_tr, y_tr)
#     lr_val_proba = lr.predict_proba(X_val)[:, 1]
    
#     # 모델 2: SVC (Support Vector Machine)
#     svc = SVC(random_state=RANDOM_SEED, probability=True, C=1.0, kernel='rbf')
#     svc.fit(X_tr, y_tr)
#     svc_val_proba = svc.predict_proba(X_val)[:, 1]

#     # Validation 기반 앙상블 및 최적 임계값 찾기
#     val_final_proba = (lr_val_proba + svc_val_proba) / 2
#     best_thr, _ = _best_threshold_mcc(y_val, val_final_proba)
#     print(f"   최적 threshold (MCC 기준, val): {best_thr:.2f}")

#     # 3. 전체 Train 데이터로 최종 재학습 (성능 극대화)
#     print("\n[3/4] 전체 Train 데이터로 재학습 중...")
#     lr.fit(X_train, y_train)
#     svc.fit(X_train, y_train)

#     lr_test_proba = lr.predict_proba(X_test)[:, 1]
#     svc_test_proba = svc.predict_proba(X_test)[:, 1]
#     final_test_proba = (lr_test_proba + svc_test_proba) / 2

#     # 4. 평가
#     print("\n[4/4] 최종 평가")
#     print("=" * 65)
#     print("  결과 비교 (GCN Embedding Baseline)")
#     print("=" * 65)
#     _print_metrics("Logistic Reg (0.5)", y_test, lr_test_proba, 0.5)
#     _print_metrics("SVM (0.5)", y_test, svc_test_proba, 0.5)
#     _print_metrics("LR + SVM 앙상블 (0.5)", y_test, final_test_proba, 0.5)
#     _print_metrics(f"LR + SVM ({best_thr:.2f})", y_test, final_test_proba, best_thr)
#     print("=" * 65)

# if __name__ == "__main__":
#     main()

"""
train_twostream.py (Stacking Ensemble Version)
=============================================================
Track 1 (Graph/GCN) + Track 2 (Tabular/Fingerprint) 예측값을 기반으로
Meta-Learner(Logistic Regression)가 최적의 결합 비율을 학습하는 구조입니다.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, matthews_corrcoef, f1_score
)

# ── 경로 설정 ────────────────────────────────────────────────
EMBEDDING_TRAIN_PATH = "/app/src/models/my_model/outputs/train_embeddings.npy"
EMBEDDING_TEST_PATH  = "/app/src/models/my_model/outputs/test_embeddings.npy"
FP_TRAIN_PATH = "/app/src/models/my_model/outputs/train_fps.npy"
FP_TEST_PATH  = "/app/src/models/my_model/outputs/test_fps.npy"
LABEL_TRAIN_PATH     = "/app/src/models/my_model/outputs/train_labels.npy"
LABEL_TEST_PATH      = "/app/src/models/my_model/outputs/test_labels.npy"

RANDOM_SEED = 42

def _best_threshold_mcc(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_mcc, best_thr = -1.0, 0.5
    for thr in thresholds:
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
    return best_thr, best_mcc

def _print_metrics(label, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    print(f"  {label:<25} "
          f"AUC={roc_auc_score(y_true, y_prob):.4f}  "
          f"ACC={accuracy_score(y_true, y_pred):.4f}  "
          f"MCC={matthews_corrcoef(y_true, y_pred):.4f}  "
          f"F1={f1_score(y_true, y_pred):.4f}  "
          f"thr={threshold:.2f}")

def main():
    np.random.seed(RANDOM_SEED)

    print("[1/4] 데이터 로드 및 전처리")
    y_train = np.load(LABEL_TRAIN_PATH).astype(int)
    y_test  = np.load(LABEL_TEST_PATH).astype(int)

    # Track 1 데이터 
    X_train_emb_raw = np.load(EMBEDDING_TRAIN_PATH)
    X_test_emb_raw  = np.load(EMBEDDING_TEST_PATH)
    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb_raw)
    X_test_emb  = scaler.transform(X_test_emb_raw)

    # Track 2 데이터 
    X_train_fp = np.load(FP_TRAIN_PATH)
    X_test_fp  = np.load(FP_TEST_PATH)

    print(f"   Graph 임베딩 (Track 1) dim: {X_train_emb.shape}")
    print(f"   Fingerprint (Track 2) dim : {X_train_fp.shape}")

    # Meta-Learner 학습을 위한 데이터 분할 (Hold-out for Stacking)
    idx_train, idx_val = train_test_split(
        np.arange(len(y_train)), test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    # 2. Track 1 학습
    print("\n[2/4] Track 1 학습 중... (Graph Embeddings -> Linear Models)")
    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=0.1, solver="saga")
    svc = SVC(random_state=RANDOM_SEED, probability=True, C=1.0, kernel='rbf')

    lr.fit(X_train_emb[idx_train], y_train[idx_train])
    svc.fit(X_train_emb[idx_train], y_train[idx_train])
    
    track1_val_proba = (lr.predict_proba(X_train_emb[idx_val])[:, 1] + svc.predict_proba(X_train_emb[idx_val])[:, 1]) / 2

    lr.fit(X_train_emb, y_train)
    svc.fit(X_train_emb, y_train)
    track1_test_proba = (lr.predict_proba(X_test_emb)[:, 1] + svc.predict_proba(X_test_emb)[:, 1]) / 2

    # 3. Track 2 학습
    print("[3/4] Track 2 학습 중... (Fingerprints -> Tree Models)")
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED)
    xgb = XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0)
    lgbm = LGBMClassifier(n_estimators=300, random_state=RANDOM_SEED, verbose=-1)

    rf.fit(X_train_fp[idx_train], y_train[idx_train])
    xgb.fit(X_train_fp[idx_train], y_train[idx_train])
    lgbm.fit(X_train_fp[idx_train], y_train[idx_train])

    track2_val_proba = (rf.predict_proba(X_train_fp[idx_val])[:, 1] + xgb.predict_proba(X_train_fp[idx_val])[:, 1] + lgbm.predict_proba(X_train_fp[idx_val])[:, 1]) / 3

    rf.fit(X_train_fp, y_train)
    xgb.fit(X_train_fp, y_train)
    lgbm.fit(X_train_fp, y_train)
    track2_test_proba = (rf.predict_proba(X_test_fp)[:, 1] + xgb.predict_proba(X_test_fp)[:, 1] + lgbm.predict_proba(X_test_fp)[:, 1]) / 3

    # 4. Meta-Learner (Stacking) 학습 및 평가
    print("\n[4/4] Meta-Learner (Stacking) 학습 및 최종 평가")
    
    # Validation 예측값을 피처로 묶어 메타 모델 학습
    X_meta_train = np.column_stack((track1_val_proba, track2_val_proba))
    y_meta_train = y_train[idx_val]

    meta_model = LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced')
    meta_model.fit(X_meta_train, y_meta_train)

    # 심판이 부여한 가중치 확인
    weights = meta_model.coef_[0]
    print(f"   [심판의 판정] Track 1 가중치: {weights[0]:.4f} | Track 2 가중치: {weights[1]:.4f}")

    # 최종 Test 예측
    X_meta_test = np.column_stack((track1_test_proba, track2_test_proba))
    stacking_test_proba = meta_model.predict_proba(X_meta_test)[:, 1]

    # 단순 5:5 평균 (비교용)
    simple_avg_proba = (track1_test_proba + track2_test_proba) / 2

    # 메타 모델의 최적 Threshold 탐색
    stacking_val_proba = meta_model.predict_proba(X_meta_train)[:, 1]
    best_thr, _ = _best_threshold_mcc(y_meta_train, stacking_val_proba)
    print(f"   최적 threshold (MCC 기준, val): {best_thr:.2f}")

    print("\n" + "=" * 70)
    print("  결과 비교 (Stacking Ensemble)")
    print("=" * 70)
    _print_metrics("Track 1 (Graph Only)", y_test, track1_test_proba, 0.5)
    _print_metrics("Track 2 (Tabular Only)", y_test, track2_test_proba, 0.5)
    print("-" * 70)
    _print_metrics("단순 5:5 평균 (비교용)", y_test, simple_avg_proba, 0.5)
    _print_metrics(f"Stacking 모델 (0.50)", y_test, stacking_test_proba, 0.5)
    _print_metrics(f"Stacking 모델 ({best_thr:.2f})", y_test, stacking_test_proba, best_thr)
    print("=" * 70)

if __name__ == "__main__":
    main()