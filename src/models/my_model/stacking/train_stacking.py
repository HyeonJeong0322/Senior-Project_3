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

# """
# train_twostream.py (Stacking Ensemble Version)
# =============================================================
# Track 1 (Graph/GCN) + Track 2 (Tabular/Fingerprint) 예측값을 기반으로
# Meta-Learner(Logistic Regression)가 최적의 결합 비율을 학습하는 구조입니다.
# """

# import os
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     roc_auc_score, accuracy_score, matthews_corrcoef, f1_score
# )

# # ── 경로 설정 ────────────────────────────────────────────────
# EMBEDDING_TRAIN_PATH = "/app/src/models/my_model/outputs/train_embeddings.npy"
# EMBEDDING_TEST_PATH  = "/app/src/models/my_model/outputs/test_embeddings.npy"
# FP_TRAIN_PATH = "/app/src/models/my_model/outputs/train_fps.npy"
# FP_TEST_PATH  = "/app/src/models/my_model/outputs/test_fps.npy"
# LABEL_TRAIN_PATH     = "/app/src/models/my_model/outputs/train_labels.npy"
# LABEL_TEST_PATH      = "/app/src/models/my_model/outputs/test_labels.npy"

# RANDOM_SEED = 42

# def _best_threshold_mcc(y_true, y_prob):
#     thresholds = np.linspace(0.1, 0.9, 81)
#     best_mcc, best_thr = -1.0, 0.5
#     for thr in thresholds:
#         mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
#         if mcc > best_mcc:
#             best_mcc, best_thr = mcc, thr
#     return best_thr, best_mcc

# def _print_metrics(label, y_true, y_prob, threshold=0.5):
#     y_pred = (y_prob >= threshold).astype(int)
#     print(f"  {label:<25} "
#           f"AUC={roc_auc_score(y_true, y_prob):.4f}  "
#           f"ACC={accuracy_score(y_true, y_pred):.4f}  "
#           f"MCC={matthews_corrcoef(y_true, y_pred):.4f}  "
#           f"F1={f1_score(y_true, y_pred):.4f}  "
#           f"thr={threshold:.2f}")

# def main():
#     np.random.seed(RANDOM_SEED)

#     print("[1/4] 데이터 로드 및 전처리")
#     y_train = np.load(LABEL_TRAIN_PATH).astype(int)
#     y_test  = np.load(LABEL_TEST_PATH).astype(int)

#     # Track 1 데이터 
#     X_train_emb_raw = np.load(EMBEDDING_TRAIN_PATH)
#     X_test_emb_raw  = np.load(EMBEDDING_TEST_PATH)
#     scaler = StandardScaler()
#     X_train_emb = scaler.fit_transform(X_train_emb_raw)
#     X_test_emb  = scaler.transform(X_test_emb_raw)

#     # Track 2 데이터 
#     X_train_fp = np.load(FP_TRAIN_PATH)
#     X_test_fp  = np.load(FP_TEST_PATH)

#     print(f"   Graph 임베딩 (Track 1) dim: {X_train_emb.shape}")
#     print(f"   Fingerprint (Track 2) dim : {X_train_fp.shape}")

#     # Meta-Learner 학습을 위한 데이터 분할 (Hold-out for Stacking)
#     idx_train, idx_val = train_test_split(
#         np.arange(len(y_train)), test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
#     )

#     # 2. Track 1 학습
#     print("\n[2/4] Track 1 학습 중... (Graph Embeddings -> Linear Models)")
#     lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=0.1, solver="saga")
#     svc = SVC(random_state=RANDOM_SEED, probability=True, C=1.0, kernel='rbf')

#     lr.fit(X_train_emb[idx_train], y_train[idx_train])
#     svc.fit(X_train_emb[idx_train], y_train[idx_train])
    
#     track1_val_proba = (lr.predict_proba(X_train_emb[idx_val])[:, 1] + svc.predict_proba(X_train_emb[idx_val])[:, 1]) / 2

#     lr.fit(X_train_emb, y_train)
#     svc.fit(X_train_emb, y_train)
#     track1_test_proba = (lr.predict_proba(X_test_emb)[:, 1] + svc.predict_proba(X_test_emb)[:, 1]) / 2

#     # 3. Track 2 학습
#     print("[3/4] Track 2 학습 중... (Fingerprints -> Tree Models)")
#     rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED)
#     xgb = XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0)
#     lgbm = LGBMClassifier(n_estimators=300, random_state=RANDOM_SEED, verbose=-1)

#     rf.fit(X_train_fp[idx_train], y_train[idx_train])
#     xgb.fit(X_train_fp[idx_train], y_train[idx_train])
#     lgbm.fit(X_train_fp[idx_train], y_train[idx_train])

#     track2_val_proba = (rf.predict_proba(X_train_fp[idx_val])[:, 1] + xgb.predict_proba(X_train_fp[idx_val])[:, 1] + lgbm.predict_proba(X_train_fp[idx_val])[:, 1]) / 3

#     rf.fit(X_train_fp, y_train)
#     xgb.fit(X_train_fp, y_train)
#     lgbm.fit(X_train_fp, y_train)
#     track2_test_proba = (rf.predict_proba(X_test_fp)[:, 1] + xgb.predict_proba(X_test_fp)[:, 1] + lgbm.predict_proba(X_test_fp)[:, 1]) / 3

#     # 4. Meta-Learner (Stacking) 학습 및 평가
#     print("\n[4/4] Meta-Learner (Stacking) 학습 및 최종 평가")
    
#     # Validation 예측값을 피처로 묶어 메타 모델 학습
#     X_meta_train = np.column_stack((track1_val_proba, track2_val_proba))
#     y_meta_train = y_train[idx_val]

#     meta_model = LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced')
#     meta_model.fit(X_meta_train, y_meta_train)

#     # 심판이 부여한 가중치 확인
#     weights = meta_model.coef_[0]
#     print(f"   [심판의 판정] Track 1 가중치: {weights[0]:.4f} | Track 2 가중치: {weights[1]:.4f}")

#     # 최종 Test 예측
#     X_meta_test = np.column_stack((track1_test_proba, track2_test_proba))
#     stacking_test_proba = meta_model.predict_proba(X_meta_test)[:, 1]

#     # 단순 5:5 평균 (비교용)
#     simple_avg_proba = (track1_test_proba + track2_test_proba) / 2

#     # 메타 모델의 최적 Threshold 탐색
#     stacking_val_proba = meta_model.predict_proba(X_meta_train)[:, 1]
#     best_thr, _ = _best_threshold_mcc(y_meta_train, stacking_val_proba)
#     print(f"   최적 threshold (MCC 기준, val): {best_thr:.2f}")

#     print("\n" + "=" * 70)
#     print("  결과 비교 (Stacking Ensemble)")
#     print("=" * 70)
#     _print_metrics("Track 1 (Graph Only)", y_test, track1_test_proba, 0.5)
#     _print_metrics("Track 2 (Tabular Only)", y_test, track2_test_proba, 0.5)
#     print("-" * 70)
#     _print_metrics("단순 5:5 평균 (비교용)", y_test, simple_avg_proba, 0.5)
#     _print_metrics(f"Stacking 모델 (0.50)", y_test, stacking_test_proba, 0.5)
#     _print_metrics(f"Stacking 모델 ({best_thr:.2f})", y_test, stacking_test_proba, best_thr)
#     print("=" * 70)

# if __name__ == "__main__":
#     main()

# import os
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pickle
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, matthews_corrcoef, 
#     precision_score, recall_score, f1_score, confusion_matrix
# )

# # ── 경로 설정 ────────────────────────────────────────────────
# EMBEDDING_TRAIN_PATH = "/app/src/models/my_model/outputs/train_embeddings.npy"
# EMBEDDING_TEST_PATH  = "/app/src/models/my_model/outputs/test_embeddings.npy"
# FP_TRAIN_PATH        = "/app/src/models/my_model/outputs/train_fps.npy"
# FP_TEST_PATH         = "/app/src/models/my_model/outputs/test_fps.npy"
# LABEL_TRAIN_PATH     = "/app/src/models/my_model/outputs/train_labels.npy"
# LABEL_TEST_PATH      = "/app/src/models/my_model/outputs/test_labels.npy"

# SAVE_PATH = "./Model" # 모델 저장 경로
# os.makedirs(SAVE_PATH, exist_ok=True)

# RANDOM_SEED = 42

# def main():
#     np.random.seed(RANDOM_SEED)

#     # 1. 데이터 로드 및 전처리
#     print("[1/4] 데이터 로드 및 전처리")
#     y_train = np.load(LABEL_TRAIN_PATH).astype(int)
#     y_test  = np.load(LABEL_TEST_PATH).astype(int)

#     # Track 1 (Graph Embeddings) - 스케일링 필요
#     X_train_emb_raw = np.load(EMBEDDING_TRAIN_PATH)
#     X_test_emb_raw  = np.load(EMBEDDING_TEST_PATH)
#     scaler = StandardScaler()
#     X_train_emb = scaler.fit_transform(X_train_emb_raw)
#     X_test_emb  = scaler.transform(X_test_emb_raw)

#     # Track 2 (Fingerprints) - 스케일링 불필요
#     X_train_fp = np.load(FP_TRAIN_PATH)
#     X_test_fp  = np.load(FP_TEST_PATH)

#     # 2. 베이스 모델 정의 및 학습
#     print("\n[2/4] 베이스 모델 학습 및 확률값(Proba) 추출 (StackDILI 원본 방식)")
#     models = {
#         'LR': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=0.1, solver="saga"),
#         'SVC': SVC(random_state=RANDOM_SEED, probability=True, C=1.0, kernel='rbf'),
#         'RF': RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED),
#         'XGB': XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0),
#         'LGBM': LGBMClassifier(n_estimators=300, random_state=RANDOM_SEED, verbose=-1)
#     }

#     prob_features_train = []
#     prob_features_test = []

#     for name, model in models.items():
#         print(f"  -> {name} 학습 중...")
#         # 임베딩(LR, SVC)과 FP(RF, XGB, LGBM)를 구분하여 학습
#         if name in ['LR', 'SVC']:
#             model.fit(X_train_emb, y_train)
#             train_prob = model.predict_proba(X_train_emb)[:, 1]
#             test_prob  = model.predict_proba(X_test_emb)[:, 1]
#         else:
#             model.fit(X_train_fp, y_train)
#             train_prob = model.predict_proba(X_train_fp)[:, 1]
#             test_prob  = model.predict_proba(X_test_fp)[:, 1]

#         # 논문 코드처럼 개별 모델 저장
#         with open(os.path.join(SAVE_PATH, f"best_model_{name}.pkl"), 'wb') as f:
#             pickle.dump(model, f)

#         prob_features_train.append(train_prob)
#         prob_features_test.append(test_prob)

#     # 메타 피처 생성 (Column Stack)
#     X_train_prob = np.column_stack(prob_features_train)
#     X_test_prob  = np.column_stack(prob_features_test)

#     # 3. 메타 모델 (Stacking - ExtraTrees) 학습 및 튜닝 (10회 반복)
#     print("\n[3/4] Stacking 모델 (ExtraTreesClassifier) 학습 (10회 반복 탐색)")
#     stacking_model = ExtraTreesClassifier(random_state=RANDOM_SEED)

#     best_stacking_auc = -np.inf
#     best_stacking_model = None

#     for i in range(10):
#         # 논문 코드와 동일하게 매 훈련마다 새로운 랜덤 시드 부여
#         current_seed = np.random.randint(0, 10000)
#         stacking_model.set_params(random_state=current_seed)
        
#         stacking_model.fit(X_train_prob, y_train)
        
#         y_pred = stacking_model.predict(X_test_prob)
#         y_prob = stacking_model.predict_proba(X_test_prob)[:, 1]
        
#         acc = accuracy_score(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_prob)
#         mcc = matthews_corrcoef(y_test, y_pred)
        
#         print(f"  {i + 1}번째 Stacking 모델 결과: ACC: {acc:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}")
        
#         if auc > best_stacking_auc:
#             best_stacking_auc = auc
#             best_stacking_model = pickle.dumps(stacking_model)

#     stacking_model_file = os.path.join(SAVE_PATH, "best_model_Stacking.pkl")
#     with open(stacking_model_file, 'wb') as f:
#         f.write(best_stacking_model)
#     print(f"\n[최적 Stacking 모델 저장 완료] AUC: {best_stacking_auc:.4f}")

#     # 4. 최종 결과 출력 (논문 Evaluation 코드 통합)
#     print("\n[4/4] 개별 모델 및 최종 Stacking 모델 성능 비교")
#     model_names = list(models.keys()) + ['Stacking']
    
#     # 평가를 위해 최적의 Stacking 모델 다시 로드
#     with open(stacking_model_file, 'rb') as f:
#         best_stacking_clf = pickle.loads(f.read())

#     print("=" * 70)
#     for model_name in model_names:
#         if model_name == 'Stacking':
#             y_pred = best_stacking_clf.predict(X_test_prob)
#             y_prob = best_stacking_clf.predict_proba(X_test_prob)[:, 1]
#         else:
#             with open(os.path.join(SAVE_PATH, f"best_model_{model_name}.pkl"), 'rb') as f:
#                 clf = pickle.loads(f.read())
            
#             if model_name in ['LR', 'SVC']:
#                 y_pred = clf.predict(X_test_emb)
#                 y_prob = clf.predict_proba(X_test_emb)[:, 1]
#             else:
#                 y_pred = clf.predict(X_test_fp)
#                 y_prob = clf.predict_proba(X_test_fp)[:, 1]

#         acc = accuracy_score(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_prob)
#         precision = precision_score(y_test, y_pred, zero_division=0)
#         recall = recall_score(y_test, y_pred, zero_division=0)
#         f1 = f1_score(y_test, y_pred, zero_division=0)
#         tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

#         print(f"[{model_name}]")
#         print(f"  Accuracy: {acc:.4f} | Sensitivity(Recall): {recall:.4f} | Specificity: {specificity:.4f}")
#         print(f"  Precision: {precision:.4f} | F1 Score: {f1:.4f} | AUC: {auc:.4f}\n")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef, 
    precision_score, recall_score, f1_score, confusion_matrix
)

# ── 경로 설정 ────────────────────────────────────────────────
EMBEDDING_TRAIN_PATH = "/app/src/models/my_model/outputs/train_embeddings.npy"
EMBEDDING_TEST_PATH  = "/app/src/models/my_model/outputs/test_embeddings.npy"
FP_TRAIN_PATH        = "/app/src/models/my_model/outputs/train_fps.npy"
FP_TEST_PATH         = "/app/src/models/my_model/outputs/test_fps.npy"
LABEL_TRAIN_PATH     = "/app/src/models/my_model/outputs/train_labels.npy"
LABEL_TEST_PATH      = "/app/src/models/my_model/outputs/test_labels.npy"

SAVE_PATH = "./Model" 
os.makedirs(SAVE_PATH, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5 # 5-Fold 교차 검증

def main():
    np.random.seed(RANDOM_SEED)

    # 1. 데이터 로드 및 전처리
    print("[1/4] 데이터 로드 및 전처리")
    y_train = np.load(LABEL_TRAIN_PATH).astype(int)
    y_test  = np.load(LABEL_TEST_PATH).astype(int)

    # Track 1 (Graph Embeddings) - 스케일링
    X_train_emb_raw = np.load(EMBEDDING_TRAIN_PATH)
    X_test_emb_raw  = np.load(EMBEDDING_TEST_PATH)
    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb_raw)
    X_test_emb  = scaler.transform(X_test_emb_raw)

    # Track 2 (Fingerprints)
    X_train_fp = np.load(FP_TRAIN_PATH)
    X_test_fp  = np.load(FP_TEST_PATH)

    # 2. 베이스 모델 정의
    models = {
        'LR': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=0.1, solver="saga"),
        'SVC': SVC(random_state=RANDOM_SEED, probability=True, C=1.0, kernel='rbf'),
        'RF': RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED),
        'XGB': XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0),
        'LGBM': LGBMClassifier(n_estimators=300, random_state=RANDOM_SEED, verbose=-1)
    }

    # OOF 결과를 저장할 빈 배열 (Train용, Test용)
    oof_train = np.zeros((len(y_train), len(models)))
    oof_test  = np.zeros((len(y_test), len(models)))

    # 3. K-Fold OOF 예측 (데이터 누수 방지 핵심 로직)
    print(f"\n[2/4] {N_SPLITS}-Fold OOF 기반 베이스 모델 학습 및 예측")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    for col_idx, (name, model) in enumerate(models.items()):
        print(f"  -> {name} OOF 추출 중...")
        
        # 각 모델별 Test 예측값을 Fold마다 누적할 임시 배열
        test_preds = np.zeros((len(y_test), N_SPLITS))
        
        # 사용할 피처 선택
        X_tr_curr = X_train_emb if name in ['LR', 'SVC'] else X_train_fp
        X_te_curr = X_test_emb if name in ['LR', 'SVC'] else X_test_fp

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_tr_curr, y_train)):
            # Fold별 Train/Val 나누기
            X_fold_train, y_fold_train = X_tr_curr[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val     = X_tr_curr[val_idx], y_train[val_idx]

            # 모델 학습
            model.fit(X_fold_train, y_fold_train)

            # 검증 데이터(Val) 예측 -> OOF 배열에 채워넣기 (안 본 데이터 예측)
            oof_train[val_idx, col_idx] = model.predict_proba(X_fold_val)[:, 1]

            # 실제 Test 데이터 예측 -> 배열에 누적
            test_preds[:, fold_idx] = model.predict_proba(X_te_curr)[:, 1]

        # Test 데이터는 5번 예측한 값의 '평균'을 사용
        oof_test[:, col_idx] = test_preds.mean(axis=1)
        
        # 전체 데이터로 최종 베이스 모델 1번 더 학습하여 저장 (선택 사항)
        model.fit(X_tr_curr, y_train)
        with open(os.path.join(SAVE_PATH, f"best_model_{name}_OOF.pkl"), 'wb') as f:
            pickle.dump(model, f)

    # 4. 메타 모델 (Stacking - ExtraTrees) 학습 및 튜닝
    print("\n[3/4] 메타 모델 (ExtraTreesClassifier) 학습")
    # 누수가 없으므로, 메타 모델은 OOF_train을 보고 객관적인 가중치를 학습합니다.
    stacking_model = ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_SEED)
    stacking_model.fit(oof_train, y_train)

    with open(os.path.join(SAVE_PATH, "best_model_Stacking_OOF.pkl"), 'wb') as f:
        pickle.dump(stacking_model, f)

    # 5. 최종 결과 평가
    print("\n[4/4] 최종 OOF Stacking 성능 평가")
    
    y_pred = stacking_model.predict(oof_test)
    y_prob = stacking_model.predict_proba(oof_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("=" * 70)
    print("[OOF 정석 Stacking]")
    print(f"  Accuracy: {acc:.4f} | Sensitivity(Recall): {recall:.4f} | Specificity: {specificity:.4f}")
    print(f"  Precision: {precision:.4f} | F1 Score: {f1:.4f} | AUC: {auc:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()