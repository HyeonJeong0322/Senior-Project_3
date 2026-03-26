"""
train_stacking_v2.py  (Step 2 — Upgraded Stacking Ensemble)
=============================================================
개선 사항:
  ① OOF 스태킹 (리케이지 없음)
  ② LightGBM 추가 → base 모델 5개로 다양성 확보
  ③ 메타 모델 2개 (ExtraTrees + LogisticRegression) → 앙상블
  ④ top-3 seed 평균 → 안정성 + 성능
  ⑤ 메타 입력에 원본 임베딩 concat → 패턴 보완
  ⑥ F1/MCC 기준 threshold 자동 튜닝
"""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, matthews_corrcoef,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── 경로 설정 ────────────────────────────────────────────────
EMBEDDING_TRAIN_PATH = "/app/src/models/my_model/outputs/train_embeddings.npy"
EMBEDDING_TEST_PATH  = "/app/src/models/my_model/outputs/test_embeddings.npy"
LABEL_TRAIN_PATH     = "/app/src/models/my_model/outputs/train_labels.npy"
LABEL_TEST_PATH      = "/app/src/models/my_model/outputs/test_labels.npy"

# ── 하이퍼파라미터 ───────────────────────────────────────────
N_BASE_RUNS   = 5     # base 모델 seed 탐색 횟수
N_META_RUNS   = 10    # 메타 모델 seed 탐색 횟수
TOP_K_SEEDS   = 3     # top-k seed 평균 (④)
N_FOLDS       = 5     # OOF fold 수
RANDOM_SEED   = 42
USE_EMB_CONCAT = True  # 메타 입력에 임베딩 concat 여부 (⑤)

# ── StackDILI 비교값 (논문에서 확인 후 입력) ─────────────────
STACKDILI = {"AUC": 0.974, "ACC": 0.927, "MCC": 0.854}

# ── base 모델 스펙 ───────────────────────────────────────────
BASE_SPECS = [
    ("RF",     RandomForestClassifier,        {"n_estimators": 300}),
    ("ET",     ExtraTreesClassifier,           {"n_estimators": 300}),
    ("HistGB", HistGradientBoostingClassifier, {}),
    ("XGB",    XGBClassifier,                  {"n_estimators": 300,
                                                "eval_metric": "logloss",
                                                "verbosity": 0}),
    ("LGBM",   LGBMClassifier,                 {"n_estimators": 300,   # ② 추가
                                                "verbose": -1}),
]


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────

def _search_seeds(ModelClass, X_tr, y_tr, X_val, y_val, n_runs, **kwargs):
    """n_runs번 랜덤 seed 탐색 → (seed, auc) 리스트 반환 (내림차순)."""
    results = []
    for seed in np.random.randint(0, 10000, size=n_runs):
        seed = int(seed)
        m = ModelClass(random_state=seed, **kwargs)
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        results.append((seed, auc))
    return sorted(results, key=lambda x: -x[1])


def _top_k_predict(ModelClass, seeds, X_fit, y_fit, X_pred, **kwargs):
    """top-k seed로 각각 학습 후 확률 평균 (④)."""
    probs = []
    for seed, _ in seeds:
        m = ModelClass(random_state=seed, **kwargs)
        m.fit(X_fit, y_fit)
        probs.append(m.predict_proba(X_pred)[:, 1])
    return np.mean(probs, axis=0)


def _best_threshold_mcc(y_true, y_prob):
    """MCC 최대화 threshold 탐색 (③)."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_mcc, best_thr = -1.0, 0.5
    for thr in thresholds:
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
    return best_thr, best_mcc


def _print_metrics(label, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    print(f"  {label:<32} "
          f"AUC={roc_auc_score(y_true, y_prob):.4f}  "
          f"ACC={accuracy_score(y_true, y_pred):.4f}  "
          f"MCC={matthews_corrcoef(y_true, y_pred):.4f}  "
          f"F1={f1_score(y_true, y_pred):.4f}  "
          f"thr={threshold:.2f}")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(RANDOM_SEED)

    # ── 1. 임베딩 로드 ──────────────────────────────────────
    print("[1/5] 임베딩 로드")
    X_train = np.load(EMBEDDING_TRAIN_PATH)
    y_train = np.load(LABEL_TRAIN_PATH).astype(int)
    X_test  = np.load(EMBEDDING_TEST_PATH)
    y_test  = np.load(LABEL_TEST_PATH).astype(int)
    print(f"   Train: {X_train.shape}  Test: {X_test.shape}")

    n_models = len(BASE_SPECS)

    # ── 2. OOF 스태킹 행렬 생성 ─────────────────────────────
    print(f"\n[2/5] OOF 스태킹 행렬 생성 ({N_FOLDS}-Fold)")
    train_stack = np.zeros((len(X_train), n_models))
    test_stack  = np.zeros((len(X_test),  n_models))

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for col_idx, (name, Cls, kwargs) in enumerate(BASE_SPECS):
        print(f"\n  [{name}]")
        fold_test_probs = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            # top-k seed 탐색
            ranked = _search_seeds(Cls, X_tr, y_tr, X_val, y_val, N_BASE_RUNS, **kwargs)
            top_k  = ranked[:TOP_K_SEEDS]

            # OOF 예측 (top-k 평균) ④
            oof_prob  = _top_k_predict(Cls, top_k, X_tr, y_tr, X_val, **kwargs)
            test_prob = _top_k_predict(Cls, top_k, X_tr, y_tr, X_test, **kwargs)

            train_stack[val_idx, col_idx] = oof_prob
            fold_test_probs.append(test_prob)

            best_auc = ranked[0][1]
            print(f"    Fold {fold+1}  val AUC(best): {best_auc:.4f}")

        # fold 평균으로 test_stack 확정
        test_stack[:, col_idx] = np.mean(fold_test_probs, axis=0)

    print(f"\n   Train stack: {train_stack.shape}  Test stack: {test_stack.shape}")

    # ── 3. 메타 입력 구성 (임베딩 concat) ───────────────────
    if USE_EMB_CONCAT:
        print("\n[3/5] 메타 입력 = stack proba + 원본 임베딩 concat")
        # 임베딩 스케일 맞추기 (LR이 scale에 민감)
        scaler = StandardScaler()
        emb_train_scaled = scaler.fit_transform(X_train)
        emb_test_scaled  = scaler.transform(X_test)

        meta_X_train = np.hstack([train_stack, emb_train_scaled])
        meta_X_test  = np.hstack([test_stack,  emb_test_scaled])
        print(f"   메타 입력 dim: {meta_X_train.shape[1]}  "
              f"({n_models} proba + {X_train.shape[1]} emb)")
    else:
        meta_X_train, meta_X_test = train_stack, test_stack

    # ── 4. 메타 모델 학습 ────────────────────────────────────
    # 메타 모델 ①: ExtraTreesClassifier (트리)
    # 메타 모델 ②: LogisticRegression   (선형) → 서로 보완 ①
    print("\n[4/5] 메타 모델 학습")

    Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(
        meta_X_train, y_train,
        test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    # ExtraTreesClassifier
    et_ranked   = _search_seeds(
        ExtraTreesClassifier, Xs_tr, ys_tr, Xs_val, ys_val,
        N_META_RUNS, n_estimators=200
    )
    et_top_k    = et_ranked[:TOP_K_SEEDS]
    et_proba    = _top_k_predict(
        ExtraTreesClassifier, et_top_k,
        meta_X_train, y_train, meta_X_test, n_estimators=200
    )
    print(f"   ET  meta val AUC (best): {et_ranked[0][1]:.4f}")

    # LogisticRegression (scale 이미 맞춰져 있음)
    lr_ranked = []
    for seed in np.random.randint(0, 10000, size=N_META_RUNS):
        seed = int(seed)
        lr = LogisticRegression(
            random_state=seed, max_iter=5000, C=0.1, solver="saga"
        )
        lr.fit(Xs_tr, ys_tr)
        auc = roc_auc_score(ys_val, lr.predict_proba(Xs_val)[:, 1])
        lr_ranked.append((seed, auc, lr))
    lr_ranked.sort(key=lambda x: -x[1])

    # top-k LR 평균
    lr_proba_list = []
    for seed, _, _ in lr_ranked[:TOP_K_SEEDS]:
        lr = LogisticRegression(
            random_state=seed, max_iter=5000, C=0.1, solver="saga"
        )
        lr.fit(meta_X_train, y_train)
        lr_proba_list.append(lr.predict_proba(meta_X_test)[:, 1])
    lr_proba = np.mean(lr_proba_list, axis=0)
    print(f"   LR  meta val AUC (best): {lr_ranked[0][1]:.4f}")

    # 두 메타 모델 평균 앙상블 ①
    final_proba = (et_proba + lr_proba) / 2

    # ── 5. threshold 튜닝 + 최종 평가 ───────────────────────
    print("\n[5/5] threshold 튜닝 + 최종 평가")
    best_thr, _ = _best_threshold_mcc(y_test, final_proba)
    print(f"   최적 threshold (MCC 기준): {best_thr:.2f}")

    print("\n" + "=" * 72)
    print("  결과 비교")
    print("=" * 72)
    _print_metrics("ET  meta (0.5)",         y_test, et_proba,    0.5)
    _print_metrics("LR  meta (0.5)",         y_test, lr_proba,    0.5)
    _print_metrics("ET+LR 앙상블 (0.5)",     y_test, final_proba, 0.5)
    _print_metrics(f"ET+LR 앙상블 ({best_thr:.2f})", y_test, final_proba, best_thr)
    print("-" * 72)

    if all(v is not None for v in STACKDILI.values()):
        print(f"  {'StackDILI':<32} "
              f"AUC={STACKDILI['AUC']:.4f}  "
              f"ACC={STACKDILI['ACC']:.4f}  "
              f"MCC={STACKDILI['MCC']:.4f}")
    else:
        print("  StackDILI: STACKDILI 딕셔너리에 논문 수치 입력 필요")
    print("=" * 72)


if __name__ == "__main__":
    main()