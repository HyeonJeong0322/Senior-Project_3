import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import config
from data_preprocessing import load_all_data

def main():
    print("=== Raw Data (Fingerprints) vs XGBoost Baseline ===")
    print("1. 데이터 전처리 파이프라인 호출 중...")
    fp_all, frags_all, graphs_all, labels = load_all_data()
    n_samples = len(labels)
    
    print(f"  전체 획득 샘플 수: {n_samples}")
    
    # DL 모델에서 사용했던 것과 완벽히 동일한 시드 번호와 분할 비율을 사용해
    # 완전하게 공정한 Train(80%) / Test(20%) Hold-out 분할 수행
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=config.TEST_SIZE, stratify=labels, random_state=config.RANDOM_SEED
    )
    
    # 딥러닝 임베딩(128차원) 대신 원본 지문 Feature 모두 사용 (679차원 등)
    X_train = fp_all[train_idx]
    y_train = labels[train_idx]
    
    X_test = fp_all[test_idx]
    y_test = labels[test_idx]
    
    print(f"  Hold-out 분할 완료 -> Train: {len(y_train)}건 | Test: {len(y_test)}건")
    print(f"  입력 피처 차원: {X_train.shape[1]} 차원 (Morgan + MACCS 순수 원본 지문)")
    
    print("\n2. XGBoost 베이스라인 학습 시작 (딥러닝 GCN 미사용)...")
    xgb = XGBClassifier(eval_metric='logloss', random_state=config.RANDOM_SEED)
    xgb.fit(X_train, y_train)
    
    print("\n3. 모델 성능 평가 진행 중...")
    preds = xgb.predict(X_test)
    probs = xgb.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    with open("baseline_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        
    print("\n" + "="*55)
    print("📊 [원본 데이터(Ablation Study) 통제군 결과]")
    print(f"👉 Raw FP + XGBoost Accuracy : {acc:.4f}")
    print(f"👉 Raw FP + XGBoost ROC-AUC  : {auc:.4f}")
    print("="*55)
    print("\n비고: 위 결과와 128차원 임베딩 기반 'eval_and_report.py'의 결과를 대조하여")
    print("     Differential Attention의 피처 압축 및 노이즈 제거 능력을 논문에 서술해 주세요.\n")

if __name__ == "__main__":
    main()
