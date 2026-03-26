"""
train_dl_model.py  (— End-to-End 학습 및 임베딩 추출)
=================================================================
[신규] DILIGCNModel (1-Layer GCN + Differential Attention Readout) 기반 학습 파이프라인

  ① End-to-End K-Fold 교차 검증:
       Data Leakage 방지를 위해 전체 데이터를 80% (Train/Val) / 20% (Hold-out Test)로 분할 후 K-Fold 진행
  ② Feature Selection: Random Forest로 FP 차원 축소
  ③ Early Stopping: Val Loss 기준
  ④ Warm-up Scheduler: LinearLR + CosineAnnealingLR
  ⑤ 클래스 불균형: pos_weight 동적 적용
  ⑥ 5-Fold 앙상블 임베딩 저장: 5개 Fold 모델의 결과를 평균내어 Person B 파이프라인 연동 준비 완료
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

import config
from data_preprocessing import (
    load_all_data,
    collate_graph_batch,
)
from model_builder import build_gcn_model


# ─────────────────────────────────────────────────────────────
# Dataset — GCN 그래프 배치 지원
# ─────────────────────────────────────────────────────────────

class DILIGraphDataset(Dataset):
    """
    GCN 입력용 Dataset.
    fp:     (N, fp_dim) numpy array — RF 선택된 Global FP
    graphs: list of (node_feats (N_i, F), adj (N_i, N_i)) tuples
    labels: (N,) numpy array
    """
    def __init__(self, fp: np.ndarray, graphs: list, labels: np.ndarray):
        self.fp     = fp
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.fp[idx], self.graphs[idx], self.labels[idx]


def gcn_collate_fn(batch):
    """
    Variable-size 그래프 배치를 패딩하여 텐서 튜플로 반환.
    batch: list of (fp_arr, (node_feats, adj), label)
    """
    fp_list     = [torch.tensor(b[0], dtype=torch.float32) for b in batch]
    graph_list  = [b[1] for b in batch]
    label_list  = [b[2] for b in batch]

    fp_batch = torch.stack(fp_list)                       # (B, fp_dim)
    x_batch, adj_batch, mask_batch = collate_graph_batch(graph_list)
    labels   = torch.tensor(label_list, dtype=torch.long) # (B,)

    return fp_batch, x_batch, adj_batch, mask_batch, labels


# ─────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """Val Loss 기반 Early Stopping. Best weight 자동 복원."""
    def __init__(self, patience: int = config.PATIENCE,
                 save_path: str = config.MODEL_WEIGHTS_PATH):
        self.patience   = patience
        self.save_path  = save_path
        self.best_loss  = np.inf
        self.counter    = 0
        self.early_stop = False

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ─────────────────────────────────────────────────────────────
# 학습 & 검증 루프
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []

    for fp, x, adj, mask, y in loader:
        fp, x, adj, mask, y = fp.to(device), x.to(device), adj.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(fp, x, adj, mask)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        probs       = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        preds_all.extend(probs)
        labels_all.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(labels_all, preds_all)
    return avg_loss, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []

    for fp, x, adj, mask, y in loader:
        fp, x, adj, mask, y = fp.to(device), x.to(device), adj.to(device), mask.to(device), y.to(device)
        logits = model(fp, x, adj, mask)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        probs       = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds_all.extend(probs)
        labels_all.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(labels_all, preds_all)
    return avg_loss, auc


# ─────────────────────────────────────────────────────────────
# 임베딩 추출 & 앙상블 (Auxiliary Head 제거 모드)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings_memory(model, fp_arr, graphs, device):
    """
    model.forward(..., return_embedding=True) 호출 → Concat 벡터 추출.
    배열(Array)로 바로 반환합니다. (Ensemble 평균 계산을 위해 저장 경로 없음)
    """
    model.eval()
    embs = []
    B    = config.BATCH_SIZE

    for i in range(0, len(fp_arr), B):
        fp_batch_np   = fp_arr[i:i+B]
        graphs_batch  = graphs[i:i+B]

        fp_t               = torch.tensor(fp_batch_np, dtype=torch.float32).to(device)
        x_t, adj_t, mask_t = collate_graph_batch(graphs_batch)

        emb = model(fp_t, x_t.to(device), adj_t.to(device), mask_t.to(device), return_embedding=True)
        embs.append(emb.cpu().numpy())

    embs = np.vstack(embs)
    return embs


def plot_history(history: dict, fold: int, save_dir: str = config.OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss Curve (Fold {fold})"); axes[0].legend()

    axes[1].plot(epochs, history["train_auc"], label="Train AUC")
    axes[1].plot(epochs, history["val_auc"],   label="Val AUC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title(f"AUC Curve (Fold {fold})"); axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"training_curve_fold{fold}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────
# 메인 학습 루프 (Train/Test 분리 + K-Fold + RF Feature Selection + 앙상블)
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}\n")

    # 1. 데이터 파이프라인
    fp_all, graphs_all, labels = load_all_data()
    n_samples = len(labels)

    # [중요] 임베딩 앙상블 과정에서의 Data Leakage 방지를 위해
    # 처음부터 80% Train_val / 20% Hold-out Test 셋으로 나눕니다.
    print(f"전체 샘플 수: {n_samples}")
    indices = np.arange(n_samples)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=config.TEST_SIZE, stratify=labels, random_state=config.RANDOM_SEED
    )

    fp_tv = fp_all[train_val_idx]
    y_tv = labels[train_val_idx]
    graphs_tv = [graphs_all[i] for i in train_val_idx]

    fp_test = fp_all[test_idx]
    y_test = labels[test_idx]
    graphs_test = [graphs_all[i] for i in test_idx]

    print(f"  Hold-out 분할 완료 -> Train/Val: {len(y_tv)}건 | Test: {len(y_test)}건\n")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    fold_aucs = []

    fold_models_info = []

    for fold, (train_inner, val_inner) in enumerate(kf.split(np.zeros(len(y_tv)), y_tv), 1):
        print(f"\n{'='*20} Fold {fold}/5 {'='*20}")

        # 2. Train / Val 분할
        fp_train, fp_val = fp_tv[train_inner], fp_tv[val_inner]
        y_train, y_val   = y_tv[train_inner], y_tv[val_inner]
        graphs_train = [graphs_tv[i] for i in train_inner]
        graphs_val   = [graphs_tv[i] for i in val_inner]

        # 3. RF 기반 Feature Selection (FP 차원 축소)
        print("  [RF] Training Random Forest for Feature Selection...")
        rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
        rf.fit(fp_train, y_train)
        top_k_idx    = np.argsort(rf.feature_importances_)[::-1][:config.FP_SELECT_DIM]
        
        fp_train_sel = fp_train[:, top_k_idx]
        fp_val_sel   = fp_val[:,   top_k_idx]
        print(f"  [RF] FP dim reduced: {fp_train.shape[1]} → {fp_train_sel.shape[1]}")

        # 4. 데이터로더 생성 (GCN 그래프 배치 patching)
        train_ds = DILIGraphDataset(fp_train_sel, graphs_train, y_train)
        val_ds   = DILIGraphDataset(fp_val_sel,   graphs_val,   y_val)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                  shuffle=True,  collate_fn=gcn_collate_fn)
        val_loader   = DataLoader(val_ds,  batch_size=config.BATCH_SIZE,
                                  shuffle=False, collate_fn=gcn_collate_fn)

        # 5. 모델 생성
        fp_dim = fp_train_sel.shape[1]
        model  = build_gcn_model(fp_dim, device)

        # 동적 pos_weight (클래스 불균형 보정)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight.item()]).to(device),
            label_smoothing=0.1
        )
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE,
                          weight_decay=config.WEIGHT_DECAY)
                          
        # Warm-up Scheduler 구현
        warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 5)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS - warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        early_stopping = EarlyStopping(
            patience  = config.PATIENCE,
            save_path = config.MODEL_WEIGHTS_PATH.replace(".pt", f"_gcn_fold{fold}.pt")
        )

        history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}

        # 6. 학습 루프
        fold_best_val_auc = 0
        for epoch in range(1, config.EPOCHS + 1):
            tr_loss, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_auc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
            history["train_auc"].append(tr_auc);   history["val_auc"].append(va_auc)

            improved = early_stopping.step(va_loss, model)
            if va_auc > fold_best_val_auc:
                fold_best_val_auc = va_auc

            star = "*" if improved else ""
            if epoch % 10 == 0 or improved or early_stopping.early_stop:
                print(f"  [Epoch {epoch:>2}] tr_loss: {tr_loss:.4f}, va_loss: {va_loss:.4f} | va_auc: {va_auc:.4f} {star}")

            if early_stopping.early_stop:
                print(f"  → Early Stopping at epoch {epoch}")
                break

        print(f"  [Fold {fold} 완료] Best Val AUC: {fold_best_val_auc:.4f}")
        fold_aucs.append(fold_best_val_auc)
        plot_history(history, fold)
        
        # 앙상블 모델 정보 저장
        fold_models_info.append({
            'model_path': early_stopping.save_path,
            'fp_dim': fp_dim,
            'top_k_idx': top_k_idx
        })

    print("\n" + "="*50)
    print(f"[완료] K-Fold 학습 종료 (평균 AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f})")

    # 7. 5-Fold 앙상블로 임베딩 추출 (Data Leakage 차단)
    print(f"\n[임베딩 추출 - 5-Fold 앙상블 추출 모드]")
    train_embs_list = []
    test_embs_list = []

    for fold_idx in range(5):
        print(f"  > Fold {fold_idx + 1} 모델로 임베딩 추출 중...")
        m_info = fold_models_info[fold_idx]
        top_k_idx = m_info['top_k_idx']

        # 개별 Fold의 모델 로드
        best_model = build_gcn_model(m_info['fp_dim'], device)
        best_model.load_state_dict(torch.load(m_info['model_path'], map_location=device, weights_only=True))

        # 전체 Train/Val 및 Test에 대해 RF 인덱스 적용
        fp_tv_sel   = fp_tv[:, top_k_idx]
        fp_test_sel = fp_test[:, top_k_idx]

        emb_train = extract_embeddings_memory(best_model, fp_tv_sel, graphs_tv, device)
        emb_test  = extract_embeddings_memory(best_model, fp_test_sel, graphs_test, device)
        
        train_embs_list.append(emb_train)
        test_embs_list.append(emb_test)

    # 5개의 임베딩 배열(Array) 산술 평균
    ensemble_train_embs = np.mean(train_embs_list, axis=0)
    ensemble_test_embs  = np.mean(test_embs_list, axis=0)

    # 최종 저장 (Person B 용도)
    np.save(config.EMBEDDING_TRAIN_PATH, ensemble_train_embs)
    np.save(config.LABEL_TRAIN_PATH, y_tv)

    np.save(config.EMBEDDING_TEST_PATH, ensemble_test_embs)
    np.save(config.LABEL_TEST_PATH, y_test)

    print(f"\n[완료] 5-Fold 앙상블 기반 임베딩 추출 & 저장 완료.")
    print(f"   Train 임베딩 차원: {ensemble_train_embs.shape}  -> {config.EMBEDDING_TRAIN_PATH}")
    print(f"   Test  임베딩 차원: {ensemble_test_embs.shape}  -> {config.EMBEDDING_TEST_PATH}")
    print(f"   (임베딩 차원 = PROJECT_DIM + Z_readout, Person B 연동 대기중)")


if __name__ == "__main__":
    main()
