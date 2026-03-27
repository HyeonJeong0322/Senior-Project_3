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
import warnings

# PyTorch SequentialLR 내부 버그로 인한 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module=".*lr_scheduler.*")

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
from sklearn.feature_selection import VarianceThreshold

from . import config
from .data_preprocessing import (
    load_all_data,
    collate_graph_batch,
)
from .model_builder import build_gcn_model


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
    """Val AUC 기반 Early Stopping (최대화). Best weight 자동 복원."""
    def __init__(self, patience: int = config.PATIENCE,
                 save_path: str = config.MODEL_WEIGHTS_PATH):
        self.patience    = patience
        self.save_path   = save_path
        self.best_score  = -np.inf
        self.counter     = 0
        self.early_stop  = False

    def step(self, val_score: float, model: nn.Module):
        if val_score > self.best_score:
            self.best_score = val_score
            self.counter    = 0
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

    # 2. Feature Selection — K-Fold 외부에서 1회만 수행 (Data Leakage 방지)
    # Fold마다 다른 index를 선택하면 각 Fold 모델의 입력 공간이 달라져 앙상블 비교 불가

    # 1단계: Variance Threshold — 희소/편재 비트 제거 (fp_tv 기준 fit, fp_test는 transform만)
    print("  [VT] Removing low-variance FP bits...")
    vt = VarianceThreshold(threshold=config.FP_VARIANCE_THRESHOLD)
    fp_tv_vt   = vt.fit_transform(fp_tv)
    fp_test_vt = vt.transform(fp_test)
    print(f"  [VT] FP dim: {fp_tv.shape[1]} → {fp_tv_vt.shape[1]} ({fp_tv.shape[1] - fp_tv_vt.shape[1]}개 희소 비트 제거)")

    # 2단계: RF — 정제된 비트에서 상위 K개 선택
    print("  [RF] Training Random Forest for Feature Selection (1회 고정)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf.fit(fp_tv_vt, y_tv)
    top_k_idx   = np.argsort(rf.feature_importances_)[::-1][:config.FP_SELECT_DIM]
    fp_tv_sel   = fp_tv_vt[:, top_k_idx]
    fp_test_sel = fp_test_vt[:, top_k_idx]
    print(f"  [RF] FP dim: {fp_tv_vt.shape[1]} → {fp_tv_sel.shape[1]}\n")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    fold_aucs = []
    train_embs_list = []
    test_embs_list  = []

    for fold, (train_inner, val_inner) in enumerate(kf.split(np.zeros(len(y_tv)), y_tv), 1):
        print(f"\n{'='*20} Fold {fold}/5 {'='*20}")

        # 3. Train / Val 분할 (RF 선택된 FP 사용)
        fp_train_sel = fp_tv_sel[train_inner]
        fp_val_sel   = fp_tv_sel[val_inner]
        y_train, y_val   = y_tv[train_inner], y_tv[val_inner]
        graphs_train = [graphs_tv[i] for i in train_inner]
        graphs_val   = [graphs_tv[i] for i in val_inner]

        # 4. 데이터로더 생성 (GCN 그래프 배치 patching)
        train_ds = DILIGraphDataset(fp_train_sel, graphs_train, y_train)
        val_ds   = DILIGraphDataset(fp_val_sel,   graphs_val,   y_val)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                  shuffle=True,  collate_fn=gcn_collate_fn)
        val_loader   = DataLoader(val_ds,  batch_size=config.BATCH_SIZE,
                                  shuffle=False, collate_fn=gcn_collate_fn)

        # 4. 모델 생성
        fp_dim = fp_train_sel.shape[1]   # = config.FP_SELECT_DIM (128)
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

            improved = early_stopping.step(va_auc, model)
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

        # 7. 임베딩 추출 — 모델이 메모리에 있는 지금 즉시 추출 (별도 루프/재로드 불필요)
        print(f"  > Fold {fold} 모델로 임베딩 추출 중...")
        best_model = build_gcn_model(fp_dim, device)
        best_model.load_state_dict(
            torch.load(early_stopping.save_path, map_location=device, weights_only=True)
        )
        train_embs_list.append(extract_embeddings_memory(best_model, fp_tv_sel,   graphs_tv,   device))
        test_embs_list.append( extract_embeddings_memory(best_model, fp_test_sel, graphs_test, device))
        del best_model  # 메모리 즉시 해제

    print("\n" + "="*50)
    print(f"[완료] K-Fold 학습 종료 (평균 AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f})")

    # 7. 5-Fold 앙상블로 임베딩 추출 (Data Leakage 차단)
    print(f"\n[임베딩 추출 - 5-Fold 앙상블 추출 모드]")

    # ── OOF 방식 Train embedding 생성 ───────────────────────
    print(f"\n[임베딩 추출 - OOF 방식]")

    train_embs = None
    test_embs_list = []

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    for fold_idx, (_, val_idx) in enumerate(kf.split(np.zeros(len(y_tv)), y_tv)):
        print(f"  > Fold {fold_idx + 1} 모델로 OOF 임베딩 추출 중...")

        m_info = fold_models_info[fold_idx]
        top_k_idx = m_info['top_k_idx']

        best_model = build_gcn_model(m_info['fp_dim'], device)
        best_model.load_state_dict(torch.load(m_info['model_path'], map_location=device, weights_only=True))

        # RF 적용
        fp_tv_sel   = fp_tv[:, top_k_idx]
        fp_test_sel = fp_test[:, top_k_idx]

        # validation만 embedding 생성
        emb_val = extract_embeddings_memory(
            best_model,
            fp_tv_sel[val_idx],
            [graphs_tv[i] for i in val_idx],
            device
        )

        # 🔥 핵심: 여기에서 차원 자동 생성
        if train_embs is None:
            train_embs = np.zeros((len(fp_tv), emb_val.shape[1]))

        train_embs[val_idx] = emb_val

        # test는 평균용
        emb_test = extract_embeddings_memory(best_model, fp_test_sel, graphs_test, device)
        test_embs_list.append(emb_test)

    # test는 평균
    ensemble_test_embs = np.mean(test_embs_list, axis=0)

    # 저장
    np.save(config.EMBEDDING_TRAIN_PATH, train_embs)
    np.save(config.LABEL_TRAIN_PATH, y_tv)

    np.save(config.EMBEDDING_TEST_PATH, ensemble_test_embs)
    np.save(config.LABEL_TEST_PATH, y_test)

    # -------------------------------------------------------------
    # 수정을 좀 했습니다... 

    import os
    # 폴더가 없을 경우를 대비해 확실히 경로를 잡아줍니다 (선택사항이지만 안전함)
    output_dir = "/app/src/models/my_model/outputs"
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"{output_dir}/train_fps.npy", fp_tv)
    np.save(f"{output_dir}/test_fps.npy", fp_test)
    
    # -------------------------------------------------------------

    print(f"\n[완료] OOF 기반 임베딩 추출 & 저장 완료.")
    print(f"   Train 임베딩 차원: {train_embs.shape}  -> {config.EMBEDDING_TRAIN_PATH}")
    print(f"   Test  임베딩 차원: {ensemble_test_embs.shape}  -> {config.EMBEDDING_TEST_PATH}")
    print(f"   (임베딩 차원 = 2 * PROJECT_DIM = 256-dim, Person B 연동 대기중)")

    print(f"   Train FP 차원: {fp_tv.shape} -> {output_dir}/train_fps.npy")
    print(f"   Test  FP 차원: {fp_test.shape} -> {output_dir}/test_fps.npy")

if __name__ == "__main__":
    main()
