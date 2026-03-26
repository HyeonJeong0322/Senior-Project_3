"""
train_dl_model.py  (— End-to-End 학습 및 임베딩 추출)
=================================================================
  ① End-to-End 사전 학습:
       CrossEntropyLoss + Label Smoothing으로 Attention 블록 포함
       전체 DL 모델을 역전파로 학습.
  ② Early Stopping: PATIENCE 에포크 내에 검증 AUC 개선 없으면 종료
  ③ 클래스 불균형 처리: pos_weight를 동적으로 계산하여 가중 손실
  ④ 임베딩 추출 & 저장:
       학습 완료 후 return_embedding=True로 MLP 직전 벡터 추출
       train/test 각각 .npy 파일로 저장 → 지은이 모델의 입력값으로 사용하면 됨
  ⑤ 학습 곡선 시각화(Loss/AUC)를 PNG로 자동 저장
    해당 파트 논문 제출 때 삭제 처리 할 것.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

from . import config
from .data_preprocessing import load_and_preprocess
from .model_builder import build_model


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class DILIDataset(Dataset):
    """FP(Global) + Sub(Local) + Label 묶음 Dataset."""

    def __init__(self, fp: np.ndarray, sub: np.ndarray, labels: np.ndarray):
        self.fp     = torch.tensor(fp,     dtype=torch.float32)
        self.sub    = torch.tensor(sub,    dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.fp[idx], self.sub[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """Val AUC 기반 Early Stopping. Best weight 자동 복원."""

    def __init__(self, patience: int = config.PATIENCE,
                 save_path: str = config.MODEL_WEIGHTS_PATH):
        self.patience   = patience
        self.save_path  = save_path
        self.best_auc   = -np.inf
        self.counter    = 0
        self.early_stop = False

    def step(self, val_auc: float, model: nn.Module):
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.counter  = 0
            torch.save(model.state_dict(), self.save_path)
            return True   # improved
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

    for fp, sub, y in loader:
        fp, sub, y = fp.to(device), sub.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(fp, sub)             # (B, 2)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * len(y)
        probs          = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        preds_all.extend(probs)
        labels_all.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(labels_all, preds_all)
    return avg_loss, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []

    for fp, sub, y in loader:
        fp, sub, y = fp.to(device), sub.to(device), y.to(device)
        logits  = model(fp, sub)
        loss    = criterion(logits, y)

        total_loss    += loss.item() * len(y)
        probs          = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds_all.extend(probs)
        labels_all.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(labels_all, preds_all)
    return avg_loss, auc


# ─────────────────────────────────────────────────────────────
# 임베딩 추출 & 저장
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_and_save_embeddings(model, fp_arr, sub_arr, labels,
                                save_emb_path, save_label_path, device):
    """
    학습 완료된 모델로 MLP 직전 임베딩 추출 후 .npy 저장.
    Person B(XGBoost)가 이 파일을 입력으로 사용.
    """
    model.eval()
    fp_t  = torch.tensor(fp_arr,  dtype=torch.float32).to(device)
    sub_t = torch.tensor(sub_arr, dtype=torch.float32).to(device)

    # 대용량 대비 배치 처리
    embs = []
    batch = config.BATCH_SIZE
    for i in range(0, len(fp_t), batch):
        emb = model(fp_t[i:i+batch], sub_t[i:i+batch], return_embedding=True)
        embs.append(emb.cpu().numpy())

    embs = np.vstack(embs)   # (N, MLP_HIDDEN_DIM)

    np.save(save_emb_path,   embs)
    np.save(save_label_path, labels)
    print(f"   저장 완료: {save_emb_path}  shape={embs.shape}")
    return embs


# ─────────────────────────────────────────────────────────────
# 학습 곡선 시각화
# ─────────────────────────────────────────────────────────────

def plot_history(history: dict, save_dir: str = config.OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend()

    axes[1].plot(epochs, history["train_auc"], label="Train AUC")
    axes[1].plot(epochs, history["val_auc"],   label="Val AUC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("AUC Curve"); axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   학습 곡선 저장: {out_path}")


# ─────────────────────────────────────────────────────────────
# 메인 학습 진입점
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}\n")

    # ── 1. 데이터 로드 & 전처리 ──
    (fp_tr, sub_tr, y_tr), (fp_te, sub_te, y_te), vocab = load_and_preprocess()

    sub_dim = fp_tr.shape[1]   # 전처리 결과로 결정된 sub vocab 크기
    # 실제 sub_dim은 sub 벡터의 두번째 차원
    sub_dim = sub_tr.shape[1]
    fp_dim  = fp_tr.shape[1]

    # ── 2. DataLoader 생성 ──
    train_ds = DILIDataset(fp_tr, sub_tr, y_tr)
    test_ds  = DILIDataset(fp_te, sub_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                              shuffle=False, drop_last=False)

    # ── 3. 모델 & 최적화 설정 ──
    model = build_model(fp_dim, sub_dim, device)

    # 클래스 불균형: pos_weight = (# neg) / (# pos)
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight = {pos_weight.item():.3f}  (neg:{n_neg}, pos:{n_pos})\n")

    # Label Smoothing CrossEntropy (소규모 데이터 과적합 완화)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight.item()]).to(device),
        label_smoothing=0.1
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        save_path=config.MODEL_WEIGHTS_PATH
    )

    # ── 4. End-to-End 학습 루프 ──
    history = {"train_loss": [], "val_loss": [],
               "train_auc":  [], "val_auc":  []}

    print("=" * 55)
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} "
          f"{'TrainAUC':>10} {'ValAUC':>9} {'Best':>5}")
    print("=" * 55)

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_auc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(va_auc)

        improved = early_stopping.step(va_auc, model)
        star = "*" if improved else ""
        print(f"{epoch:>6} {tr_loss:>10.4f} {va_loss:>10.4f} "
              f"{tr_auc:>10.4f} {va_auc:>9.4f} {star:>5}")

        if early_stopping.early_stop:
            print(f"\n[Early Stopping] Best Val AUC: {early_stopping.best_auc:.4f}")
            break

    # ── 5. Best 가중치 복원 ──
    model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location=device))
    print(f"\n[Best 모델 복원] Val AUC: {early_stopping.best_auc:.4f}")

    # ── 6. 학습 곡선 저장 ──
    plot_history(history)

    # ── 7. 임베딩 추출 & Person B 전달용 .npy 저장 ──
    print("\n[임베딩 추출 및 저장]")
    extract_and_save_embeddings(
        model, fp_tr, sub_tr, y_tr,
        config.EMBEDDING_TRAIN_PATH, config.LABEL_TRAIN_PATH, device
    )
    extract_and_save_embeddings(
        model, fp_te, sub_te, y_te,
        config.EMBEDDING_TEST_PATH, config.LABEL_TEST_PATH, device
    )

    print("\n✅ Person A 작업 완료.")
    print(f"   Person B에게 전달 파일:")
    print(f"     - {config.EMBEDDING_TRAIN_PATH}")
    print(f"     - {config.EMBEDDING_TEST_PATH}")
    print(f"     - {config.LABEL_TRAIN_PATH}")
    print(f"     - {config.LABEL_TEST_PATH}")


if __name__ == "__main__":
    main()
