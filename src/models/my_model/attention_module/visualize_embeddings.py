"""
visualize_embeddings.py
=======================
DCA로 생성한 .npy 임베딩 파일을 시각화, 단순 확인을 위한 용임으로 실행하지 않아도 상관 없음.
이후 논문 제출 때엔 삭제 할 py파일.

생성되는 차트 (outputs/ 폴더에 저장):
  1. embedding_tsne.png     — t-SNE 2D 산점도 (클래스별 색상)
  2. embedding_heatmap.png  — 샘플별 임베딩 값 히트맵
  3. embedding_distribution.png — 차원별 값 분포 박스플롯
  4. class_balance.png      — Train / Test 클래스 비율 파이차트
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── 경로 설정 ──
BASE_OUTPUT = r"C:\Capston2026\CAP2026_0325\outputs"

TRAIN_EMB   = os.path.join(BASE_OUTPUT, "train_embeddings.npy")
TEST_EMB    = os.path.join(BASE_OUTPUT, "test_embeddings.npy")
TRAIN_LABEL = os.path.join(BASE_OUTPUT, "train_labels.npy")
TEST_LABEL  = os.path.join(BASE_OUTPUT, "test_labels.npy")

SAVE_DIR    = BASE_OUTPUT
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 색상 설정 ──
COLOR_MAP   = {0: "#4f8ef7", 1: "#f7634f"}    # 0=non-DILI(파랑), 1=DILI(빨강)
LABEL_NAME  = {0: "non-DILI", 1: "DILI"}
DARK_BG     = "#1a1a2e"
PANEL_BG    = "#16213e"
TEXT_COLOR  = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   TEXT_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "font.family":      "DejaVu Sans",
    "grid.color":       "#2a2a4e",
    "grid.linewidth":   0.5,
})


# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────

def load_data():
    print("[로드] 임베딩 파일 읽는 중...")
    tr_emb = np.load(TRAIN_EMB)
    te_emb = np.load(TEST_EMB)
    tr_lbl = np.load(TRAIN_LABEL)
    te_lbl = np.load(TEST_LABEL)

    emb_all = np.vstack([tr_emb, te_emb])
    lbl_all = np.concatenate([tr_lbl, te_lbl])
    split   = np.array(["Train"] * len(tr_lbl) + ["Test"] * len(te_lbl))

    print(f"  Train: {tr_emb.shape}  /  Test: {te_emb.shape}")
    print(f"  임베딩 차원: {tr_emb.shape[1]}")
    return tr_emb, te_emb, tr_lbl, te_lbl, emb_all, lbl_all, split


# ─────────────────────────────────────────────────────────────
# 1. t-SNE 2D 산점도
# ─────────────────────────────────────────────────────────────

def plot_tsne(emb_all, lbl_all, split):
    print("[1/4] t-SNE 2D 시각화 중... (30~60초 소요)")
    scaler  = StandardScaler()
    emb_sc  = scaler.fit_transform(emb_all)

    tsne    = TSNE(n_components=2, perplexity=30,
                   random_state=42, n_iter=1000, verbose=0)
    emb_2d  = tsne.fit_transform(emb_sc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE Embedding Space", fontsize=15, fontweight="bold", color=TEXT_COLOR)

    # (a) 클래스별 색
    for lbl, name in LABEL_NAME.items():
        mask = lbl_all == lbl
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                        c=COLOR_MAP[lbl], label=name,
                        alpha=0.7, s=25, edgecolors="none")
    axes[0].set_title("by DILI Class", color=TEXT_COLOR)
    axes[0].legend(framealpha=0.3)
    axes[0].grid(True)

    # (b) Train/Test 구분
    for sp, color in [("Train", "#a0d8ef"), ("Test", "#f7c59f")]:
        mask = split == sp
        axes[1].scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                        c=color, label=sp,
                        alpha=0.6, s=20, edgecolors="none")
    axes[1].set_title("by Train / Test Split", color=TEXT_COLOR)
    axes[1].legend(framealpha=0.3)
    axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "embedding_tsne.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   저장: {path}")


# ─────────────────────────────────────────────────────────────
# 2. 임베딩 히트맵 (샘플 × 차원)
# ─────────────────────────────────────────────────────────────

def plot_heatmap(tr_emb, tr_lbl):
    print("[2/4] 임베딩 히트맵 생성 중...")

    # 클래스별로 정렬하여 시각화
    order   = np.argsort(tr_lbl)
    emb_sorted = tr_emb[order]
    lbl_sorted = tr_lbl[order]

    # 최대 200개 샘플만 표시
    n_show    = min(200, len(emb_sorted))
    step      = max(1, len(emb_sorted) // n_show)
    emb_show  = emb_sorted[::step][:n_show]
    lbl_show  = lbl_sorted[::step][:n_show]

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[0.03, 1], wspace=0.03)

    # 라벨 사이드바
    ax_lbl = fig.add_subplot(gs[0])
    ax_lbl.imshow(
        lbl_show.reshape(-1, 1),
        aspect="auto", cmap="RdBu_r", vmin=0, vmax=1
    )
    ax_lbl.set_xticks([])
    ax_lbl.set_ylabel("Sample (sorted by class)", color=TEXT_COLOR)
    ax_lbl.set_title("Label", fontsize=9, color=TEXT_COLOR)

    # 히트맵 본체
    ax_map  = fig.add_subplot(gs[1])
    im = ax_map.imshow(
        emb_show,
        aspect="auto", cmap="viridis",
        vmin=np.percentile(emb_show, 2),
        vmax=np.percentile(emb_show, 98)
    )
    ax_map.set_xlabel("Embedding Dimension", color=TEXT_COLOR)
    ax_map.set_yticks([])
    ax_map.set_title(
        f"Attention Embedding Heatmap (Train, {n_show} samples × {tr_emb.shape[1]} dims)",
        color=TEXT_COLOR, fontsize=11
    )
    fig.colorbar(im, ax=ax_map, fraction=0.02, pad=0.01)

    fig.suptitle("", fontsize=1)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "embedding_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   저장: {path}")


# ─────────────────────────────────────────────────────────────
# 3. 차원별 값 분포 (클래스별 박스플롯, 상위 20 차원)
# ─────────────────────────────────────────────────────────────

def plot_distribution(tr_emb, tr_lbl):
    print("[3/4] 차원별 분포 박스플롯 생성 중...")

    # 클래스 간 평균 차이가 큰 상위 20 차원 선택
    mean_0 = tr_emb[tr_lbl == 0].mean(axis=0)
    mean_1 = tr_emb[tr_lbl == 1].mean(axis=0)
    top_dims = np.argsort(np.abs(mean_1 - mean_0))[-20:][::-1]

    fig, axes = plt.subplots(4, 5, figsize=(18, 10))
    fig.suptitle("Top-20 Discriminative Embedding Dimensions (Train)",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)
    axes = axes.flatten()

    for i, dim in enumerate(top_dims):
        ax = axes[i]
        data = [tr_emb[tr_lbl == 0, dim], tr_emb[tr_lbl == 1, dim]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        bp["boxes"][0].set_facecolor(COLOR_MAP[0])
        bp["boxes"][1].set_facecolor(COLOR_MAP[1])
        ax.set_title(f"dim {dim}", fontsize=8, color=TEXT_COLOR)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["non-DILI", "DILI"], fontsize=7)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "embedding_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   저장: {path}")


# ─────────────────────────────────────────────────────────────
# 4. 클래스 균형 파이차트
# ─────────────────────────────────────────────────────────────

def plot_class_balance(tr_lbl, te_lbl):
    print("[4/4] 클래스 균형 파이차트 생성 중...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Class Balance (DILI vs non-DILI)",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    for ax, lbl, title in zip(axes, [tr_lbl, te_lbl], ["Train", "Test"]):
        counts  = [np.sum(lbl == 0), np.sum(lbl == 1)]
        colors  = [COLOR_MAP[0], COLOR_MAP[1]]
        explode = [0.03, 0.03]
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[f"non-DILI\n({counts[0]})", f"DILI\n({counts[1]})"],
            colors=colors,
            explode=explode,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": TEXT_COLOR, "fontsize": 11},
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
        ax.set_title(title, color=TEXT_COLOR, fontsize=13)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "class_balance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   저장: {path}")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tr_emb, te_emb, tr_lbl, te_lbl, emb_all, lbl_all, split = load_data()

    plot_tsne(emb_all, lbl_all, split)
    plot_heatmap(tr_emb, tr_lbl)
    plot_distribution(tr_emb, tr_lbl)
    plot_class_balance(tr_lbl, te_lbl)

    print("\n✅ 시각화 완료! 결과 파일:")
    for fname in ["embedding_tsne.png", "embedding_heatmap.png",
                  "embedding_distribution.png", "class_balance.png"]:
        print(f"   → {os.path.join(SAVE_DIR, fname)}")
