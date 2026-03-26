# CLAUDE.md — CAP2026_0325 프로젝트 컨텍스트

## 프로젝트 목적

**약물 유발성 간손상(Drug-Induced Liver Injury, DILI) 예측** 정확도와 강건성 극대화를 위한
**이종 분자 특징(Heterogeneous Molecular Features) 간의 상호 설명적 융합(Cross-Explanatory Fusion)** 모델.

기존 방법의 한계:
- 단일 분자 지문(Single Fingerprint) 사용 → 정보량 절대 부족
- 단순 Concatenation → 노이즈 급증, 성능 저하

본 알고리즘의 핵심:
- **전역적 특징 (FP)**: MACCS Keys(167) + Morgan ECFP6(512) = 679-dim → RF로 128-dim 선택
- **국소적 특징 (Sub/GCN)**: 2-Layer GCN + Residual + JK-Cat + Differential Attention Readout → 분자 그래프 위상 보존
- **Cross-Explanatory Fusion**: 이종 특징이 서로 부족한 정보를 교차 참조(Cross-Reference)
- **Differential Attention**: 불필요한 노이즈 연관성을 상쇄 → 간 독성 직결 패턴에 집중

---

## 파일 구조 및 역할

```
Differential_AttentionBlock/
├── config.py               # 공통 하이퍼파라미터 및 경로 (단일 진실 공급원)
├── data_preprocessing.py   # SMILES → FP(Global) + Fragment(Local/BoF) + Graph(Local/GCN)
├── attention_module.py     # OneLayerGCN(레거시), TwoLayerGCN(현재), GraphAttentionReadout, DiffAttentionHead, CrossedDiffAttentionLayer
├── model_builder.py        # DILIGCNModel (현재 사용), DILIAttentionModel (레거시 BoF 기반)
├── train_dl_model.py       # K-Fold 학습 루프, EarlyStopping, 임베딩 추출
├── eval_and_report.py      # XGBoost 분류 + HTML 리포트 생성
└── visualize_embeddings.py # t-SNE / Heatmap 임베딩 시각화 (논문 제출 전 삭제 예정)
```

출력 경로: `C:\Capston2026\CAP2026_0325\outputs\`
데이터 경로: `C:\Capston2026\StackDILI-main\Data\Dataset.csv` (StackDILI 통합 데이터셋)

---

## 실행 순서

```bash
# Step 1: 데이터 전처리 확인 (단독 실행 가능)
python data_preprocessing.py

# Step 2: 모델 학습 (5-Fold K-Fold, 임베딩 자동 저장)
python train_dl_model.py

# Step 3: 평가 및 HTML 리포트 생성
python eval_and_report.py

# (선택) 임베딩 시각화
python visualize_embeddings.py
```

---

## 모델 아키텍처 (DILIGCNModel) — 2-Layer GCN + Crossed Attention (2026-03-27)

```
SMILES
  │
  ├─[Path A: Global]─────────────────────────────────────────────────────────────────┐
  │  MACCS(167) + Morgan ECFP6(512) = 679-dim                                        │
  │  → VT(분산<0.01 제거) → RF Feature Selection (K-Fold 외부 1회 고정) → 128-dim   │
  │  → ProjectionBlock (Linear → LayerNorm → ReLU → Dropout) → fp_vec (128-dim)     │
  │                                                                                   │
  └─[Path B: Local/GCN]──────────────────────────────────────────────────────────────┤
     원자 피처(25-dim) + 인접 행렬                                                     │
     → TwoLayerGCN                                                                   │
       Layer 1 (1-hop): 25 → 64  →  ReLU + LayerNorm + Dropout  →  H1              │
       Layer 2 (2-hop): 64 → 128 + Residual(W_res(H1))          →  H2              │
       JK-Cat: cat(H1, H2) → W_jk → 128-dim  →  node_mat (B, N, 128)              │
     │                                                                                │
     │  [Crossed Attention Pass 2: GCN → FP 방향]                                   │
     │  NodeFPCrossAttention                                                          │
     │    각 원자_i: gate_i = σ(W_q(node_i)·W_k(fp_vec)/√d)                         │
     │    node_i += gate_i * W_v(fp_vec)                                             │
     │    → 독성기 원자: FP와 공명 → gate↑ / 노이즈 원자: gate≈0                    │
     │  → node_mat_fp (B, N, 128)                                                    │
     │                                                                                │
     │  [Crossed Attention Pass 1: FP → GCN 방향]                                   │
     │  GraphAttentionReadout (fp_vec as Query, node_mat_fp as K/V)                  │
     │    A_diff = A1 - λ*A2  (Differential: 노이즈 원자 상쇄)                       │
     │  → z_readout (128-dim)                                                        │
     │                                                                                │
     └──────────────────────────────────────── Concat(fp_vec ‖ z_readout) ──────────┘
                                                        256-dim
                                                           │
                                                  MLP Classifier
                                            (256 → 128 → 2, Softmax)
```

**Crossed Attention 설계 원칙:**
- Pass 2 입력은 반드시 **순수 GCN node_mat** (fp_vec 개입 전) → FP 편향 방지
- FP 단일 토큰에는 Softmax 대신 **Sigmoid 게이팅** → 단일 원소 Softmax 상수화 방지
- Pass 1 Readout은 Pass 2로 업데이트된 node_mat_fp 사용 → 더 풍부한 원자 표현 참조

**2-Layer GCN 설계 원칙:**
- Residual: `W_res(H1)`을 H2에 더해 1-hop 원자 ID 보존 → Oversmoothing 방지
- JK-Cat: 1-hop(64)과 2-hop(128) 표현 병렬 보존 → 네트워크가 중요 스케일 학습으로 선택
- 출력 차원 GCN_OUT_DIM=128 유지 → 하위 모듈 하위 호환

---

## 현재 하이퍼파라미터 (config.py 기준)

| 항목 | 값 | 비고 |
|------|-----|------|
| `PROJECT_DIM` | 128 | Linear Projection 차원 |
| `GCN_NODE_FEAT_DIM` | 25 | 원자 피처 차원 (원자번호 one-hot 22 + 전하 + 방향족 + 수소) |
| `GCN_HIDDEN_DIM` | 64 | TwoLayerGCN Layer 1 출력 차원 (1-hop) |
| `GCN_OUT_DIM` | 128 | TwoLayerGCN Layer 2 출력 차원 (2-hop + JK-Cat 후), PROJECT_DIM과 통일 |
| `NUM_HEADS` | 4 | Multi-head Attention |
| `NUM_STACKS` | 1 | Crossed Diff Attention 반복 횟수 |
| `ATTN_DROPOUT` | 0.2 | Attention Dropout |
| `FF_DROPOUT` | 0.3 | Feed-Forward Dropout (GCN Layer 간 Dropout 포함) |
| `BATCH_SIZE` | 64 | 미니배치 크기 |
| `EPOCHS` | 80 | 최대 학습 에폭 |
| `WARMUP_EPOCHS` | 5 | LR 스케줄러 Warmup용 |
| `LEARNING_RATE` | 1e-4 | OneCycleLR max_lr 기준 |
| `WEIGHT_DECAY` | 1e-2 | AdamW L2 정규화 |
| `PATIENCE` | 15 | Early Stopping (val AUC 기준) |
| `MLP_HIDDEN_DIM` | 128 | MLP 분류기 숨김 차원 |
| `FP_VARIANCE_THRESHOLD` | 0.01 | VT 1단계: 분산 < 0.01 비트 제거 |
| `FP_SELECT_DIM` | 128 | RF 2단계: VT 통과 비트 중 상위 K개 선택 |
| `FP_DIM` | 679 | MACCS(167) + Morgan(512) 원본 차원 |

---

## 학습 설계 원칙 (변경 금지 사항)

1. **RF Feature Selection은 K-Fold 외부에서 1회만 수행**
   - Fold마다 다른 feature index를 선택하면 각 Fold 모델의 입력 공간이 달라져 비교 불가
   - `train_dl_model.py`의 `main()` 진입 직후, `kf.split()` 루프 시작 전에 수행

2. **GCN은 TwoLayerGCN (2-Layer + Residual + JK-Cat) 사용**
   - Layer 1 (1-hop, 25→64) + Layer 2 (2-hop, 64→128) + Residual(W_res) + JK-Cat
   - Residual: 1-hop 신호를 직접 더해 Oversmoothing 방지, 원자 ID 보존
   - JK-Cat: cat(H1, H2) → W_jk → 128, 두 스케일 동시 보존
   - `OneLayerGCN`은 레거시로 보존 (삭제 금지, DILIAttentionModel 하위 호환)

3. **Lambda 계산은 sigmoid 기반**
   - `GraphAttentionReadout`의 lambda: `sigmoid(λq · λk)` 형태 유지
   - exp() 기반 lambda는 학습 초반 gradient 폭발 위험 (DiffAttentionHead는 현재 미사용)

4. **데이터 누수 방지**
   - Fragment Vocabulary(`build_fragment_vocab`)는 반드시 Train split에서만 생성
   - RF Feature Selection index도 Test에 맞춘 선택 금지

5. **Early Stopping 기준: val AUC**
   - val loss 기반은 Dropout 효과로 train loss가 artificially inflate되어 왜곡 발생
   - val AUC 기준으로 체크포인트 저장 (`EarlyStopping.step(val_auc, model)`)

6. **LR 스케줄러: OneCycleLR (배치 단위 step)**
   - `train_one_epoch()` 내부 각 배치 후 `scheduler.step()` 호출
   - epoch 단위 호출 시 T_max 불일치로 스케줄 붕괴

---

## 알려진 문제 및 원인 (2026-03-27 분석 기록)

### Training Curve 불안정
| 원인 | 위치 |
|------|------|
| Fold마다 다른 RF Feature index → 입력 공간 불일치 | `train_dl_model.py` (K-Fold 외부 1회 수행으로 수정 완료) |
| Warmup 없는 CosineAnnealingLR → 초반 loss spike | `train_dl_model.py` (LinearLR+CosineAnnealingLR SequentialLR로 교체 완료) |
| FF_DROPOUT=0.5로 train loss 인위적 진동 | `config.py` (0.3으로 완화 완료) |
| pos_weight 동적 계산 → Fold 간 loss scale 변동 | `train_dl_model.py` (구조적 허용, 모니터링) |
| EarlyStopping val_loss 기준 → Dropout으로 왜곡 | `train_dl_model.py` (val_AUC 최대화 기준으로 수정 완료) |
| 앙상블 임베딩 추출을 별도 루프로 수행 → 모델 5회 재로드 | `train_dl_model.py` (Fold 학습 직후 즉시 추출로 통합 완료) |

### Val/Train Loss Gap 미해소
| 원인 | 위치 |
|------|------|
| WEIGHT_DECAY=1e-2 (과도한 L2) → Underfitting | `config.py` (5e-4로 완화 완료) |
| Dropout으로 train loss > val loss 역전 현상 | `config.py` (완화 완료) |
| PROJECT_DIM=64 → 표현력 부족 | `config.py` (128로 향상 완료) |
| val AUC 대신 val loss 기준 Early Stopping | `train_dl_model.py` (val AUC 기준으로 전환 완료) |

### 향후 검토 항목 (미적용, 필요 시 추가)
- SMILES Enumeration Augmentation (데이터 3~5배 증강)
- Focal Loss 도입 (pos_weight + label_smoothing 이중 처리 통합)
- `DiffAttentionHead` lambda sigmoid 통일 (현재 CrossedDiffAttentionStack은 DILIGCNModel에서 미사용)

---

## 협업 참고

- `train_embeddings.npy`, `test_embeddings.npy`: DILIGCNModel이 추출한 256-dim 임베딩 → 지은이 파트 입력
- `attention_model_gcn_fold{1~5}.pt`: 각 Fold 최적 가중치 (val AUC 기준)
- `training_curve_fold{1~5}.png`: 학습 곡선 시각화
- `outputs/reports/0{1~5}.html`: 각 Fold 평가 리포트 (XGBoost + 파이프라인 다이어그램 포함)
