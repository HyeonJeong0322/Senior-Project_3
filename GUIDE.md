# 프로젝트 가이드

DILI(Drug-Induced Liver Injury) 예측 모델 실험 플랫폼입니다.
여러 모델을 동일한 환경에서 실행하고 성능을 비교하는 것이 목표입니다.

처음이라면 먼저 [SETUP.md](SETUP.md)를 따라 환경을 세팅하세요.

---

## 프로젝트 구조

```
Dili/
├── run.sh                  # 실행 명령어 모음 (Makefile 대신 사용)
├── docker-compose.yml      # Docker 컨테이너 설정
├── Dockerfile
├── data/
│   └── dilirank.csv        # 원본 데이터 (직접 넣어야 함)
└── src/
    ├── train.py            # 진입점: 모델 이름 받아서 실행
    ├── registry.py         # 모델 이름 → 클래스 매핑 테이블
    ├── dataset.py          # 공용 유틸 (fingerprint 변환 등)
    ├── env_test.py         # 환경 체크 스크립트
    └── models/
        ├── stackdili/      # StackDILI (트리 기반 앙상블)
        │   ├── model.py
        │   └── Code/, Data/, Feature/, Model/
        └── my_model/       # Crossed Differential Attention 모델
            ├── model.py
            ├── outputs/    # 학습 결과 자동 저장 (실행 후 생성)
            └── attention_module/
                ├── config.py
                ├── attention_module.py
                ├── data_preprocessing.py
                ├── model_builder.py
                └── train_dl_model.py
```

---

## 모델 실행

### 기본 명령어

```bash
./run.sh run <모델이름>      # Mac/Linux
bash run.sh run <모델이름>   # Windows
```

### StackDILI 실행

```bash
./run.sh run stackdili
```

Jupyter 노트북 2개(ML_model.ipynb → stacking.ipynb)를 순서대로 실행하고 AUC를 출력합니다.

```
Running StackDILI pipeline...
Executing Code/ML_model.ipynb...
Executing Code/stacking.ipynb...
StackDILI AUC: 0.97
```

결과는 `src/models/stackdili/result.txt`에도 저장됩니다.

### my_model 실행

```bash
./run.sh run my_model
```

데이터 전처리 → Attention 모델 학습 → AUC 출력 순으로 진행됩니다.

```
[1/5] 데이터 로드: ...
[2/5] SMILES 유효성 검증 및 Mol 변환
[3/5] Global Fingerprint 추출 (MACCS + Morgan ECFP6)
[4/5] Local Substructure 추출 (BRICS + Bemis-Murcko)
[5/5] 완료 | Train: N개, Test: N개

======================================================
 Epoch  TrainLoss    ValLoss   TrainAUC    ValAUC  Best
======================================================
     1     0.6231     0.6102     0.7124    0.7089     *
   ...
[Best 모델 복원] Val AUC: 0.8xxx
```

학습 완료 후 아래 파일들이 자동 저장됩니다.

```
src/models/my_model/outputs/
├── attention_model.pt        # 모델 가중치
├── train_embeddings.npy      # 학습 임베딩 (XGBoost 입력용)
├── test_embeddings.npy
├── train_labels.npy
├── test_labels.npy
└── training_curve.png        # Loss/AUC 학습 곡선
```

---

## 전체 명령어 정리

| 명령어 | 설명 |
|---|---|
| `./run.sh build` | Docker 이미지 빌드 (최초 1회) |
| `./run.sh run <모델>` | 모델 실행 |
| `./run.sh env-test` | 환경 체크 |
| `./run.sh shell` | 컨테이너 내부 쉘 접속 |

---

## 새 모델 추가하는 방법

### 1. 모델 폴더 만들기

```
src/models/<모델이름>/
```

### 2. model.py 작성

`run()` 메서드 하나만 있으면 됩니다.
데이터 로드, 학습, 결과 출력까지 이 안에서 처리하세요.

```python
class Model:
    def run(self):
        # 데이터 로드
        # 학습
        # AUC 출력
        print("AUC:", ...)
```

### 3. registry.py에 등록

```python
from models.<모델이름>.model import Model as MYMODEL

MODEL_REGISTRY = {
    "stackdili": STACKDILI,
    "my_model":  MY_MODEL,
    "<모델이름>": MYMODEL,   # 추가
}
```

### 4. 실행

```bash
./run.sh run <모델이름>
```

---

## 구조 설계 원칙

| 레이어 | 역할 |
|---|---|
| **Docker 컨테이너** | 실행 환경. OS/라이브러리 버전 통일 |
| **src/** | 진입점과 모델 등록만 담당. 학습 로직 없음 |
| **src/models/<이름>/** | 모델의 모든 것 (데이터, 학습, 결과) |

각 모델은 `run()` 하나로 실행되는 독립적인 실험 단위입니다.
모델끼리 서로의 코드에 영향을 주지 않습니다.
