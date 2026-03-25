# DILI ML Experiment Platform

이 프로젝트는 다양한 머신러닝 모델을 **동일한 환경에서 실행하고 성능을 비교**하기 위한 실험 플랫폼입니다.

Docker 기반으로 구성되어 있어, Mac / Windows 관계없이 동일한 환경에서 실행할 수 있습니다.

---

## 📦 프로젝트 목표

* 팀원 간 **환경 차이 제거**
* 다양한 모델을 **같은 데이터셋으로 비교**
* 실험 과정을 **자동화 (Makefile + Docker)**

---

## 🧱 전체 구조

```
.
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── environment.yml
├── src/
│   ├── train.py
│   ├── registry.py
│   └── models/
│       ├── stackdili/  (StackDILI clone)
│       │   ├── model.py
│       │   ├── Code/
│       │   ├── Feature/
│       │   └── Model/
│       └── my_model/  (우리가 개발할 모델을 여기로 추가)
```

---

## 🚀 처음 시작하는 방법 (중요 ⭐)

### 1️⃣ Docker 설치

👉 아래 사이트에서 Docker Desktop 설치

* [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

설치 후 실행까지 해줘야 함. 

---

### 2️⃣ 프로젝트 클론

```
git clone <repo-url>
cd Dili
```

---

### 3️⃣ 이미지 빌드 (최초 1회)

```
make build
```

👉 Python, RDKit, PyTorch 등 필요한 라이브러리가 모두 설치됩니다.


---

### (+) 설치된 환경 확인  

```
make env_test
```

👉 설치된 라이브러리들의 버전 정보가 나옵니다. 정상적으로 환경이 세팅되었다면 아래와 같이 출력됩니다. 
```
=== 환경 체크 시작 ===

[OK] numpy
[OK] pandas
[OK] torch
[OK] rdkit
[OK] torch_geometric
[OK] sklearn
[OK] xgboost

=== 상세 체크 ===
torch version: 2.10.0
rdkit test: True
torch_geometric OK
xgboost version: 3.2.0

=== 완료 ===
```

- 추가로 설치해야 하는 라이브러리가 있다면 백지은에게 전달해주세요! 

---

## 🧪 모델 실행 방법

### StackDILI 실행

```
make run MODEL=stackdili
```

실행 흐름:

1. ML_model.ipynb 실행 (기본 모델 학습)
2. stacking.ipynb 실행 (Stacking 수행)
3. AUC 결과 출력

예시:

```
StackDILI AUC: 0.97
```

---

## 📊 결과 확인

StackDILI 결과는 아래 파일에 저장됩니다:

```
src/models/stackdili/result.txt
```

---

## ⚙️ 자주 사용하는 명령어

### 컨테이너 접속

```
make shell
```

👉 내부에서 직접 디버깅 가능

---

### 컨테이너 정리 (에러 날 때)

```
docker compose down --remove-orphans
```

---

## 🧠 코드 구조 설명 (핵심)

### train.py

* 전체 실행 진입점
* MODEL 파라미터로 어떤 모델 실행할지 결정

---

### registry.py

* 모델을 이름으로 관리

예:

```
MODEL_REGISTRY = {
    "stackdili": StackDILI
}
```

---

### model.py (각 모델 폴더 내부)

* 실제 모델 실행 코드
* 반드시 아래 구조를 따라야 함

```
class Model:
    def train(self, _):
        pass

    def predict(self, _):
        pass
```

---

## 📁 StackDILI 구조 설명

```
stackdili/
├── Code/        # notebook 실행 위치
├── Feature/     # 데이터
├── Model/       # 학습된 모델 저장
```

---

## ⚠️ 주의사항

### 1. 경로 문제

Notebook은 Code 폴더 기준으로 실행됩니다.

따라서:

```
../Feature/Feature.csv
```

처럼 접근해야 합니다.

---

### 2. 실행 순서 중요

StackDILI는 아래 순서를 따릅니다:

1. ML_model.ipynb
2. stacking.ipynb

---

## 🧩 새로운 모델 추가 방법

### 1️⃣ 폴더 생성

```
src/models/my_model/
```

---

### 2️⃣ model.py 작성

```
class Model:
    def train(self, _):
        print("My model training")

    def predict(self, _):
        return None
```

---

### 3️⃣ registry 등록

```
from models.my_model.model import Model as MYMODEL

MODEL_REGISTRY = {
    "stackdili": STACKDILI,
    "my_model": MYMODEL
}
```

---

### 4️⃣ 실행

```
make run MODEL=my_model (우리가 개발한 모델)
make run MODEL=stackdili (실행 시 StackDILI ALU 출력됨)
```

---

## 💡 협업 방식

* 모든 코드는 GitHub에 push
* Docker 환경은 동일 → “내 컴퓨터에서는 되는데?” 문제 없음
* 모델만 추가해서 비교 가능

---

## 🙋‍♀️ 문제 발생 시

1. make build 다시 실행
2. docker compose down --remove-orphans
3. make run 재실행

그래도 안 되면 팀원에게 로그 공유 👍