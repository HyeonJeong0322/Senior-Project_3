# 환경 설정 가이드

처음 프로젝트를 세팅할 때 이 문서를 따라하세요.

---

## 1. Docker Desktop 설치

[https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

설치 후 Docker Desktop을 **실행(켜둔) 상태**로 유지해야 합니다.

---

## 2. 프로젝트 클론

```bash
git clone <repo-url>
cd Dili
```

---

## 3. 이미지 빌드 (최초 1회)

**Mac / Linux**
```bash
./run.sh build
```

**Windows (Git Bash 또는 WSL)**
```bash
bash run.sh build
```

Python, RDKit, PyTorch, XGBoost 등 필요한 라이브러리가 모두 설치됩니다.
처음 빌드는 몇 분 걸릴 수 있습니다.

---

## 4. 환경 확인

```bash
./run.sh env-test       # Mac/Linux
bash run.sh env-test    # Windows
```

정상 설치 시 아래와 같이 출력됩니다.

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
torch version: 2.x.x
rdkit test: True
torch_geometric OK
xgboost version: x.x.x

=== 완료 ===
```

설치가 안 된 라이브러리가 있으면 백지은에게 알려주세요.

---

## 5. 데이터 준비

모델 실행 전에 데이터 파일을 아래 경로에 넣어야 합니다.

```
data/dilirank.csv
```

파일이 없으면 모델 실행 시 오류가 발생합니다.

---

## 문제 발생 시

**빌드 실패 또는 실행 오류**
```bash
docker compose down --remove-orphans
./run.sh build
```

**컨테이너 내부에서 직접 확인하고 싶을 때**
```bash
./run.sh shell      # Mac/Linux
bash run.sh shell   # Windows
```

그래도 해결 안 되면 터미널 로그를 팀원에게 공유해주세요.
