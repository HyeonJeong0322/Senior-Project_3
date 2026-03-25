FROM continuumio/miniconda3

WORKDIR /app

# conda 환경 복사
COPY docker/environment.yml .

# 환경 생성
RUN conda env create -f environment.yml

# bash 실행 시 자동 activate
SHELL ["conda", "run", "-n", "dili-env", "/bin/bash", "-c"]

# 코드 복사
COPY . .

CMD ["bash"]