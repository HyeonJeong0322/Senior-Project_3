build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose run ml bash

train:
	docker compose run ml python src/train.py

data:
	docker compose run ml bash scripts/download_data.sh

run:
	docker compose run ml conda run -n dili_env python src/train.py --model=$(MODEL)

env_test:
	docker compose run ml conda run -n dili_env python src/env_test.py