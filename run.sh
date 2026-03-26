#!/usr/bin/env bash

CMD=${1}
MODEL=${2}

case "$CMD" in
  build)
    docker compose build
    ;;
  shell)
    docker compose run --rm ml bash
    ;;
  run)
    if [ -z "$MODEL" ]; then
      echo "Usage: ./run.sh run <model>"
      exit 1
    fi
    docker compose run --rm ml conda run -n dili_env python src/train.py --model="$MODEL"
    ;;
  env-test)
    docker compose run --rm ml conda run -n dili_env python src/env_test.py
    ;;
  *)
    echo "Usage: ./run.sh {build|shell|run <model>|env-test}"
    exit 1
    ;;
esac
