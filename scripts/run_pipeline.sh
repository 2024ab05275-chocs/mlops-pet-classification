#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${KAGGLE_USERNAME:?KAGGLE_USERNAME not set}"
: "${KAGGLE_KEY:?KAGGLE_KEY not set}"
: "${KAGGLE_DATASET:=bhavikjikadara/dog-and-cat-classification-dataset}"

python scripts/download_data.py
PYTHONPATH=. python scripts/preprocess.py --max-total 10000
PYTHONPATH=. python scripts/train.py --epochs 3

echo "Pipeline completed. Model saved in models/ and MLflow runs in mlruns/"
