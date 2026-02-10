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

echo '[step 1] Download dataset (Kaggle) if not present'
python scripts/download_data.py

echo '[step 2] Preprocess data (resize, split, cap 10k balanced)'
PYTHONPATH=. python scripts/preprocess.py --max-total 10000

echo '[step 3] Train model and log experiments (MLflow)'
PYTHONPATH=. python scripts/train.py --epochs 3

echo '[step 4] Done. Model saved in models/ and MLflow runs in mlruns/'
