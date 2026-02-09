#!/usr/bin/env bash
set -euo pipefail

python scripts/create_sample_image.py

health=$(curl -s http://localhost:8000/health)
if [[ "$health" != *"ok"* ]]; then
  echo "Health check failed: $health"
  exit 1
fi

resp=$(curl -s -X POST -F "file=@scripts/sample.jpg" http://localhost:8000/predict)
if [[ "$resp" != *"label"* ]]; then
  echo "Prediction failed: $resp"
  exit 1
fi

echo "Smoke test passed"
