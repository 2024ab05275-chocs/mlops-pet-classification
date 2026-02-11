#!/usr/bin/env bash
set -euo pipefail

API_PORT=${API_PORT:-8002}
python scripts/create_sample_image.py

health=$(curl -s http://localhost:${API_PORT}/health)
if [[ "$health" != *"ok"* ]]; then
  echo "Health check failed: $health"
  exit 1
fi

resp=$(curl -s -X POST -F "file=@scripts/sample.jpg" http://localhost:${API_PORT}/predict)
if [[ "$resp" != *"label"* ]]; then
  echo "Prediction failed: $resp"
  exit 1
fi

echo "Smoke test passed"
