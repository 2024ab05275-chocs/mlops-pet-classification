#!/usr/bin/env bash
set -euo pipefail


kill_port() {
  local port=$1
  local pid
  pid=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "[info] Port $port in use by PID $pid. Killing..."
    kill -9 $pid || true
  fi
}

cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${KAGGLE_USERNAME:?KAGGLE_USERNAME not set}"
: "${KAGGLE_KEY:?KAGGLE_KEY not set}"
: "${KAGGLE_DATASET:=bhavikjikadara/dog-and-cat-classification-dataset}"

if [[ -n "${DVC_GDRIVE_SERVICE_ACCOUNT_JSON:-}" ]]; then
  echo "[step 0.2] Configure DVC GDrive service account"
  bash scripts/configure_dvc_gdrive.sh
fi

echo "[step 0] Run code quality checks"
bash scripts/run_lint.sh

echo "[step 0.1] Run unit tests"
PYTHONPATH=. pytest -q

echo "[step 1] Download dataset (Kaggle) if not present"
python scripts/download_data.py

echo "[step 2] Preprocess data (resize, split, cap 10k balanced)"
PYTHONPATH=. python scripts/preprocess.py --max-total 10000

echo "[step 3] Train model and log experiments (MLflow)"
PYTHONPATH=. python scripts/train.py --epochs 10 --model logreg --image-size 64 --early-stop-patience 3

echo "[step 4] Start MLflow UI on port ${MLFLOW_PORT:-5001} (Ctrl+C to stop)"
MLFLOW_PORT=${MLFLOW_PORT:-5001}
kill_port $MLFLOW_PORT
mlflow ui --backend-store-uri ./mlruns --port $MLFLOW_PORT &
MLFLOW_PID=$!

echo "[step 5] Build Docker image"
docker build -t pet-classifier:local .

echo "[step 6] Run Docker container"
if docker ps -a --format '{{.Names}}' | grep -q '^pet-api$'; then
  docker rm -f pet-api
fi

kill_port 8000
docker run -d -p 8000:8000 --name pet-api pet-classifier:local

sleep 5

echo "[step 7] Test endpoints"
python scripts/create_sample_image.py
curl -f http://localhost:8000/health
curl -f -X POST -F "file=@scripts/sample.jpg" http://localhost:8000/predict
curl -f http://localhost:8000/metrics

echo "[step 8] Stop container and MLflow UI"
docker stop pet-api
kill $MLFLOW_PID || true

echo "[step 9] Apply Kubernetes manifests (Ingress)"
if command -v kubectl >/dev/null 2>&1; then
  kubectl apply -f deploy/k8s/deployment.yaml
  kubectl apply -f deploy/k8s/service.yaml
  kubectl apply -f deploy/k8s/ingress.yaml
  echo "K8s applied. Test: curl http://pet.local/health (ensure /etc/hosts and ingress controller)"
else
  echo "kubectl not found; skipping K8s apply"
fi

echo "Pipeline completed. Model saved in models/ and MLflow runs in mlruns/"
