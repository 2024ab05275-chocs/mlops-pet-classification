# MLOps Pet Classification (Cats vs Dogs)

End-to-end MLOps pipeline for binary image classification using open-source tools.

## Project Structure
- `data/raw` - Raw dataset (DVC tracked)
- `data/processed` - Preprocessed train/val/test splits (DVC tracked)
- `models` - Trained model artifacts
- `mlruns` - MLflow local tracking
- `src` - Core Python package
- `scripts` - CLI utilities for data, training, evaluation
- `deploy` - Deployment manifests (Docker Compose)
- `.github/workflows` - CI/CD pipelines

## Setup
1. Create a Python virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`

## Dataset Download (Kaggle)
This project expects a Kaggle dataset containing `Cat` and `Dog` image folders.

Set environment variables:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `KAGGLE_DATASET` (example: `bhavikjikadara/dog-and-cat-classification-dataset`)

Download:
`python scripts/download_data.py`

## Preprocessing
Resizes to 224x224 RGB, applies augmentation for train, and splits 80/10/10.

`python scripts/preprocess.py --input data/raw --output data/processed`

## Training (MLflow)
`python scripts/train.py --data data/processed --epochs 3`

MLflow UI:
`mlflow ui --backend-store-uri ./mlruns`

If you need a quick placeholder model for CI or local smoke tests:
`python scripts/create_dummy_model.py`

## Inference Service
Run locally:
`uvicorn src.api.main:app --host 0.0.0.0 --port 8000`

Endpoints:
- `GET /health`
- `POST /predict` (multipart image file)

## Docker
Build:
`docker build -t pet-classifier:local .`

Run:
`docker run -p 8000:8000 pet-classifier:local`

## Docker Compose (Deployment)
`docker compose -f deploy/docker-compose.yml up -d`

Smoke test:
`bash scripts/smoke_test.sh`

Post-deploy performance tracking:
`python scripts/collect_metrics.py`

## DVC
Initialize DVC:
`dvc init`

Track data:
`dvc add data/raw data/processed`

## CI/CD
GitHub Actions builds, tests, pushes to GHCR, deploys with Docker Compose, and runs smoke tests on main.

## Notes
- Do not commit large datasets to Git. Use DVC.
- Model artifacts are saved to `models/` and logged in MLflow.
