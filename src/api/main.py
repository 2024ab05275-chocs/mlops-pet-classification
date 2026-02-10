import io
import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.utils.model_utils import load_model, predict_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

app = FastAPI(title="Pet Classifier")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pth"
META_PATH = MODEL_DIR / "metadata.json"

REQUEST_COUNT = Counter("requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Prediction latency")

_request_count = 0
_total_latency = 0.0


@app.on_event("startup")
def _load():
    global _model, _classes, _image_size
    _model, _classes, _image_size = load_model(MODEL_PATH, META_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global _request_count, _total_latency
    start = time.time()
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    result = predict_image(_model, image, _classes, image_size=_image_size)
    latency = time.time() - start

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(latency)

    _request_count += 1
    _total_latency += latency
    logger.info(
        "request=%s latency_ms=%.2f total_requests=%d avg_latency_ms=%.2f",
        file.filename,
        latency * 1000,
        _request_count,
        (_total_latency / _request_count) * 1000,
    )

    return JSONResponse(result)
