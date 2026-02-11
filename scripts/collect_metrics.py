import json
from pathlib import Path

import os
import requests
from PIL import Image


def main():
    sample = Path("scripts/sample.jpg")
    sample.parent.mkdir(parents=True, exist_ok=True)
    if not sample.exists():
        img = Image.new("RGB", (224, 224), color=(120, 180, 200))
        img.save(sample)
    # Simulated requests with dummy labels
    api_port = os.environ.get("API_PORT", "8002")
    samples = [
        {"path": "scripts/sample.jpg", "label": "Cat"},
        {"path": "scripts/sample.jpg", "label": "Dog"},
    ]

    results = []
    for s in samples:
        with open(s["path"], "rb") as f:
            resp = requests.post(f"http://localhost:{api_port}/predict", files={"file": f})
        results.append({
            "true_label": s["label"],
            "prediction": resp.json().get("label"),
            "probabilities": resp.json().get("probabilities"),
        })

    out = Path("reports/post_deploy_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
