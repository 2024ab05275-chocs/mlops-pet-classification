import json
from pathlib import Path

import torch
from PIL import Image

from src.utils.model_utils import LogisticRegression, load_model, predict_image


def test_load_logreg_and_predict(tmp_path: Path):
    model = LogisticRegression(input_dim=3 * 64 * 64, num_classes=2)
    model_path = tmp_path / "model.pth"
    meta_path = tmp_path / "metadata.json"

    torch.save(model.state_dict(), model_path)
    meta_path.write_text(json.dumps({"classes": ["Cat", "Dog"], "image_size": 64, "model_type": "logreg"}))

    loaded, classes, image_size = load_model(model_path, meta_path)
    assert classes == ["Cat", "Dog"]
    assert image_size == 64

    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    result = predict_image(loaded, img, classes, image_size=image_size)
    assert "label" in result
    assert "probabilities" in result
