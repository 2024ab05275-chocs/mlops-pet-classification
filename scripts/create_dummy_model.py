import json
from pathlib import Path

import torch

from src.utils.model_utils import SimpleCNN


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model = SimpleCNN(num_classes=2)
    torch.save(model.state_dict(), models_dir / "model.pth")
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"classes": ["Cat", "Dog"], "image_size": 224}, f)
    print("Dummy model created")


if __name__ == "__main__":
    main()
