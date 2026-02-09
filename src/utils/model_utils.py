import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_model(model_path: Path, metadata_path: Path, device: str = "cpu") -> Tuple[nn.Module, List[str]]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta: Dict = json.load(f)
    classes = meta["classes"]
    model = SimpleCNN(num_classes=len(classes))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, classes


def predict_image(model: nn.Module, image: Image.Image, classes: List[str]) -> Dict:
    transform = default_transform()
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    pred_idx = int(torch.argmax(logits, dim=1).item())
    return {
        "label": classes[pred_idx],
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
    }
