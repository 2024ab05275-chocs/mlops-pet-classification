import argparse
import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.model_utils import SimpleCNN


def get_dataloaders(data_dir: Path, batch_size: int = 32):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        train_ds.classes,
    )


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = correct / total if total else 0.0
    return acc, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[step 3.1] Device: {device}")
    data_dir = Path(args.data)
    print(f"[step 3.2] Loading data from {data_dir}")
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir, args.batch_size)
    print(f"[step 3.3] Classes: {classes}")

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("[step 3.4] Initializing MLflow tracking")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("pet-classifier")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": device,
        })

        for epoch in range(args.epochs):
            print(f"[step 3.5] Epoch {epoch+1}/{args.epochs}")
            model.train()
            running_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_acc, _, _ = evaluate(model, val_loader, device)
            train_loss = running_loss / max(1, len(train_loader))
            print(f"  train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        print("[step 3.6] Evaluating on test set")
        test_acc, y_true, y_pred = evaluate(model, test_loader, device)
        cm = confusion_matrix(y_true, y_pred)
        mlflow.log_metric("test_acc", test_acc)
        print(f"[step 3.7] Test accuracy: {test_acc:.4f}")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "model.pth"
        meta_path = models_dir / "metadata.json"
        torch.save(model.state_dict(), model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"classes": classes, "image_size": 224}, f)

        print("[step 3.8] Logging artifacts to MLflow")
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))

        cm_path = models_dir / "confusion_matrix.txt"
        with open(cm_path, "w", encoding="utf-8") as f:
            f.write(str(cm))
        mlflow.log_artifact(str(cm_path))

        print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
