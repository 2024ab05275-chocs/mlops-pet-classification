import argparse
import json
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.model_utils import SimpleCNN


def get_dataloaders(data_dir: Path, batch_size: int = 32, image_size: int = 224, augment: bool = True):
    train_tf = [transforms.Resize((image_size, image_size))]
    if augment:
        train_tf += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    train_tf += [transforms.ToTensor()]
    train_transform = transforms.Compose(train_tf)
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
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
    all_probs = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    acc = correct / total if total else 0.0
    return acc, all_labels, all_preds, all_probs


def save_confusion_matrix(cm, classes, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_roc_curve(y_true, y_score, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return roc_auc


def save_loss_curve(losses, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(losses, label="train_loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "logreg"])
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[step 3.1] Device: {device}")
    data_dir = Path(args.data)
    print(f"[step 3.2] Loading data from {data_dir}")

    image_size = args.image_size
    if args.model == "logreg" and args.image_size == 224:
        image_size = 64
        print("[step 3.2.1] Using smaller image_size=64 for logreg")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir, args.batch_size, image_size=image_size, augment=(args.model == "cnn")
    )
    print(f"[step 3.3] Classes: {classes}")

    if args.model == "logreg":
        input_dim = 3 * image_size * image_size
        model = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, len(classes))).to(device)
    else:
        model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("[step 3.4] Initializing MLflow tracking")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("pet-classifier")

    start_time = time.time()

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": device,
            "model": args.model,
            "image_size": image_size,
        })

        train_losses = []
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

            val_acc, _, _, _ = evaluate(model, val_loader, device)
            train_loss = running_loss / max(1, len(train_loader))
            train_losses.append(train_loss)
            print(f"  train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        print("[step 3.6] Evaluating on test set")
        test_acc, y_true, y_pred, y_probs = evaluate(model, test_loader, device)
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        print(f"[step 3.7] Test accuracy: {test_acc:.4f}")
        print(f"[step 3.7.1] precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "model.pth"
        meta_path = models_dir / "metadata.json"
        torch.save(model.state_dict(), model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"classes": classes, "image_size": image_size, "model_type": args.model}, f)

        print("[step 3.8] Logging artifacts to MLflow")
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))

        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        cm_txt = reports_dir / "confusion_matrix.txt"
        cm_png = reports_dir / "confusion_matrix.png"
        with open(cm_txt, "w", encoding="utf-8") as f:
            f.write(str(cm))
        save_confusion_matrix(cm, classes, cm_png)

        y_score = [p[1] for p in y_probs] if len(y_probs) else []
        roc_png = reports_dir / "roc_curve.png"
        roc_auc = None
        if y_score:
            roc_auc = save_roc_curve(y_true, y_score, roc_png)
            mlflow.log_metric("roc_auc", roc_auc)

        loss_png = reports_dir / "train_loss.png"
        save_loss_curve(train_losses, loss_png)

        metrics_path = reports_dir / "metrics.json"
        metrics = {
            "accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2))

        mlflow.log_artifact(str(cm_txt))
        mlflow.log_artifact(str(cm_png))
        mlflow.log_artifact(str(roc_png))
        mlflow.log_artifact(str(loss_png))
        mlflow.log_artifact(str(metrics_path))

        train_time = time.time() - start_time
        mlflow.log_metric("train_time_sec", train_time)
        print(f"[step 3.9] Training time: {train_time:.2f}s")

        print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
