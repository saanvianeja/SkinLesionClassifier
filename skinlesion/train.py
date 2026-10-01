"""Training loop extracted from the original script. Does not invent dataset splits.

Run only when you already have train/val CSV files from an inspected labeling process:

    python -m skinlesion.train --train-csv data/train.csv --val-csv data/val.csv --img-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from skinlesion.config import DEFAULT_WEIGHTS_PATH, NUM_CLASSES
from skinlesion.dataset import ISICDataset
from skinlesion.model import create_model
from skinlesion.transforms import eval_transforms, train_transforms


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs: int = 10,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
):
    if device is None:
        device = torch.device("cpu")
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_WEIGHTS_PATH

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        print("-" * 50)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path}  F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses, val_accuracies, val_f1_scores


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on provided train/val CSVs.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    if not args.train_csv.exists() or not args.val_csv.exists():
        raise FileNotFoundError(
            "Train/val CSVs were not found. Split strategy is not defined in this project yet; "
            "provide inspected split files rather than generating them here."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ISICDataset(args.train_csv, args.img_dir, train_transforms())
    val_dataset = ISICDataset(args.val_csv, args.img_dir, eval_transforms())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=args.epochs,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
