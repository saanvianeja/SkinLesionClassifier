"""Train MobileNetV2 on fixed HAM10000 lesion-level splits.

    python -m skinlesion.train --experiment frozen
    python -m skinlesion.train --experiment finetune --init-checkpoint artifacts/mobilenetv2_frozen_baseline.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from skinlesion.config import (
    FINETUNED_PATH,
    FROZEN_BASELINE_PATH,
    LEGACY_WEIGHTS_PATH,
    OUTPUTS_DIR,
    SEED,
    SPLITS_DIR,
)
from skinlesion.dataset import HAM10000Dataset
from skinlesion.device import select_device
from skinlesion.metrics import compute_metrics, plot_history, save_json
from skinlesion.model import (
    create_model,
    freeze_backbone,
    load_model,
    trainable_parameter_counts,
    unfreeze_backbone,
)
from skinlesion.seed import dataloader_generator, seed_worker, set_seed
from skinlesion.transforms import eval_transforms, train_transforms

IMBALANCE_RATIO_THRESHOLD = 1.5


def _refuse_legacy_overwrite(path: Path) -> None:
    if path.resolve() == LEGACY_WEIGHTS_PATH.resolve():
        raise ValueError(f"Refusing to overwrite legacy checkpoint {LEGACY_WEIGHTS_PATH}")


def class_weights_from_train(train_csv: Path, device: torch.device) -> torch.Tensor | None:
    labels = pd.read_csv(train_csv)["label"].to_numpy()
    counts = np.bincount(labels, minlength=2)
    ratio = float(counts.max() / max(counts.min(), 1))
    print(f"Train class counts: benign={int(counts[0])} malignant={int(counts[1])} ratio={ratio:.2f}")
    if ratio < IMBALANCE_RATIO_THRESHOLD:
        print("Imbalance below threshold; using unweighted CrossEntropyLoss")
        return None
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    print(f"Using balanced class weights: benign={weights[0]:.4f} malignant={weights[1]:.4f}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def collect_outputs(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, preds, scores = [], [], []
    with torch.no_grad():
        n_batches = 0
        val_bar = tqdm(loader, desc="Validation", leave=True)
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            batch_loss = criterion(logits, labels).item()
            total_loss += batch_loss
            n_batches += 1
            prob = torch.softmax(logits, dim=1)[:, 1]
            ys.extend(labels.cpu().numpy())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            scores.extend(prob.cpu().numpy())
            val_bar.set_postfix(loss=f"{total_loss / n_batches:.4f}")
    metrics = compute_metrics(ys, preds, scores)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def make_loader(csv_path, transform, batch_size, shuffle, seed):
    ds = HAM10000Dataset(csv_path, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(seed),
    )


def train_loop(model, train_loader, val_loader, criterion, optimizer, args, checkpoint_path, out_dir):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision_malignant": [],
        "val_recall_malignant": [],
        "val_specificity": [],
        "val_f1_malignant": [],
        "val_roc_auc": [],
    }
    best_auc = -1.0
    best_recall = -1.0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        running = 0.0
        n_batches = 0
        train_bar = tqdm(train_loader, desc="Train", leave=True)
        for images, labels in train_bar:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
            train_bar.set_postfix(loss=f"{running / n_batches:.4f}")
        train_loss = running / max(n_batches, 1)
        val_metrics = collect_outputs(model, val_loader, criterion, args.device)
        auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else -1.0
        rec = val_metrics["recall_malignant"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision_malignant"].append(val_metrics["precision_malignant"])
        history["val_recall_malignant"].append(rec)
        history["val_specificity"].append(val_metrics["specificity"])
        history["val_f1_malignant"].append(val_metrics["f1_malignant"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        print(
            f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"mal_rec={rec:.4f} spec={val_metrics['specificity']:.4f} "
            f"f1={val_metrics['f1_malignant']:.4f} auc={val_metrics['roc_auc']}"
        )

        improved = auc > best_auc + 1e-6 or (abs(auc - best_auc) <= 1e-6 and rec > best_recall)
        if improved:
            best_auc, best_recall = auc, rec
            patience_counter = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            save_json(val_metrics, out_dir / "best_val_metrics.json")
            print(f"Saved {checkpoint_path} (val ROC-AUC={auc}, malignant recall={rec:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    save_json(history, out_dir / "history.json")
    plot_history(history, out_dir / "history.png")
    return history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["frozen", "finetune"], required=True)
    p.add_argument("--train-csv", type=Path, default=SPLITS_DIR / "train.csv")
    p.add_argument("--val-csv", type=Path, default=SPLITS_DIR / "val.csv")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--classifier-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.device = select_device(args.device)

    if args.experiment == "frozen":
        checkpoint = args.checkpoint or FROZEN_BASELINE_PATH
        out_dir = OUTPUTS_DIR / "frozen_baseline"
        if args.init_checkpoint:
            raise ValueError("frozen experiment starts from ImageNet; omit --init-checkpoint")
        model = create_model(pretrained=True)
        freeze_backbone(model)
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.classifier_lr)
    else:
        checkpoint = args.checkpoint or FINETUNED_PATH
        out_dir = OUTPUTS_DIR / "finetune"
        init = args.init_checkpoint or FROZEN_BASELINE_PATH
        model, _ = load_model(init, device=args.device)
        unfreeze_backbone(model)
        optimizer = optim.Adam(
            [
                {"params": model.features.parameters(), "lr": args.backbone_lr},
                {"params": model.classifier.parameters(), "lr": args.classifier_lr},
            ]
        )

    _refuse_legacy_overwrite(checkpoint)
    model.to(args.device)
    n_train, n_total = trainable_parameter_counts(model)
    print(f"Trainable params {n_train}/{n_total} device={args.device}")

    weights = class_weights_from_train(args.train_csv, args.device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = make_loader(args.train_csv, train_transforms(), args.batch_size, True, args.seed)
    val_loader = make_loader(args.val_csv, eval_transforms(), args.batch_size, False, args.seed)

    config = {
        "experiment": args.experiment,
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "classifier_lr": args.classifier_lr,
        "backbone_lr": args.backbone_lr,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "class_weights": None if weights is None else weights.detach().cpu().tolist(),
        "selection_metric": "val_roc_auc (tie-break: val malignant recall)",
        "test_set_unused": True,
    }
    save_json(config, out_dir / "train_config.json")
    print(json.dumps(config, indent=2))

    train_loop(model, train_loader, val_loader, criterion, optimizer, args, checkpoint, out_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
