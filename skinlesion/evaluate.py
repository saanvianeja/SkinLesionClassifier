"""Evaluate a checkpoint on a fixed split. Never used for model selection.

    python -m skinlesion.evaluate --checkpoint artifacts/mobilenetv2_frozen_baseline.pth --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from skinlesion.config import OUTPUTS_DIR, SEED, SPLITS_DIR
from skinlesion.dataset import HAM10000Dataset
from skinlesion.device import select_device
from skinlesion.metrics import compute_metrics, plot_confusion_matrix, plot_roc, save_json
from skinlesion.model import load_model
from skinlesion.seed import dataloader_generator, seed_worker, set_seed
from skinlesion.transforms import eval_transforms


def evaluate_split(checkpoint: Path, split_csv: Path, out_dir: Path, batch_size: int = 16, device=None):
    set_seed(SEED)
    device = select_device(str(device) if device is not None else None)
    model, load_result = load_model(checkpoint, device=device)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {load_result}")

    dataset = HAM10000Dataset(split_csv, transform=eval_transforms())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(SEED),
    )
    ys, preds, scores = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            ys.extend(labels.numpy())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    metrics = compute_metrics(ys, preds, scores)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / "metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    if metrics["roc_auc"] is not None:
        plot_roc(ys, scores, out_dir / "roc_curve.png", metrics["roc_auc"])
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    split_csv = args.split_csv or (SPLITS_DIR / f"{args.split}.csv")
    out_dir = args.out_dir or (OUTPUTS_DIR / args.checkpoint.stem / args.split)
    metrics = evaluate_split(args.checkpoint, split_csv, out_dir, args.batch_size)
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(
        f"acc={metrics['accuracy']:.4f} mal_rec={metrics['recall_malignant']:.4f} "
        f"spec={metrics['specificity']:.4f} f1={metrics['f1_malignant']:.4f} auc={metrics['roc_auc']}"
    )


if __name__ == "__main__":
    main()
