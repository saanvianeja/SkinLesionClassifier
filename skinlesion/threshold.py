"""Validation-only malignant-probability threshold analysis.

    python -m skinlesion.threshold --checkpoint artifacts/mobilenetv2_finetuned.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from skinlesion.config import FINETUNED_PATH, OUTPUTS_DIR, SEED, SPLITS_DIR
from skinlesion.dataset import HAM10000Dataset
from skinlesion.device import select_device
from skinlesion.metrics import plot_roc, save_json
from skinlesion.model import load_model
from skinlesion.seed import dataloader_generator, seed_worker, set_seed
from skinlesion.transforms import eval_transforms

VAL_CSV = SPLITS_DIR / "val.csv"
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def _assert_val_only(split_csv: Path) -> None:
    resolved = split_csv.resolve()
    if "test" in resolved.name.lower() or resolved.name.lower() == "test.csv":
        raise ValueError("Refusing to use the test split for threshold analysis.")
    if resolved != VAL_CSV.resolve():
        raise ValueError(f"Threshold analysis is validation-only. Expected {VAL_CSV}, got {split_csv}")


def collect_val_scores(checkpoint: Path, batch_size: int = 16):
    _assert_val_only(VAL_CSV)
    set_seed(SEED)
    device = select_device()
    model, load_result = load_model(checkpoint, device=device)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {load_result}")
    model.eval()
    loader = DataLoader(
        HAM10000Dataset(VAL_CSV, transform=eval_transforms()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(SEED),
    )
    ys, scores = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation scores", leave=True):
            images = images.to(device)
            logits = model(images)
            ys.extend(labels.numpy())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.asarray(ys), np.asarray(scores)


def metrics_at_threshold(y_true, y_score, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_accuracy": 0.5 * (sens + spec),
        "youden_j": sens + spec - 1.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _best_for_min_sensitivity(rows: list[dict], min_sens: float) -> dict | None:
    eligible = [r for r in rows if r["sensitivity"] >= min_sens]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r["specificity"], r["sensitivity"], r["f1"]))


def plot_sens_spec(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots()
    ax.plot(t, [r["sensitivity"] for r in rows], label="Sensitivity (malignant)")
    ax.plot(t, [r["specificity"] for r in rows], label="Specificity")
    ax.set_xlabel("Threshold (P(malignant))")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyze(checkpoint: Path, out_dir: Path, batch_size: int = 16) -> dict:
    y_true, y_score = collect_val_scores(checkpoint, batch_size=batch_size)
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y_true, y_score))
    rows = [metrics_at_threshold(y_true, y_score, t) for t in THRESHOLDS]
    by_t = {round(r["threshold"], 2): r for r in rows}

    youden = max(rows, key=lambda r: (r["youden_j"], r["sensitivity"]))
    f1_best = max(rows, key=lambda r: (r["f1"], r["sensitivity"]))
    at_50 = by_t[0.50]
    at_85 = _best_for_min_sensitivity(rows, 0.85)
    at_90 = _best_for_min_sensitivity(rows, 0.90)

    # Screening-style default: ≥85% malignant sensitivity, then highest specificity.
    # Fall back to Youden if 85% is unreachable on this grid.
    recommended = at_85 or youden
    recommendation_reason = (
        "Highest-specificity threshold on the validation grid with malignant sensitivity "
        "≥ 85%, to favor catching more malignant cases without collapsing specificity. "
        "Accuracy was not used as the selection criterion."
        if at_85
        else "No grid point reached 85% malignant sensitivity; Youden's J was used instead."
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "threshold_metrics.csv", index=False)
    plot_roc(y_true, y_score, out_dir / "val_roc_curve.png", auc)
    plot_sens_spec(rows, out_dir / "sensitivity_specificity_vs_threshold.png")

    summary = {
        "split": str(VAL_CSV),
        "checkpoint": str(checkpoint),
        "n_images": int(len(y_true)),
        "n_malignant": int((y_true == 1).sum()),
        "roc_auc": auc,
        "candidates": {
            "threshold_0.50": at_50,
            "youden_j": youden,
            "max_f1": f1_best,
            "sens_at_least_0.85_max_specificity": at_85,
            "sens_at_least_0.90_max_specificity": at_90,
        },
        "recommended_threshold": recommended["threshold"],
        "recommended_metrics": recommended,
        "recommendation_reason": recommendation_reason,
        "test_data_used": False,
    }
    save_json(summary, out_dir / "threshold_analysis.json")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=FINETUNED_PATH)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "finetune" / "val_threshold")
    args = p.parse_args()
    summary = analyze(args.checkpoint, args.out_dir, args.batch_size)
    print(f"validation ROC-AUC={summary['roc_auc']:.6f}")
    print(f"recommended threshold={summary['recommended_threshold']}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
