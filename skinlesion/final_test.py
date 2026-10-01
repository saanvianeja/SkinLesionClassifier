"""ONE-SHOT held-out test evaluation. Reporting only. Threshold is locked.

    python -m skinlesion.final_test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from skinlesion.config import FINETUNED_PATH, OUTPUTS_DIR, SEED, SPLITS_DIR
from skinlesion.dataset import HAM10000Dataset
from skinlesion.device import select_device
from skinlesion.metrics import plot_confusion_matrix, plot_roc, save_json
from skinlesion.model import load_model
from skinlesion.seed import dataloader_generator, seed_worker, set_seed
from skinlesion.transforms import eval_transforms

LOCKED_CHECKPOINT = FINETUNED_PATH
LOCKED_THRESHOLD = 0.29
TEST_CSV = SPLITS_DIR / "test.csv"
OUT_DIR = OUTPUTS_DIR / "finetune" / "test"
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def _counts(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def _point_metrics(y_true, y_score, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = _counts(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "accuracy": acc,
        "balanced_accuracy": 0.5 * (sens + spec),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "pred_label": y_pred,
    }


def _bootstrap_ci(y_true, y_score, threshold: float, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, senss, specs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = _point_metrics(yt, ys, threshold)
        aucs.append(m["roc_auc"])
        senss.append(m["sensitivity"])
        specs.append(m["specificity"])

    def ci(values):
        arr = np.asarray(values)
        return {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
            "n_valid_resamples": int(len(arr)),
        }

    return {
        "roc_auc": ci(aucs),
        "sensitivity": ci(senss),
        "specificity": ci(specs),
        "n_bootstrap_requested": n_boot,
        "bootstrap_seed": seed,
    }


def main():
    if not TEST_CSV.exists():
        raise FileNotFoundError(TEST_CSV)
    set_seed(SEED)
    device = select_device()
    ckpt_bytes_before = LOCKED_CHECKPOINT.read_bytes()
    model, load_result = load_model(LOCKED_CHECKPOINT, device=device)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {load_result}")
    model.eval()

    dataset = HAM10000Dataset(TEST_CSV, transform=eval_transforms())
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(SEED),
    )
    meta = pd.read_csv(TEST_CSV)
    ys, scores = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="FINAL TEST", leave=True):
            logits = model(images.to(device))
            ys.extend(labels.numpy())
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    y_true = np.asarray(ys)
    y_score = np.asarray(scores)
    assert len(y_true) == len(meta)

    point = _point_metrics(y_true, y_score, LOCKED_THRESHOLD)
    y_pred = point.pop("pred_label")
    ci = _bootstrap_ci(y_true, y_score, LOCKED_THRESHOLD, N_BOOTSTRAP, BOOTSTRAP_SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(
        {
            "image_id": meta["image_id"],
            "true_label": y_true.astype(int),
            "malignant_probability": y_score,
            "predicted_label": y_pred.astype(int),
        }
    )
    preds.to_csv(OUT_DIR / "final_test_predictions.csv", index=False)

    report = {
        "result_type": "FINAL_TEST",
        "notes": (
            "Reporting-only held-out evaluation. Threshold 0.29 was locked from validation "
            "and was not retuned on test."
        ),
        "checkpoint": str(LOCKED_CHECKPOINT),
        "locked_threshold": LOCKED_THRESHOLD,
        "split": str(TEST_CSV),
        "n_images": int(len(y_true)),
        "n_benign": int((y_true == 0).sum()),
        "n_malignant": int((y_true == 1).sum()),
        "metrics": point,
        "bootstrap_95ci": ci,
        "weights_unchanged": ckpt_bytes_before == LOCKED_CHECKPOINT.read_bytes(),
        "threshold_unchanged": True,
    }
    save_json(report, OUT_DIR / "final_test_metrics.json")
    plot_confusion_matrix(point["confusion_matrix"], OUT_DIR / "final_test_confusion_matrix.png")
    plot_roc(y_true, y_score, OUT_DIR / "final_test_roc_curve.png", point["roc_auc"])

    cm = point["confusion_matrix"]
    print("FINAL TEST")
    print(f"n={report['n_images']} benign={report['n_benign']} malignant={report['n_malignant']}")
    print(
        f"auc={point['roc_auc']:.4f} acc={point['accuracy']:.4f} bal_acc={point['balanced_accuracy']:.4f} "
        f"sens={point['sensitivity']:.4f} spec={point['specificity']:.4f} "
        f"prec={point['precision']:.4f} f1={point['f1']:.4f}"
    )
    print(f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"weights_unchanged={report['weights_unchanged']} wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
