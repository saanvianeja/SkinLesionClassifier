"""Classification metrics. Positive class = 1 (Malignant)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from skinlesion.labels import CLASS_INDEX_TO_NAME


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "positive_class": {"index": 1, "name": CLASS_INDEX_TO_NAME[1]},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malignant": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_malignant": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_malignant": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_class": {},
    }
    for idx, name in CLASS_INDEX_TO_NAME.items():
        metrics["per_class"][name] = {
            "precision": float(precision_score(y_true, y_pred, pos_label=idx, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=idx, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=idx, zero_division=0)),
        }
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def plot_confusion_matrix(cm_dict: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
    fig, ax = plt.subplots()
    ax.imshow(matrix)
    ax.set_xticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_yticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, int(val), ha="center", va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(y_true, y_score, path: Path, auc: float | None) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots()
    label = f"ROC (AUC={auc:.3f})" if auc is not None else "ROC"
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_history(history: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history.get("val_roc_auc", []), label="val ROC-AUC")
    axes[1].plot(epochs, history.get("val_recall_malignant", []), label="val malignant recall")
    axes[1].set_title("Validation")
    axes[1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
