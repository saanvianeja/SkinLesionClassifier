"""Reusable metric helpers. Does not split data or invent a test set.

The original training CSV is not in this repository. Do not report performance
numbers until the dataset and split strategy have been inspected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skinlesion.config import CLASS_INDEX_TO_NAME, DATA_DIR, OUTPUTS_DIR


def compute_classification_metrics(y_true, y_pred, labels=(0, 1)) -> dict:
    """Compute common binary metrics from already-defined predictions."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    target_names = [CLASS_INDEX_TO_NAME[i] for i in labels]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_recall": {
            CLASS_INDEX_TO_NAME[i]: float(recall)
            for i, recall in zip(labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        ),
        "label_mapping_note": (
            "CLASS_INDEX_TO_NAME is an unverified compatibility mapping from the old "
            "inference code. Confirm against the original training CSV before interpreting metrics."
        ),
    }


def load_split_csv(csv_path: str | Path):
    """Load a caller-provided split file. Does not create splits."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    import pandas as pd

    return pd.read_csv(path)


def dataset_available(data_dir: Path | None = None) -> bool:
    root = Path(data_dir) if data_dir else DATA_DIR
    labels = root / "isic_labels.csv"
    return labels.exists()


def main():
    print("Evaluation infrastructure only — no split strategy is defined yet.")
    print(f"Expected labels file (not required for this command): {DATA_DIR / 'isic_labels.csv'}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    if not dataset_available():
        print("No local ISIC CSV found. Metrics will not be computed.")
        print("Provide inspected split CSVs and predictions later; do not generate ad-hoc splits here.")
        return
    print("Labels file is present, but this script will not auto-split or score until a split protocol is chosen.")


if __name__ == "__main__":
    main()
