"""Lesion-level train/val/test manifests. Run once; training must not regenerate these."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from skinlesion.config import HAM10000_DIR, SEED, SPLITS_DIR
from skinlesion.dataset import build_image_index
from skinlesion.labels import dx_to_label


def _assert_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for col in ("lesion_id", "image_id"):
        a, b, c = set(train[col]), set(val[col]), set(test[col])
        if a & b or a & c or b & c:
            raise AssertionError(f"Overlap on {col}: train∩val={a&b} train∩test={a&c} val∩test={b&c}")


def _split_stats(name: str, df: pd.DataFrame) -> dict:
    n = len(df)
    n_mal = int((df["label"] == 1).sum())
    n_ben = n - n_mal
    return {
        "split": name,
        "images": n,
        "lesions": int(df["lesion_id"].nunique()),
        "benign": n_ben,
        "benign_pct": round(100 * n_ben / n, 2) if n else 0.0,
        "malignant": n_mal,
        "malignant_pct": round(100 * n_mal / n, 2) if n else 0.0,
    }


def create_splits(
    metadata_csv: Path | None = None,
    ham_root: Path | None = None,
    out_dir: Path | None = None,
    seed: int = SEED,
) -> dict:
    ham_root = Path(ham_root) if ham_root else HAM10000_DIR
    metadata_csv = Path(metadata_csv) if metadata_csv else ham_root / "HAM10000_metadata.csv"
    out_dir = Path(out_dir) if out_dir else SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    index = build_image_index(ham_root)
    meta["label"] = meta["dx"].map(dx_to_label)
    if meta["label"].isna().any():
        bad = meta.loc[meta["label"].isna(), "dx"].unique().tolist()
        raise ValueError(f"Unmapped dx values: {bad}")
    meta["label"] = meta["label"].astype(int)

    missing = [i for i in meta["image_id"] if i not in index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} metadata rows have no image file, e.g. {missing[:3]}")

    rel_paths = []
    for image_id in meta["image_id"]:
        path = index[image_id]
        rel_paths.append(str(path.relative_to(ham_root)))
    meta["image_path"] = rel_paths

    lesions = meta.groupby("lesion_id", as_index=False).agg(label=("label", "first"))
    train_les, temp_les = train_test_split(
        lesions, test_size=0.30, random_state=seed, stratify=lesions["label"]
    )
    val_les, test_les = train_test_split(
        temp_les, test_size=0.50, random_state=seed, stratify=temp_les["label"]
    )

    train = meta[meta["lesion_id"].isin(train_les["lesion_id"])].copy()
    val = meta[meta["lesion_id"].isin(val_les["lesion_id"])].copy()
    test = meta[meta["lesion_id"].isin(test_les["lesion_id"])].copy()
    _assert_no_overlap(train, val, test)

    cols = ["lesion_id", "image_id", "dx", "label", "image_path", "dx_type", "age", "sex", "localization"]
    cols = [c for c in cols if c in meta.columns]
    stats = []
    for name, df in (("train", train), ("val", val), ("test", test)):
        df[cols].to_csv(out_dir / f"{name}.csv", index=False)
        s = _split_stats(name, df)
        stats.append(s)
        print(
            f"{name}: images={s['images']} lesions={s['lesions']} "
            f"benign={s['benign']} ({s['benign_pct']}%) "
            f"malignant={s['malignant']} ({s['malignant_pct']}%)"
        )
    print("lesion/image overlap: none")
    return {"stats": stats, "out_dir": str(out_dir)}


def main():
    create_splits()


if __name__ == "__main__":
    main()
