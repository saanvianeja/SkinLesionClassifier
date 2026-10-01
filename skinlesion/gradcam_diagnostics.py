"""Build a Grad-CAM diagnostic grid from the TRAIN split only (not for model selection).

    python -m skinlesion.gradcam_diagnostics
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from skinlesion.config import DECISION_THRESHOLD, HAM10000_DIR, OUTPUTS_DIR, SPLITS_DIR
from skinlesion.device import select_device
from skinlesion.gradcam import generate_gradcam
from skinlesion.inference import predict_from_pil
from skinlesion.model import load_model

NEEDED = {"TN": 3, "TP": 3, "FP": 2, "FN": 2}
OUT_DIR = OUTPUTS_DIR / "gradcam_diagnostics"


def _cell(img: Image.Image, overlay: Image.Image, caption: str, size=(280, 210)) -> Image.Image:
    a = img.convert("RGB").resize(size)
    b = overlay.convert("RGB").resize(size)
    canvas = Image.new("RGB", (size[0] * 2, size[1] + 48), (255, 255, 255))
    canvas.paste(a, (0, 48))
    canvas.paste(b, (size[0], 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), caption, fill=(20, 20, 20))
    return canvas


def main():
    import pandas as pd

    device = select_device()
    model, _ = load_model(device=device)
    model.eval()
    df = pd.read_csv(SPLITS_DIR / "train.csv")
    collected: dict[str, list] = {k: [] for k in NEEDED}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.sample(frac=1, random_state=0).iterrows():
        if all(len(collected[k]) >= n for k, n in NEEDED.items()):
            break
        path = HAM10000_DIR / row["image_path"]
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        result = predict_from_pil(image, model=model, device=device)
        true_idx = int(row["label"])
        pred_idx = int(result["pred_index"])
        if true_idx == 0 and pred_idx == 0:
            key = "TN"
        elif true_idx == 1 and pred_idx == 1:
            key = "TP"
        elif true_idx == 0 and pred_idx == 1:
            key = "FP"
        else:
            key = "FN"
        if len(collected[key]) >= NEEDED[key]:
            continue
        overlay, _, _ = generate_gradcam(image, model=model, device=device, target_index=pred_idx)
        rec = {
            "image_id": row["image_id"],
            "true": "Malignant" if true_idx else "Benign",
            "pred": result["pred_label"],
            "p_mal": result["malignant_probability"],
            "image": image,
            "overlay": overlay,
            "outcome": key,
        }
        collected[key].append(rec)
        rec["overlay"].save(OUT_DIR / f"{key}_{len(collected[key])}_{row['image_id']}_cam.png")
        image.save(OUT_DIR / f"{key}_{len(collected[key])}_{row['image_id']}_orig.png")

    rows_out = []
    for key in ("TN", "TP", "FP", "FN"):
        for rec in collected[key]:
            rows_out.append(
                {
                    "outcome": rec["outcome"],
                    "image_id": rec["image_id"],
                    "true_class": rec["true"],
                    "predicted_class": rec["pred"],
                    "malignant_probability": rec["p_mal"],
                    "threshold": DECISION_THRESHOLD,
                    "split": "train",
                }
            )
    pd.DataFrame(rows_out).to_csv(OUT_DIR / "examples.csv", index=False)

    cells = []
    for key in ("TN", "TP", "FP", "FN"):
        for rec in collected[key]:
            cap = (
                f"{key}  true={rec['true']}  pred={rec['pred']}  "
                f"P(mal)={rec['p_mal']:.3f}  {rec['image_id']}"
            )
            cells.append(_cell(rec["image"], rec["overlay"], cap))
    if cells:
        w, h = cells[0].size
        grid = Image.new("RGB", (w, h * len(cells)), (255, 255, 255))
        for i, cell in enumerate(cells):
            grid.paste(cell, (0, i * h))
        grid.save(OUT_DIR / "diagnostic_grid.png")
        cells[0].save(OUT_DIR / "readme_example.png")
    print("saved", OUT_DIR, {k: len(v) for k, v in collected.items()})


if __name__ == "__main__":
    main()
