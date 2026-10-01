# Skin Lesion Classifier

Educational MobileNetV2 classifier for dermoscopic / lesion photos. **Not a medical device.**

The app scores an image with a CNN only (softmax over two classes). It does not use ABCDE heuristics, metadata risk scores, or fused thresholds.

## Run locally

```bash
cd SkinLesionClassifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

Docker:

```bash
docker build -t skin-lesion-classifier .
docker run --rm -p 8501:8501 skin-lesion-classifier
```

## Model

- Architecture: MobileNetV2, classifier `Dropout(0.2) + Linear(last_channel, 2)`
- Weights: [`artifacts/best_isic_model.pth`](artifacts/best_isic_model.pth) (existing checkpoint; not retrained in this repo layout)
- Eval preprocess: resize 224×224, ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- Streamlit still loads the legacy `artifacts/best_isic_model.pth`. That file’s original training-label protocol is undocumented; new experiments use `skinlesion/labels.py`.

## Labels (HAM10000)

HAM10000 is a **7-class** diagnostic dataset (`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`). It has **no official benign/malignant binary target**.

This project uses a modeling mapping:

- **0 Benign:** `nv`, `bkl`, `df`, `vasc`
- **1 Malignant:** `mel`, `bcc`, `akiec`

`akiec` combines actinic keratoses and intraepithelial carcinoma (Bowen's disease). It is grouped into the malignant/positive class for this binary experiment only. That does **not** mean every actinic keratosis is clinically equivalent to invasive malignancy.

Lesion-level splits (seed 42) live in `data/splits/` and are not regenerated during training.

## Training / evaluation

HAM10000 images/metadata stay local (`data/ham10000/`, gitignored). The Streamlit app still loads the legacy `artifacts/best_isic_model.pth` until it is wired to a new checkpoint.

```bash
python -m skinlesion.train --experiment frozen
python -m skinlesion.evaluate --checkpoint artifacts/mobilenetv2_frozen_baseline.pth --split test
```

Do not overwrite `artifacts/best_isic_model.pth`.

## Layout

```
skinlesion/     model, transforms, dataset, train, evaluate, inference, Grad-CAM
data/splits/    fixed lesion-level train/val/test manifests
artifacts/      checkpoints (legacy + new experiments)
streamlit_app.py
```
