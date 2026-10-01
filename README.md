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
- Inference mapping **0 → Benign, 1 → Malignant** is what the original `predict_with_cnn` code used. **It has not been verified** against the original training CSV or label-generation process.

## Training / evaluation

Training data is not shipped. Split strategy is **not defined yet** — do not treat any auto-split as a finished protocol.

- Train (only if you already have inspected train/val CSVs):  
  `python -m skinlesion.train --train-csv ... --val-csv ... --img-dir ...`
- Metric helpers: `python -m skinlesion.evaluate` (will not invent splits or report scores without data)

## Layout

```
skinlesion/     model, transforms, dataset, train, evaluate, inference, Grad-CAM
artifacts/      saved checkpoint
streamlit_app.py
```
