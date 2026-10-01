# Skin Lesion Classifier

Educational binary classification of HAM10000 dermoscopic images using **PyTorch**, **MobileNetV2** transfer learning, **Grad-CAM**, and a **Streamlit** demo.

This is a computer-vision / ML portfolio project, **not a medical device**. It does not diagnose melanoma or any other disease.

## Demo

Streamlit app (local): `streamlit run streamlit_app.py`

Deployed URL: _not published yet_

## Overview

This project investigates binary classification of HAM10000 dermoscopic lesion images with transfer learning. A pretrained MobileNetV2 is adapted to a project-defined benign vs positive class, an operating threshold is chosen on **validation only**, and a single locked held-out **test** evaluation is reported. Grad-CAM provides a spatial visualization of regions that influenced the displayed class output.

The goal is an educational ML / computer-vision project, not medical diagnosis.

## Results

Locked fine-tuned MobileNetV2 on the held-out **test** set (threshold **0.29**):

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.896 |
| Sensitivity | 82.6% |
| Specificity | 76.0% |
| Balanced accuracy | 79.3% |
| Accuracy | 0.773 |
| Precision (positive) | 0.449 |
| F1 (positive) | 0.582 |

Test set: **1,505** images (**288** malignant / positive, **1,217** benign).

Confusion matrix: TN 925, FP 292, FN 50, TP 238.

95% bootstrap CIs (1,000 resamples, seed 42): ROC-AUC **0.877–0.913**; sensitivity **0.782–0.867**; specificity **0.736–0.784**.

These are experiment metrics on HAM10000, not clinical performance.

![Test ROC](docs/images/test_roc_curve.png)

![Test confusion matrix](docs/images/test_confusion_matrix.png)

## ML Pipeline

HAM10000  
→ lesion-level split (seed 42)  
→ MobileNetV2 / ImageNet  
→ frozen-backbone classifier baseline  
→ full-network fine-tuning  
→ validation threshold selection (locked **0.29**)  
→ one-shot held-out test evaluation  
→ Grad-CAM  
→ Streamlit demo

Frozen baseline best **validation** ROC-AUC ≈ **0.876**. Fine-tuning best **validation** ROC-AUC **0.921946**. Fine-tuning improved validation discrimination relative to the frozen baseline (no formal significance test).

```bash
python -m skinlesion.train --experiment frozen

python -m skinlesion.train \
  --experiment finetune \
  --init-checkpoint artifacts/mobilenetv2_frozen_baseline.pth

python -m skinlesion.threshold --checkpoint artifacts/mobilenetv2_finetuned.pth

python -m skinlesion.final_test
```

`python -m skinlesion.evaluate` reports **argmax** metrics on a chosen split (default: validation). Do not use it for the locked test numbers above; those come from `final_test` at threshold 0.29.

## Dataset

[HAM10000](https://doi.org/10.1038/sdata.2018.161): **10,015** images, **7,470** unique lesions, **seven** original `dx` categories.

| dx | count | name |
|---|---|---|
| nv | 6705 | melanocytic nevus |
| mel | 1113 | melanoma |
| bkl | 1099 | benign keratosis-like |
| bcc | 514 | basal cell carcinoma |
| akiec | 327 | actinic keratosis / intraepithelial carcinoma |
| vasc | 142 | vascular lesion |
| df | 115 | dermatofibroma |

HAM10000 **does not** provide an official binary benign/malignant label. The grouping below is an **experimental modeling choice for this project**:

- **0 Benign:** `nv`, `bkl`, `df`, `vasc`
- **1 Positive / “malignant” class:** `mel`, `bcc`, `akiec`

`akiec` combines actinic keratoses and intraepithelial carcinoma (Bowen's disease). Grouping it with the positive class is a simplification. It does **not** mean every actinic keratosis is equivalent to invasive malignancy.

Images live locally in `data/ham10000/` (gitignored).

Splits are **lesion-level** so images of the same lesion cannot leak across train / validation / test:

| split | images | lesions | benign | positive |
|---|---|---|---|---|
| train | 7,010 | 5,229 | 5,641 | 1,369 |
| validation | 1,500 | 1,120 | 1,203 | 297 |
| test | 1,505 | 1,121 | 1,217 | 288 |

Train imbalance ≈ **4.12:1**. Weighted cross-entropy (train-only balanced weights: benign **0.6213**, positive **2.5603**).

## Evaluation

Threshold **0.50 was not** used as the deployed rule. Candidates were compared on **validation** before the test set was opened. Locked rule: malignant if \(P(\text{malignant}) \ge 0.29\) (Youden J / ≥85% sensitivity with max specificity on validation).

The test set was evaluated **once** for reporting. It was not used to select the architecture, checkpoint, or threshold.

![Validation sensitivity and specificity vs threshold](docs/images/val_sensitivity_specificity_vs_threshold.png)

## Grad-CAM

Grad-CAM highlights spatial regions that influenced a **selected class output** (the displayed predicted class). It does **not** identify cancerous tissue or establish clinically meaningful reasoning.

![Grad-CAM example](docs/images/gradcam_example.png)

## Running Locally

From the `SkinLesionClassifier/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The demo expects `artifacts/mobilenetv2_finetuned.pth`. Cloning without that checkpoint is not enough to run inference. HAM10000 is not in Git; place it at `data/ham10000/` to retrain.

Optional Docker (Streamlit on port 8501):

```bash
docker build -t skin-lesion-classifier .
docker run -p 8501:8501 skin-lesion-classifier
```

## Project Structure

```
SkinLesionClassifier/
├── streamlit_app.py          # Streamlit UI
├── skinlesion/
│   ├── ui.py                 # presentation helpers / CSS
│   ├── inference.py          # threshold 0.29 prediction
│   ├── gradcam.py
│   ├── model.py
│   ├── train.py
│   ├── threshold.py          # validation-only threshold analysis
│   ├── final_test.py         # locked held-out evaluation
│   └── ...
├── artifacts/                # checkpoints (finetuned weights required for the demo)
├── data/splits/              # lesion-level CSV manifests
├── docs/images/              # README / app figures
├── requirements.txt
├── Dockerfile
└── README.md
```

## Limitations

- Binary reduction of a seven-class dataset; experimental class mapping (including `akiec`).
- Class imbalance and moderate positive-class precision on test (many false positives at the locked threshold).
- Trained only on HAM10000 dermoscopy; no external dataset or clinical validation.
- Ordinary phone photographs are out of the training distribution.
- Grad-CAM is an attribution visualization, not medical localization.
- False positives and false negatives both occur.
- Educational / research use only. Not a medical device. Not for diagnosis, treatment, or clinical decision-making.

## Disclaimer

This application is a machine-learning portfolio project and is **not a medical device**. Predictions should not be used for diagnosis, treatment, or clinical decision-making. Consult a qualified healthcare professional for medical concerns.

HAM10000 has its own data terms.
