"""Educational Streamlit demo for the locked fine-tuned MobileNetV2."""

from PIL import Image
import streamlit as st

from skinlesion.config import DECISION_THRESHOLD, FINETUNED_PATH
from skinlesion.device import select_device
from skinlesion.gradcam import generate_gradcam
from skinlesion.inference import predict_from_pil
from skinlesion.model import load_model

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")


@st.cache_resource
def get_model():
    device = select_device()
    model, load_result = load_model(FINETUNED_PATH, device=device)
    return model, device, load_result


model, device, load_result = get_model()

st.title("Skin Lesion Classifier")
st.subheader("Fine-tuned MobileNetV2 on HAM10000 dermoscopic images.")
st.write(
    "An educational computer-vision **classification** project. "
    "It is not a medical device and is not intended for diagnosis or patient care."
)

if load_result.missing_keys or load_result.unexpected_keys:
    st.error("Checkpoint did not load strictly. The app will not run.")
    st.stop()

uploaded = st.file_uploader("Upload a dermoscopic lesion image (JPG, JPEG, or PNG).", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    result = predict_from_pil(image, model=model, device=device)
    overlay, _idx, _cam = generate_gradcam(
        image, model=model, device=device, target_index=result["pred_index"]
    )

    left, right = st.columns(2)
    with left:
        st.image(image, caption="Original image", use_container_width=True)
    with right:
        st.image(
            overlay,
            caption="Grad-CAM highlights image regions that most influenced the selected model output.",
            use_container_width=True,
        )

    st.markdown(f"### Prediction: {result['pred_label']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Benign probability", f"{result['benign_probability']:.1%}")
    c2.metric("Malignant probability", f"{result['malignant_probability']:.1%}")
    c3.metric("Decision threshold", f"{DECISION_THRESHOLD:.2f}")
    st.progress(min(max(result["benign_probability"], 0.0), 1.0), text="Benign")
    st.progress(min(max(result["malignant_probability"], 0.0), 1.0), text="Malignant")
    st.caption("Displayed class: malignant if P(malignant) ≥ 0.29; otherwise benign.")

with st.expander("How it works"):
    st.markdown(
        """
1. The image is resized and normalized with the same preprocessing used in training (224×224, ImageNet mean/std).
2. A pretrained **MobileNetV2** provides learned visual features.
3. That network was **fine-tuned on HAM10000**.
4. The model outputs **benign** and **malignant** class probabilities.
5. A validation-selected threshold of **0.29** converts the malignant probability into the displayed class.
6. **Grad-CAM** is a spatial visualization of regions that contributed to the selected class output. It does not locate disease or prove medically meaningful reasoning.
        """
    )

with st.expander("Model performance"):
    st.markdown(
        """
Held-out test set: **1,505** images.

These are **held-out experiment metrics on HAM10000**, not clinical performance.

| Metric | Value |
|---|---|
| ROC-AUC | 0.896 |
| Sensitivity | 82.6% |
| Specificity | 76.0% |
| Balanced accuracy | 79.3% |
| Malignant F1 | 0.582 |

The dataset was split at the lesion level so that images of the same lesion could not appear across train, validation, and test sets.
        """
    )

st.divider()
st.markdown(
    """
**Disclaimer:** This project is an educational/research demonstration and is not a medical device.
Predictions should not be used for diagnosis or treatment decisions. Consult a qualified healthcare professional for medical concerns.

HAM10000 contains dermoscopic images and originally seven diagnostic categories. This project converts them into a binary classification task as an experimental modeling choice.
Grouping `akiec` with the positive/malignant class does **not** mean it is equivalent to invasive malignancy.
Performance on HAM10000 does not establish clinical generalization. Grad-CAM does not prove that the network learned medically meaningful features.
    """
)
