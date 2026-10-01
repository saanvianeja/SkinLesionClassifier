"""Educational Streamlit demo for the locked fine-tuned MobileNetV2."""

from PIL import Image
import streamlit as st

from skinlesion.config import DECISION_THRESHOLD, FINETUNED_PATH, ROOT
from skinlesion.device import select_device
from skinlesion.gradcam import generate_gradcam
from skinlesion.inference import predict_from_pil
from skinlesion.model import load_model
from skinlesion.ui import (
    disclaimer,
    hero,
    how_it_works_card,
    inject_css,
    performance_cards,
    prediction_card,
    probability_bars,
)

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()


@st.cache_resource
def get_model():
    device = select_device()
    model, load_result = load_model(FINETUNED_PATH, device=device)
    return model, device, load_result


model, device, load_result = get_model()
hero()

if load_result.missing_keys or load_result.unexpected_keys:
    st.error("Checkpoint did not load strictly. The app will not run.")
    st.stop()

left, right = st.columns(2, gap="large")
with left:
    st.markdown('<div class="card"><h3>Upload Image</h3><p>Upload a dermoscopic lesion image (JPG, JPEG, or PNG).</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag and drop a dermoscopic lesion image, or browse.",
        type=["jpg", "jpeg", "png"],
    )
    st.caption("The model was trained on dermoscopic images from HAM10000 and is not designed for ordinary phone or clinical photographs.")
with right:
    how_it_works_card()

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    result = predict_from_pil(image, model=model, device=device)
    overlay, _idx, _cam = generate_gradcam(
        image, model=model, device=device, target_index=result["pred_index"]
    )

    st.markdown('<h2 class="section-title">Analysis Results</h2>', unsafe_allow_html=True)
    img_l, img_r = st.columns(2, gap="large")
    with img_l:
        st.markdown('<div class="card"><h3>Uploaded Image</h3></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with img_r:
        st.markdown('<div class="card"><h3>Grad-CAM activation map</h3></div>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
        st.caption(
            "Warmer regions indicate areas that contributed more strongly to the model's prediction. "
            "Grad-CAM does not identify cancerous tissue."
        )

    prediction_card(
        result["pred_label"],
        result["benign_probability"],
        result["malignant_probability"],
        DECISION_THRESHOLD,
    )
    probability_bars(result["benign_probability"], result["malignant_probability"])

    with st.expander("Developer Details"):
        st.write(f"Device: `{device}`")
        st.write(f"Decision threshold: `{result['decision_threshold']}`")
        st.write(f"Predicted class index: `{result['pred_index']}`")

st.markdown("### Model Performance")
st.caption("Held-out test set results for the locked fine-tuned MobileNetV2. These numbers do not come from the uploaded image.")
performance_cards()

fig_l, fig_r = st.columns(2, gap="large")
roc_path = ROOT / "docs" / "images" / "test_roc_curve.png"
cm_path = ROOT / "docs" / "images" / "test_confusion_matrix.png"
with fig_l:
    if roc_path.exists():
        st.image(str(roc_path), caption="Held-out test ROC curve", use_container_width=True)
with fig_r:
    if cm_path.exists():
        st.image(str(cm_path), caption="Held-out test confusion matrix", use_container_width=True)

with st.expander("About the Model"):
    st.markdown(
        """
- **MobileNetV2** from torchvision, initialized with **ImageNet** pretrained weights.
- Transfer learning: a frozen-backbone baseline trained only the classifier, then the **full network was fine-tuned** at a lower backbone learning rate.
- Training used **class-weighted** cross-entropy because the train set is imbalanced (~4.12:1).
- Checkpoint selection used **validation ROC-AUC** with **early stopping**.
- Displayed class: **malignant if P(malignant) ≥ 0.29**.
        """
    )

with st.expander("Dataset & Evaluation"):
    st.markdown(
        """
- **HAM10000**: 10,015 dermoscopic images, originally seven diagnostic categories.
- Splits are **lesion-level** so images of the same lesion cannot appear across train, validation, and test.
- **Validation** was used for model and threshold selection.
- The **test set was used once** for final evaluation (1,505 images: 288 malignant, 1,217 benign).
        """
    )

with st.expander("What is Grad-CAM?"):
    st.markdown(
        """
Grad-CAM is a spatial attribution visualization. It highlights regions that influenced a selected class output of the neural network.

It does **not** identify malignant tissue or provide a medical explanation.
        """
    )

with st.expander("Limitations"):
    st.markdown(
        """
- Trained on HAM10000 **dermoscopic** images; performance may not generalize to ordinary phone photographs or other image distributions.
- The binary mapping is an **experimental simplification** of HAM10000's original diagnostic categories (including grouping `akiec` with the positive class).
- Predictions are **not medical diagnoses**.
- Performance varies with image quality and distribution.
- This model should **not** be used for clinical decision-making.
        """
    )

disclaimer()
