"""Streamlit UI for CNN-only skin lesion classification."""

from PIL import Image
import streamlit as st

from skinlesion.gradcam import generate_gradcam
from skinlesion.inference import predict_from_pil
from skinlesion.model import load_model


st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


model, load_result = get_model()

st.title("Skin Lesion Classifier")
st.caption("MobileNetV2 CNN — educational screening demo, not a medical diagnosis.")

if load_result.missing_keys or load_result.unexpected_keys:
    st.warning(
        f"Checkpoint load issue. missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
    )

uploaded = st.file_uploader("Upload a lesion image", type=["png", "jpg", "jpeg", "bmp", "webp", "gif"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    result = predict_from_pil(image, model=model)
    overlay, cam_index = generate_gradcam(image, model=model, target_index=result["pred_index"])

    st.subheader("CNN prediction")
    st.write(f"**Class:** {result['pred_label']}  \n**Index:** {result['pred_index']}")
    st.caption(
        "Class names follow the original inference mapping (0=Benign, 1=Malignant). "
        "That mapping has not been verified against the original training CSV."
    )
    st.bar_chart(result["probabilities"])
    st.write(
        {
            "logits": result["logits"],
            "probabilities": result["probabilities"],
        }
    )

    st.subheader("Grad-CAM")
    st.image(overlay, caption=f"Overlay for class index {cam_index}", use_container_width=True)

st.divider()
st.markdown(
    "**Disclaimer:** This tool is for education and portfolio demonstration only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment."
)
