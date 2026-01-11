import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Oil Spill Detection System")
st.title("🛢️ Oil Spill Detection System")

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("oil_spill_unet_model.h5")

model = load_trained_model()

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")   # grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing (UNet style)
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=(0, -1))  # (1, 256, 256, 1)

    prediction = model.predict(image)

    st.subheader("Prediction Result")

    if np.mean(prediction) > 0.5:
        st.error("⚠️ Oil Spill Detected")
    else:
        st.success("✅ No Oil Spill Detected")
