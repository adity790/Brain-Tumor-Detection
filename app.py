import streamlit as st
from PIL import Image
import os
from predict import predict_image

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload a **JPG brain MRI image** to detect tumor")

uploaded_file = st.file_uploader(
    "Upload Image (JPG only)", 
    type=["jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image temporarily
    os.makedirs("temp", exist_ok=True)
    image_path = f"temp/{uploaded_file.name}"
    image.save(image_path)

    if st.button("🔍 Detect Tumor"):
        label, confidence = predict_image(image_path)

        st.subheader("Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        if "tumor" in label.lower():
            st.error("✅ No Brain Tumor Detected")
        else:
            st.success("⚠️ Brain Tumor Detected") 
