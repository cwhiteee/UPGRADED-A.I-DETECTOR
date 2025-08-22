import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import cv2

st.set_page_config(page_title="AI Detector", page_icon="🤖")

st.title("🤖 AI Image Detector (Improved)")
st.write("Upload a photo to check if it might be **AI-generated** or **camera-captured**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def analyze_image(image: Image.Image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. Sharpness / smoothness check
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Color distribution check
    color_std = np.std(img)

    # 3. Metadata check (EXIF)
    metadata_present = False
    try:
        exif_data = image._getexif()
        if exif_data is not None and len(exif_data) > 0:
            metadata_present = True
    except:
        pass

    # Combine results
    score = 0
    if laplacian_var > 50:
        score += 0.4  # sharpness suggests real
    if color_std > 30:
        score += 0.3  # color variation suggests real
    if metadata_present:
        score += 0.3  # metadata usually means real camera

    return score, laplacian_var, color_std, metadata_present


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    score, sharpness, color_std, metadata_present = analyze_image(image)

    st.subheader("🔍 Analysis Results")
    st.write(f"**Sharpness score:** {sharpness:.2f}")
    st.write(f"**Color variation:** {color_std:.2f}")
    st.write(f"**Metadata present:** {metadata_present}")

    st.subheader("📊 Final Verdict")
    if score >= 0.6:
        st.success("✅ Likely camera-captured (confidence: {:.0%})".format(score))
    else:
        st.error("⚠️ Possibly AI-generated (confidence: {:.0%})".format(1 - score))
