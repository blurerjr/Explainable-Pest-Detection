import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import cv2

# tf-keras-vis imports
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

# Custom CSS for a nicer look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #28a745; color: white; border: none;}
    .stFileUploader {border: 2px dashed #6c757d; padding: 20px; text-align: center; border-radius: 10px;}
    .prediction {font-size: 28px; font-weight: bold; color: #155724; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# Model URL and local path
model_url = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
model_path = "best_pest_model.keras"

# Download model if not present
if not os.path.exists(model_path):
    with st.spinner("Downloading model weights..."):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("Model downloaded!")
        else:
            st.error(f"Download failed (status: {response.status_code})")
            st.stop()

# Load model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Pest classes
class_names = [
    'aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar', 'earthworms',
    'grasshopper', 'mites', 'moth', 'sawfly', 'stem_borer', 'wasp', 'weevil'
]

# Improved bounding box extraction from heatmap
def localize_pest(img_cv, heatmap, threshold=0.5, min_area_ratio=0.005):
    h, w = img_cv.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = cv2.normalize(heatmap_resized, None, 0, 1, cv2.NORM_MINMAX)
    
    _, mask = cv2.threshold(heatmap_resized, threshold, 1, cv2.THRESH_BINARY)
    mask = (mask * 255).astype(np.uint8)
    
    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, img_cv
    
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area_ratio * h * w]
    if not filtered_contours:
        return None, img_cv
    
    largest_contour = max(filtered_contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest_contour)
    
    # Slightly expand box for better visibility
    expand = 15
    x = max(0, x - expand)
    y = max(0, y - expand)
    w_box += 2 * expand
    h_box += 2 * expand
    x2 = min(w, x + w_box)
    y2 = min(h, y + h_box)
    
    boxed_img = img_cv.copy()
    cv2.rectangle(boxed_img, (x, y), (x2, y2), (0, 255, 0), 3)
    
    cropped = img_cv[y:y2, x:x2]
    return cropped, boxed_img

# App layout
st.set_page_config(page_title="Pest Detector", layout="wide", page_icon="🦟")
st.title("🦟 Pest Detection with Bounding Box")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    show_localization = st.checkbox("Show Localized Pest (Crop)", value=True)
    show_probs = st.checkbox("Show All Class Probabilities", value=False)
    box_threshold = st.slider("Box Detection Threshold", 0.3, 0.8, 0.50, 0.05)
    min_area_ratio = st.slider("Min Area Ratio", 0.001, 0.03, 0.005, 0.001)

# Two-column layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Drop image here or click to browse", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing..."):
            # Preprocess
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array_exp = np.expand_dims(img_array, axis=0)  # batch

            # Predict
            predictions = model.predict(img_array_exp)
            pred_index = np.argmax(predictions[0])
            predicted_class = class_names[pred_index]
            confidence = predictions[0][pred_index] * 100

            # Use tf-keras-vis Grad-CAM
            try:
                gradcam = Gradcam(model)
                score = CategoricalScore([pred_index])  # target the predicted class
                cam = gradcam(score, img_array_exp, penultimate_layer=-1, seek_penultimate_conv_layer=True)
                heatmap = cam[0]  # shape: (H, W)

                # Prepare OpenCV image
                img_cv = np.array(img_resized)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

                # Localize → get box and cropped region
                cropped, boxed_img = localize_pest(
                    img_cv, 
                    heatmap, 
                    threshold=box_threshold, 
                    min_area_ratio=min_area_ratio
                )

                # Show image with bounding box
                boxed_rgb = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)
                st.image(boxed_rgb, caption=f"Detected: {predicted_class}", use_column_width=True)

                if show_localization and cropped is not None:
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    st.image(cropped_rgb, caption=f"Localized {predicted_class} (cropped)", width=300)

            except Exception as e:
                st.warning(f"Grad-CAM failed: {e}\nFalling back to prediction only.")

with col2:
    if uploaded_file is not None:
        st.subheader("Result")
        st.markdown(f"<div class='prediction'>{predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%", unsafe_allow_html=True)

        if show_probs:
            with st.expander("Detailed Probabilities"):
                for i, prob in enumerate(predictions[0]):
                    st.write(f"{class_names[i]}: {prob*100:.2f}%")

        # Download option
        if st.button("Download Result Summary"):
            summary = f"Pest: {predicted_class}\nConfidence: {confidence:.2f}%"
            st.download_button("Download", summary, file_name="pest_result.txt")

st.markdown("---")
st.caption("Powered by your custom model + tf-keras-vis Grad-CAM for explainability")
