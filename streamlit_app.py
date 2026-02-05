import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import cv2
from ultralytics import YOLO

# ── Custom CSS for nicer appearance ──
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #28a745; 
        color: white; 
        border: none;
    }
    .stFileUploader {
        border: 2px dashed #6c757d; 
        padding: 20px; 
        text-align: center; 
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .prediction {
        font-size: 28px; 
        font-weight: bold; 
        color: #155724; 
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ── Classification Model (your custom model) ──
MODEL_URL = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
MODEL_PATH = "best_pest_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading classification model..."):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            st.success("Classification model downloaded")
        else:
            st.error("Failed to download classification model")
            st.stop()

try:
    clf_model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load classification model: {e}")
    st.stop()

CLASS_NAMES = [
    'aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar', 'earthworms',
    'grasshopper', 'mites', 'moth', 'sawfly', 'stem_borer', 'wasp', 'weevil'
]

# ── YOLOv8 Segmentation Model (pre-trained) ──
@st.cache_resource
def load_yolo_model():
    # Use nano for speed on Streamlit Cloud
    # You can change to "yolov8s-seg.pt" for better accuracy (larger & slower)
    return YOLO("yolov8n-seg.pt")

yolo_model = load_yolo_model()

# ── App Configuration ──
st.set_page_config(
    page_title="Pest Classifier & Segmenter",
    layout="wide",
    page_icon="🦟"
)

st.title("🦟 Pest Classification + Segmentation")

st.markdown("""
This app uses **two models**:
- Your custom ResNet50V2 model → **classifies** the pest
- Pre-trained **YOLOv8-seg** → **segments** and draws boxes/masks automatically
""")

# ── Main Layout ──
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Drop your pest image here or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Display original image
        original_image = Image.open(uploaded_file)
        original_np = np.array(original_image)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            # ── 1. Classification ──
            img_resized = original_image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            clf_preds = clf_model.predict(img_array, verbose=0)
            pred_idx = np.argmax(clf_preds[0])
            predicted_class = CLASS_NAMES[pred_idx]
            confidence = clf_preds[0][pred_idx] * 100

            # ── 2. Segmentation with YOLOv8-seg ──
            results = yolo_model(original_np, verbose=False)

            if len(results) > 0 and results[0].masks is not None:
                # Use YOLO's built-in plotting (boxes + masks + labels)
                annotated_img = results[0].plot()  # returns BGR numpy array
                
                # Convert BGR to RGB for Streamlit
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                st.image(
                    annotated_rgb,
                    caption="YOLOv8 Segmentation Result (boxes + masks)",
                    use_column_width=True
                )
            else:
                st.warning("YOLOv8-seg did not detect any objects. Showing classification only.")
                st.image(original_np, caption="Original Image", use_column_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("Classification Result")
        st.markdown(f"<div class='prediction'>{predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%", unsafe_allow_html=True)

        with st.expander("Detailed Probabilities"):
            for i, prob in enumerate(clf_preds[0]):
                st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

        st.markdown("---")
        st.caption("YOLOv8-seg provides bounding boxes (red) and segmentation masks (colored overlay)")

# ── Footer ──
st.markdown("---")
st.caption(
    "Classification: Custom ResNet50V2 • "
    "Segmentation: Pre-trained YOLOv8n-seg (Ultralytics) • "
    "No fine-tuning performed on YOLO model"
)
