import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import cv2
from ultralytics import YOLO

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #28a745; color: white;}
    .stFileUploader {border: 2px dashed #6c757d; padding: 20px; text-align: center; border-radius: 10px;}
    .prediction {font-size: 28px; font-weight: bold; color: #155724; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# ── 1. Your Classification Model ──
MODEL_URL = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
MODEL_PATH = "best_pest_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading classification model..."):
        r = requests.get(MODEL_URL)
        if r.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
        else:
            st.error("Failed to download classification model")
            st.stop()

try:
    clf_model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Classification model load failed: {e}")
    st.stop()

CLASS_NAMES = [
    'aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar', 'earthworms',
    'grasshopper', 'mites', 'moth', 'sawfly', 'stem_borer', 'wasp', 'weevil'
]

# ── 2. YOLOv8 Segmentation Model (pre-trained) ──
@st.cache_resource
def load_yolo_seg():
    # Use nano-seg for speed on Streamlit Cloud (or 'yolov8s-seg.pt' for better accuracy)
    return YOLO("yolov8n-seg.pt")  # auto-downloads from Ultralytics release

yolo_seg = load_yolo_seg()

# ── App ──
st.set_page_config(page_title="Pest Classifier + Segmenter", layout="wide", page_icon="🦟")
st.title("🦟 Pest Classification & Segmentation")

st.info("""
Two-model pipeline:  
• Your ResNet50V2 model → classifies the pest  
• YOLOv8-seg (pre-trained) → segments the pest region (mask + box)
""")

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader("Upload pest image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing..."):
            # Step 1: Classification (your model)
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, 0)

            preds = clf_model.predict(img_array)
            pred_idx = np.argmax(preds[0])
            predicted_class = CLASS_NAMES[pred_idx]
            confidence = preds[0][pred_idx] * 100

            # Step 2: Segmentation with YOLOv8-seg
            results = yolo_seg(img_np, verbose=False)

            # Get the first result (usually the best detection)
            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data  # tensor of masks
                boxes = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes

                # Use the mask with the highest confidence (or largest area)
                if len(masks) > 0:
                    mask = masks[0].cpu().numpy()  # take first mask for simplicity
                    # Create colored overlay
                    mask_color = np.zeros_like(img_np)
                    mask_color[mask > 0.5] = (0, 255, 0)  # green mask

                    # Blend mask with original image
                    alpha = 0.4
                    overlay = cv2.addWeighted(img_np, 1 - alpha, mask_color, alpha, 0)

                    # Draw bounding box (optional)
                    if len(boxes) > 0:
                        x1, y1, x2, y2 = map(int, boxes[0][:4])
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)

                    st.image(overlay, caption=f"Segmented Pest (YOLOv8-seg mask + box)", use_column_width=True)

                    # Optional: show binary mask
                    if st.checkbox("Show binary mask only"):
                        binary_mask = (mask > 0.5).astype(np.uint8) * 255
                        st.image(binary_mask, caption="Binary Pest Mask", use_column_width=True, clamp=True)
                else:
                    st.warning("No mask detected by YOLOv8-seg")
            else:
                st.warning("No detection by YOLOv8-seg → only classification available")

with col2:
    if uploaded_file is not None:
        st.subheader("Classification Result")
        st.markdown(f"<div class='prediction'>{predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        with st.expander("All Probabilities"):
            for i, p in enumerate(preds[0]):
                st.write(f"{CLASS_NAMES[i]}: {p*100:.2f}%")

        st.caption("YOLOv8-seg provides segmentation (green mask) and bounding box (blue)")

st.caption("Classification: your custom ResNet50V2 • Segmentation: pre-trained YOLOv8n-seg (Ultralytics)")
