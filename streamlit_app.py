import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import cv2
import matplotlib.pyplot as plt

# Custom CSS for beauty
st.markdown("""
    <style>
    .main {background-color: #f0f8ff;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stFileUploader {border: 2px dashed #ccc; padding: 20px; text-align: center;}
    .prediction {font-size: 24px; font-weight: bold; color: #228B22;}
    </style>
""", unsafe_allow_html=True)

# Model URL and local path
model_url = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
model_path = "best_pest_model.keras"

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    with st.spinner("Downloading the model weights..."):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error(f"Failed to download model. Status code: {response.status_code}")
            st.stop()

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Pest classes
class_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar', 'earthworms', 'grasshopper', 'mites', 'moth', 'sawfly', 'stem_borer', 'wasp', 'weevil']

# Function to find the last convolutional layer
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to superimpose heatmap on image
def superimpose_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# Function to localize pest (find bounding box from heatmap)
def localize_pest(img_cv, heatmap, threshold=0.6):
    # Threshold the heatmap
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    mask = (heatmap_resized > threshold).astype(np.uint8) * 255
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Crop the region
        cropped = img_cv[y:y+h, x:x+w]
        # Draw bounding box on original for display
        boxed_img = cv2.rectangle(img_cv.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
        return cropped, boxed_img
    else:
        return None, img_cv  # No localization found

# App config for wide layout
st.set_page_config(page_title="Explainable Pest Detection App", layout="wide", page_icon="🦟")

# App title with emoji
st.title("🦟 Interactive Pest Detection & Classification")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    show_heatmap = st.checkbox("Show Grad-CAM Heatmap", value=True)
    show_localization = st.checkbox("Show Localized Pest", value=True)
    show_probs = st.checkbox("Show Detailed Probabilities", value=False)
    heatmap_alpha = st.slider("Heatmap Opacity", 0.1, 0.8, 0.4)
    localization_threshold = st.slider("Localization Threshold", 0.1, 0.9, 0.6)

# Main layout with two columns: Left for images, Right for results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Drop Image Here or Click to Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess
        with st.spinner("Processing..."):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_class = class_names[predicted_index]
            confidence = np.max(predictions) * 100
            
            # Grad-CAM
            try:
                last_conv_layer_name = find_last_conv_layer(model)
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_index)
                
                # Prepare CV image
                img_cv = np.array(img_resized)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                
                if show_heatmap:
                    superimposed_img = superimpose_heatmap(img_cv, heatmap, alpha=heatmap_alpha)
                    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
                    st.image(superimposed_img_rgb, caption="Grad-CAM Heatmap (Red areas: High attention)", use_column_width=True)
                
                if show_localization:
                    localized_img, boxed_img = localize_pest(img_cv, heatmap, threshold=localization_threshold)
                    if localized_img is not None:
                        localized_rgb = cv2.cvtColor(localized_img, cv2.COLOR_BGR2RGB)
                        st.image(localized_rgb, caption=f"Localized {predicted_class}", use_column_width=True)
                    else:
                        st.warning("No clear pest localization found.")
                    # Show boxed image
                    boxed_rgb = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)
                    st.image(boxed_rgb, caption="Image with Bounding Box", use_column_width=True)
            except Exception as e:
                st.warning(f"Explanation features unavailable: {e}")

with col2:
    if uploaded_file is not None:
        st.subheader("Detection Results")
        st.markdown(f"<p class='prediction'>{predicted_class}</p>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        if show_probs:
            with st.expander("Detailed Probabilities"):
                probs = predictions[0]
                for i, prob in enumerate(probs):
                    st.write(f"{class_names[i]}: {prob * 100:.2f}%")
        
        # Interactive elements
        st.button("Re-process Image")  # Placeholder for re-run
        if st.button("Download Results"):
            # Example: Save summary
            summary = f"Predicted Pest: {predicted_class}\nConfidence: {confidence:.2f}%"
            st.download_button("Download Summary", summary, file_name="pest_detection.txt")
        
        # Add some info about the pest (placeholder; could fetch from web if needed)
        with st.expander("Learn More About This Pest"):
            st.write(f"Common info on {predicted_class}: (Add real info or use tools for dynamic content)")
