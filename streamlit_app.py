import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import cv2
import matplotlib.pyplot as plt

# Model URL and local path
model_url = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
model_path = "best_pest_model.keras"

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    st.info("Downloading the model weights...")
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")
    else:
        st.error(f"Failed to download model. Status code: {response.status_code}")
        st.stop()  # Stop execution if download fails

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
    # Create a model that maps the input image to the activations of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector of mean intensity over the channel axis
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by its importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to superimpose heatmap on image
def superimpose_heatmap(img, heatmap, alpha=0.4):
    # Resize the heatmap to match the image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# App title
st.title("Explainable Pest Detection and Classification App")

# Image upload
uploaded_file = st.file_uploader("Upload an image of a pest or affected crop", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for prediction
    st.info("Processing the image...")
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100
    
    # Display result
    st.subheader("Prediction Result")
    st.write(f"**Predicted Pest:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    # Optional: Display all probabilities
    if st.checkbox("Show detailed probabilities"):
        probs = predictions[0]
        for i, prob in enumerate(probs):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")
    
    # Explainability with Grad-CAM
    st.subheader("Model Explanation (Grad-CAM Heatmap)")
    try:
        last_conv_layer_name = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_index)
        
        # Prepare original image for OpenCV (uint8, BGR)
        img_cv = np.array(img_resized)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Superimpose
        superimposed_img = superimpose_heatmap(img_cv, heatmap)
        
        # Convert back to RGB for display
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        # Display the heatmap overlaid image
        st.image(superimposed_img, caption="Grad-CAM Heatmap (Red areas indicate regions of interest for the prediction)", use_column_width=True)
        
        # Optional: Save and download
        if st.button("Download Heatmap Image"):
            pil_img = Image.fromarray(superimposed_img)
            pil_img.save("heatmap.jpg")
            with open("heatmap.jpg", "rb") as f:
                st.download_button("Download", f.read(), file_name="pest_heatmap.jpg")
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}. The model may not have compatible convolutional layers.")
