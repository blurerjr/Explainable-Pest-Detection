import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Custom CSS for a beautiful, modern look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; text-align: center;}
    .stButton>button {background-color: #27ae60; color: white; border: none; padding: 10px 20px; font-size: 16px;}
    .stButton>button:hover {background-color: #219653;}
    .upload-box {border: 2px dashed #bdc3c7; border-radius: 10px; padding: 30px; text-align: center; background-color: #ecf0f1; margin: 20px 0;}
    .prediction-box {background-color: #e8f5e9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 20px 0;}
    .info-box {background-color: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 20px;}
    .confidence {font-size: 28px; font-weight: bold; color: #27ae60;}
    </style>
""", unsafe_allow_html=True)

# Model URL and local path
model_url = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
model_path = "best_pest_model.keras"

# Download model if not present
if not os.path.exists(model_path):
    with st.spinner("Downloading intelligent pest detection model..."):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("Model ready!")
        else:
            st.error("Failed to download model. Please check your connection.")
            st.stop()

# Load model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Pest classes
class_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar', 'earthworms', 'grasshopper', 'mites', 'moth', 'sawfly', 'stem_borer', 'wasp', 'weevil']

# Simple explanations & prevention tips (based on common agricultural knowledge)
pest_info = {
    'aphids': {
        'desc': "Small, soft-bodied insects that suck sap from plants, causing curled leaves, stunted growth, and honeydew (sticky residue) that attracts ants and promotes sooty mold.",
        'prevent': "Encourage natural predators (ladybugs, lacewings), avoid excess nitrogen fertilizer, use reflective mulches, spray strong water jets to dislodge them, or apply insecticidal soap/oil early."
    },
    'armyworm': {
        'desc': "Caterpillars that march in groups ('armies'), chewing large holes in leaves, often defoliating crops like maize, rice, and pastures.",
        'prevent': "Scout fields regularly, use trap crops (e.g., Napier grass), promote beneficial insects, apply Bacillus thuringiensis (Bt) for young larvae, deep ploughing to expose pupae, and avoid broad-spectrum insecticides to preserve natural enemies."
    },
    'beetle': {
        'desc': "Various beetles (e.g., leaf beetles, flea beetles) chew holes in leaves, sometimes defoliating plants or damaging roots/flowers.",
        'prevent': "Crop rotation, remove crop residues, use row covers, encourage predatory insects, and apply targeted insecticides only when thresholds are exceeded."
    },
    'bollworm': {
        'desc': "Larvae bore into bolls/fruits (especially cotton, tomatoes, okra), causing fruit drop, reduced yield, and entry points for rot fungi.",
        'prevent': "Use Bt cotton varieties if available, scout for eggs/young larvae, destroy infested plant parts, encourage natural enemies (parasitic wasps), and time planting to avoid peak moth flights."
    },
    'catterpillar': {
        'desc': "Chewing larvae of moths/butterflies that eat leaves, bore into stems/fruits, leading to defoliation and reduced photosynthesis/yield.",
        'prevent': "Hand-pick visible caterpillars, use Bt sprays, release Trichogramma wasps, practice crop rotation, and maintain field sanitation."
    },
    'earthworms': {
        'desc': "Usually beneficial (improve soil), but some species can damage seedlings by feeding on roots or dragging leaves underground.",
        'prevent': "Rarely a major pest; maintain healthy soil biology, avoid overwatering, and use barriers for seedlings if needed."
    },
    'grasshopper': {
        'desc': "Chew irregular holes in leaves and can defoliate plants in large numbers during outbreaks.",
        'prevent': "Early tillage to destroy eggs, encourage natural predators (birds, robber flies), use bait traps, and apply insecticides only during outbreaks."
    },
    'mites': {
        'desc': "Tiny arachnids that suck plant juices, causing stippling, yellowing, webbing, and leaf drop (e.g., spider mites).",
        'prevent': "Increase humidity, avoid dust, release predatory mites (e.g., Phytoseiulus), use miticidal soaps/oils, and monitor hot/dry conditions."
    },
    'moth': {
        'desc': "Adult moths lay eggs that hatch into damaging caterpillars; adults rarely cause direct damage.",
        'prevent': "Pheromone traps for monitoring, Bt for larvae, light traps, and habitat management for natural enemies."
    },
    'sawfly': {
        'desc': "Larvae resemble caterpillars and chew leaves/needles, often in groups, defoliating trees/shrubs.",
        'prevent': "Hand removal of larvae clusters, encourage birds/parasitoids, prune affected parts, and use selective insecticides."
    },
    'stem_borer': {
        'desc': "Larvae bore into stems, causing dead hearts, wilting, reduced tillering, and yield loss (common in rice, maize).",
        'prevent': "Use resistant varieties, destroy stubble, early planting, release egg parasitoids (Trichogramma), and apply granular insecticides if needed."
    },
    'wasp': {
        'desc': "Some parasitic wasps are beneficial; pest wasps may sting or rarely damage crops directly.",
        'prevent': "Usually not a crop pest; focus on beneficial species preservation."
    },
    'weevil': {
        'desc': "Beetles with snouts; larvae bore into stems, roots, seeds, or fruits (e.g., rice weevil, boll weevil).",
        'prevent': "Sanitation (remove residues), crop rotation, use resistant varieties, store grains properly, and apply targeted controls."
    }
}

# App layout
st.set_page_config(page_title="Smart Pest Detector", layout="wide", page_icon="🪲")

st.title("🪲 Smart Pest Detector – Identify & Protect Your Crops")

# Two-column layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Upload Your Crop/Pest Image")
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop image here or click to browse (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your uploaded image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            # Preprocess
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_class = class_names[predicted_index]
            confidence = np.max(predictions) * 100

with col2:
    if uploaded_file is not None:
        st.subheader("Detection Result")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: #27ae60;'>{predicted_class.capitalize()}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='confidence'>{confidence:.1f}% Confidence</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Explanation section
        if predicted_class in pest_info:
            info = pest_info[predicted_class]
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader("About This Pest")
            st.write(info['desc'])
            st.subheader("How to Prevent & Control It")
            st.write(info['prevent'])
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Basic info coming soon for this pest!")

        # Optional details
        if st.checkbox("View probability breakdown"):
            with st.expander("All Class Probabilities"):
                probs = predictions[0]
                for i, prob in enumerate(probs):
                    st.write(f"{class_names[i].capitalize()}: {prob * 100:.1f}%")

# Footer / call to action
st.markdown("---")
st.caption("Powered by AI • Built for farmers & gardeners • Early detection saves crops!")
