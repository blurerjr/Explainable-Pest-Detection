import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageOps
import requests
from io import BytesIO
import pandas as pd

# -------------------------------
# 1. Page Configuration & Styling
# -------------------------------
st.set_page_config(
    page_title="AgriGuard | Smart Pest Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1b5e20;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 2. Model Loading
# -------------------------------
@st.cache_resource
def load_pest_model():
    # URL provided in original prompt
    url = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
    try:
        response = requests.get(url)
        response.raise_for_status()
        model_bytes = BytesIO(response.content)
        # Load model with compile=False to avoid issues with custom optimizers/losses during inference
        model = tf.keras.models.load_model(model_bytes, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Critical Error: Could not load model. Details: {e}")
        st.stop()

model = load_pest_model()

# -------------------------------
# 3. Data & Classes
# -------------------------------
CLASSES = [
    'aphids', 'armyworm', 'beetle', 'bollworm', 'caterpillar', 'earthworms',
    'grasshopper', 'mites', 'moth', 'sawfly', 'stem_borer', 'wasp', 'weevil'
]

PEST_DATA = {
    "aphids": {
        "name": "Aphids",
        "description": "Small, soft-bodied insects that suck sap from leaves and stems, causing distortion.",
        "cause": "Warm, humid weather; excess nitrogen fertilizer.",
        "control": ["Spray strong jet of water.", "Neem oil application.", "Introduce ladybugs."],
        "severity": "Moderate"
    },
    "armyworm": {
        "name": "Armyworm",
        "description": "Caterpillars that move in large groups, consuming grasses and cereals.",
        "cause": "Night-time migrations, warm/moist springs.",
        "control": ["Bacillus thuringiensis (Bt) spray.", "Field sanitation.", "Nocturnal hand-picking."],
        "severity": "High"
    },
    "beetle": {
        "name": "Leaf Beetle",
        "description": "Hard-shelled insects that skeletonize leaves.",
        "cause": "High moisture, weeds, overwintering adults.",
        "control": ["Crop rotation.", "Pyrethroid sprays.", "Encourage spiders."],
        "severity": "Moderate"
    },
    "bollworm": {
        "name": "Bollworm",
        "description": "Larvae that bore into flower buds and fruits (also known as Corn Earworm).",
        "cause": "Warm temps, nearby corn/soybean hosts.",
        "control": ["Pheromone traps.", "Spinosad spray.", "Bt resistant varieties."],
        "severity": "High"
    },
    "caterpillar": {
        "name": "Caterpillar (General)",
        "description": "Larvae of moths/butterflies with chewing mouthparts.",
        "cause": "Butterfly/moth egg laying.",
        "control": ["Hand-picking.", "Bt spray.", "Row covers."],
        "severity": "Variable"
    },
    "earthworms": {
        "name": "Earthworms",
        "description": "Beneficial soil organisms. Not usually a pest.",
        "cause": "Healthy, organic-rich soil.",
        "control": ["No action needed (Beneficial).", "Reduce irrigation if excessive."],
        "severity": "None"
    },
    "grasshopper": {
        "name": "Grasshopper",
        "description": "Chewing insects that can strip whole plants.",
        "cause": "Dry, hot conditions; unkept field borders.",
        "control": ["Nosema locustae bait.", "Trim field borders.", "Early season spray."],
        "severity": "High"
    },
    "mites": {
        "name": "Spider Mites",
        "description": "Tiny arachnids causing yellow stippling and webbing.",
        "cause": "Hot, dry, dusty conditions.",
        "control": ["Increase humidity/misting.", "Horticultural oil.", "Predatory mites."],
        "severity": "Moderate"
    },
    "moth": {
        "name": "Moth (Adult/Larvae)",
        "description": "Flying adults lay eggs; larvae cause the damage.",
        "cause": "Attraction to lights, migration.",
        "control": ["Light traps.", "Mating disruption.", "Neem oil."],
        "severity": "Variable"
    },
    "sawfly": {
        "name": "Sawfly Larvae",
        "description": "Wasp-like larvae that skeletonize leaves.",
        "cause": "Warm, humid periods.",
        "control": ["Spinosad.", "Hand-picking.", "Pruning infested parts."],
        "severity": "Low to Moderate"
    },
    "stem_borer": {
        "name": "Stem Borer",
        "description": "Larvae tunnel inside stems causing 'dead heart'.",
        "cause": "Old crop residues, cracks in plant stems.",
        "control": ["Systemic insecticides.", "Pheromone traps.", "Destroy residues."],
        "severity": "Critical"
    },
    "wasp": {
        "name": "Fruit Wasp",
        "description": "Sting fruit causing premature drop.",
        "cause": "Ripe/rotting fruit, sugary baits.",
        "control": ["Fruit bagging.", "Sugar/Vinegar traps.", "Sanitation."],
        "severity": "Moderate"
    },
    "weevil": {
        "name": "Weevil",
        "description": "Snout beetles; larvae feed on roots, adults on leaves.",
        "cause": "Crop debris, warm/moist soil.",
        "control": ["Remove residues.", "Nematode application.", "Pyrethroids."],
        "severity": "Moderate"
    },
}

# -------------------------------
# 4. Helper Functions
# -------------------------------
def preprocess_image(image):
    """Resize, normalize, and handle alpha channels."""
    # Convert to RGB to handle PNG transparency or grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# -------------------------------
# 5. Sidebar Layout
# -------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/188/188333.png", width=100)
    st.title("AgriGuard 🌿")
    st.markdown("---")
    
    st.subheader("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold (%)", 
        min_value=0, max_value=100, value=40, step=5,
        help="Only show results if the model is this confident."
    )
    
    st.markdown("---")
    st.info(
        """
        **How to use:**
        1. Upload a photo or use your camera.
        2. Wait for the analysis.
        3. Read the control recommendations.
        """
    )
    st.caption(f"Model v1.0 | {len(CLASSES)} Classes")

# -------------------------------
# 6. Main Application Logic
# -------------------------------
st.title("🐛 Intelligent Pest Advisor")
st.markdown("### Identify agricultural pests and get immediate control solutions.")

# Tabs for input method
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Camera Capture"])

image_input = None

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_input = Image.open(uploaded_file)

with tab2:
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        image_input = Image.open(camera_file)

# Processing Block
if image_input is not None:
    # Layout: Image on left, Results on right
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image_input, caption="Analyzed Image", use_column_width=True, output_format="JPEG")
    
    with col2:
        try:
            with st.spinner("🔍 Analyzing leaf patterns..."):
                # Preprocess
                processed_img = preprocess_image(image_input)
                
                # Predict
                prediction = model.predict(processed_img)
                predicted_class_idx = np.argmax(prediction)
                confidence = float(np.max(prediction)) * 100
                predicted_pest_key = CLASSES[predicted_class_idx]
                
                # Get Top 3 Predictions for Chart
                top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                top_3_probs = prediction[0][top_3_indices] * 100
                top_3_names = [CLASSES[i] for i in top_3_indices]
                
                chart_data = pd.DataFrame({
                    "Pest": top_3_names,
                    "Confidence": top_3_probs
                })

            # Logic based on Threshold
            if confidence >= confidence_threshold:
                pest_info = PEST_DATA.get(predicted_pest_key, {})
                display_name = pest_info.get('name', predicted_pest_key.capitalize())
                
                # Trigger image tag for clarity
                st.markdown(f"## Detected: **{display_name}**")
                
                # Confidence Meter
                if confidence > 85:
                    st.success(f"Confidence: {confidence:.2f}% (High)")
                elif confidence > 60:
                    st.warning(f"Confidence: {confidence:.2f}% (Moderate)")
                else:
                    st.error(f"Confidence: {confidence:.2f}% (Low)")
                
                # Diagram trigger for visual verification
                st.write(f"")

                # Probability Chart
                with st.expander("📊 View Probability Breakdown"):
                    st.bar_chart(chart_data.set_index("Pest"))

                # Pest Details
                st.markdown("---")
                st.subheader("📝 Diagnosis Report")
                
                st.markdown(f"**Severity:** {pest_info.get('severity', 'Unknown')}")
                st.markdown(f"**Symptoms:** {pest_info.get('description', 'No description available.')}")
                
                st.markdown("### 🛡️ Recommended Actions")
                if 'control' in pest_info:
                    for step in pest_info['control']:
                        st.markdown(f"✅ {step}")
                else:
                    st.write("Consult a local agronomist.")

            else:
                st.warning(f"⚠️ Pest detected as **{predicted_pest_key}**, but confidence ({confidence:.2f}%) is below your threshold of {confidence_threshold}%.")
                st.markdown("Please try uploading a clearer image or zooming in on the pest.")
                with st.expander("See what the model thought"):
                     st.bar_chart(chart_data.set_index("Pest"))

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

else:
    # Empty state placeholder
    st.info("👆 Please upload an image or take a photo to start the diagnosis.")
