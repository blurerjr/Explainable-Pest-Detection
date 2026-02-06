import streamlit as st
import tensorflow as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import requests
import os
import cv2
import tempfile

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriGuard | AI Pest Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, clean look
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2E7D32;
        font-weight: 700;
    }
    
    /* Upload Area Styling */
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
    }
    
    /* Results Container */
    .result-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2E7D32;
    }
    
    /* Metrics */
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    .confidence-score {
        font-size: 24px;
        font-weight: bold;
        color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & DATA
# -----------------------------------------------------------------------------
MODEL_URL = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
MODEL_PATH = "best_pest_model.keras"
CLASS_NAMES = ['aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar', 
               'earthworms', 'grasshopper', 'mites', 'moth', 'sawfly', 
               'stem_borer', 'wasp', 'weevil']

# Detailed Pest Information Database
PEST_INFO = {
    'aphids': {
        'name': 'Aphids (Plant Lice)',
        'details': 'Small sap-sucking insects that cluster on new growth and undersides of leaves. They excrete sticky "honeydew" which attracts ants and promotes sooty mold.',
        'cause': 'Warm temperatures and high nitrogen levels in plants (often from over-fertilizing).',
        'control': 'Spray with strong stream of water; Apply Neem oil or insecticidal soaps; Introduce predators like ladybugs.'
    },
    'armyworm': {
        'name': 'Armyworm',
        'details': 'Caterpillars that move in large groups "like an army", stripping leaves and destroying crops rapidly.',
        'cause': 'Moths laying eggs in grassy or weedy areas; Cool, wet springs followed by warm weather.',
        'control': 'Apply Bacillus thuringiensis (Bt); Use trichogramma wasps; Remove weeds; Chemical insecticides if infestation is severe.'
    },
    'beetle': {
        'name': 'Beetle (General)',
        'details': 'Hard-shelled insects with chewing mouthparts. Includes species like Japanese beetles or Flea beetles that skeletonize leaves.',
        'cause': 'Overwintering in soil or debris; Attraction to specific plant pheromones.',
        'control': 'Hand-picking for larger beetles; Neem oil; Floating row covers; Nematodes for soil-dwelling larvae.'
    },
    'bollworm': {
        'name': 'Bollworm',
        'details': 'Larvae that bore into fruit, pods, or stems (common in cotton, corn, tomatoes).',
        'cause': 'Migration of moths from warmer areas; monoculture cropping.',
        'control': 'Pheromone traps; Planting resistant varieties; Biological control (Trichogramma); Deep plowing to kill pupae.'
    },
    'catterpillar': {
        'name': 'Caterpillar (General)',
        'details': 'Larval stage of moths/butterflies. voracious feeders that chew irregular holes in leaves.',
        'cause': 'Butterflies/moths laying eggs on host plants.',
        'control': 'Hand-picking; Bacillus thuringiensis (Bt) spray; Diatomaceous earth; Encouraging birds.'
    },
    'earthworms': {
        'name': 'Earthworm',
        'details': 'Segmented worms living in soil. Generally beneficial for soil aeration, but can be a nuisance in turfgrass (casting).',
        'cause': 'Healthy, organic-rich soil with moisture.',
        'control': 'Usually control is not recommended as they are beneficial. For turf aesthetics: Dethatching and reducing irrigation.'
    },
    'grasshopper': {
        'name': 'Grasshopper',
        'details': 'Jumping insects with chewing mouthparts. Can consume large amounts of foliage.',
        'cause': 'Dry, warm weather; nearby uncultivated land breeding grounds.',
        'control': 'Nolo Bait (biological control); Floating row covers; Tilling soil in fall to destroy eggs.'
    },
    'mites': {
        'name': 'Mites (e.g., Spider Mites)',
        'details': 'Tiny arachnids causing stippling (yellow dots) on leaves and fine silk webbing.',
        'cause': 'Hot, dusty, and dry conditions.',
        'control': 'Increase humidity/misting; Predatory mites; Horticultural oils or miticides.'
    },
    'moth': {
        'name': 'Moth',
        'details': 'Adult stage of many pests. While adults often feed on nectar, they lay eggs that turn into destructive larvae.',
        'cause': 'Attracted to light at night; Open access to host plants.',
        'control': 'Light traps; Pheromone traps; Netting to prevent egg-laying.'
    },
    'sawfly': {
        'name': 'Sawfly Larvae',
        'details': 'Wasp-like insect larvae that look like caterpillars but have more prolegs. They strip leaves (skeletonizers).',
        'cause': 'Adults emerging from soil in spring.',
        'control': 'Insecticidal soap (Bt is NOT effective on sawflies); Hort oil; Hand-picking.'
    },
    'stem_borer': {
        'name': 'Stem Borer',
        'details': 'Larvae that bore into the stems of plants, causing wilting and death of the plant part above the hole.',
        'cause': 'Eggs laid on stems; poor field sanitation.',
        'control': 'Remove and destroy infected stems immediately; Crop rotation; Systemic insecticides.'
    },
    'wasp': {
        'name': 'Wasp',
        'details': 'Flying insects with stingers. Many are predatory and beneficial, but some damage fruit or are safety hazards.',
        'cause': 'Nesting sites (eaves, trees) and food sources (sugar/protein).',
        'control': 'Traps; Aerosol sprays for nests (if dangerous); removing ripe fruit.'
    },
    'weevil': {
        'name': 'Weevil',
        'details': 'Beetles with distinct snouts. They damage stored grains, cotton, and roots.',
        'cause': 'Infested seeds/grain; debris left in fields.',
        'control': 'Sanitation (clean storage); Crop rotation; Diatomaceous earth for stored grains.'
    }
}

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_resource
def get_model():
    """Downloads (if needed) and loads the Keras model."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights for the first time..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None

    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_data):
    """Resizes and normalizes image for the model."""
    # Resize to 224x224 as required
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    image_array = img_to_array(image)
    # Normalize (Assuming standard 0-1 scaling for custom Keras models unless specified otherwise)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array

def process_video_frame(video_file):
    """Extracts the middle frame from a video file."""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get a frame from the middle of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    
    cap.release()
    os.remove(tfile.name) # Clean up
    
    if ret:
        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

# -----------------------------------------------------------------------------
# 4. MAIN APP LAYOUT
# -----------------------------------------------------------------------------

def main():
    # Header
    st.markdown("# 🌿 AgriGuard <span style='font-size: 20px; color: #666;'>| Intelligent Pest Detection</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Upload Settings")
        source_type = st.radio("Select Input Type", ["Image", "Video/Camera"])
        
        st.markdown("### About")
        st.info(
            "This AI tool detects **13 types of agricultural pests** "
            "and provides actionable control advice to help protect your crops."
        )
        st.markdown("Made with Streamlit & TensorFlow")

    # Model Loading
    model = get_model()
    
    if model is None:
        st.warning("Model could not be loaded. Please check the URL or internet connection.")
        return

    # File Uploader logic
    uploaded_file = None
    processed_image = None
    
    if source_type == "Image":
        uploaded_file = st.file_uploader("Upload a leaf or pest image...", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            processed_image = Image.open(uploaded_file).convert("RGB")
            
    elif source_type == "Video/Camera":
        uploaded_file = st.file_uploader("Upload a short video...", type=["mp4", "mov", "avi"])
        if uploaded_file:
            with st.spinner("Extracting frame from video..."):
                processed_image = process_video_frame(uploaded_file)
                if processed_image is None:
                    st.error("Could not process video file.")

    # Detection & Results
    if processed_image:
        # Layout: Left for Image, Right for Info
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.subheader("🔍 Analyzed Image")
            # Display rounded image
            st.image(processed_image, use_container_width=True, caption="Uploaded Sample")
            
        with col2:
            st.subheader("📊 Analysis Results")
            
            with st.spinner("Identifying pest..."):
                # Preprocess and Predict
                display_img, input_arr = preprocess_image(processed_image)
                predictions = model.predict(input_arr)
                
                # Get top prediction
                score = np.max(predictions)
                class_idx = np.argmax(predictions)
                detected_class = CLASS_NAMES[class_idx]
                pest_data = PEST_INFO.get(detected_class, {})

                # Dynamic Color based on confidence
                conf_color = "#2E7D32" if score > 0.8 else "#F9A825" if score > 0.5 else "#C62828"

                # Display Result Box
                st.markdown(f"""
                <div class="result-box">
                    <h2 style="margin-top:0; color: {conf_color};">{pest_data.get('name', detected_class.title())}</h2>
                    <div class="metric-container">
                        <span>Confidence Score:</span>
                        <span class="confidence-score" style="color: {conf_color};">{score:.2%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable Details
                st.markdown("### 📋 Pest Details")
                st.write(pest_data.get('details', 'No details available.'))
                
                st.markdown("### 🍂 Probable Cause")
                st.info(pest_data.get('cause', 'Unknown.'))
                
                st.markdown("### 🛡️ Control & Prevention")
                st.success(pest_data.get('control', 'Consult a local agronomist.'))

if __name__ == "__main__":
    main()
