import streamlit as st
import tensorflow as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import altair as alt
import requests
import os
import cv2
import tempfile
import time
from datetime import datetime
import random
from io import BytesIO
# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriGuard Pro | AI Pest Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #f4f7f6;
    }
   
    /* Typography */
    h1, h2, h3 {
        color: #1b4332;
        font-family: 'Helvetica Neue', sans-serif;
    }
   
    /* Custom Cards */
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
   
    /* Success/Warning/Error Badges */
    .badge-success {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
   
    /* Prediction Score */
    .big-score {
        font-size: 48px;
        font-weight: 800;
        color: #2d6a4f;
    }
   
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
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
PEST_INFO = {
    'aphids': {
        'name': 'Aphids (Plant Lice)',
        'details': 'Small sap-sucking insects that cluster on new growth.',
        'cause': 'High nitrogen levels; lack of natural predators.',
        'control': 'Neem oil, insecticidal soaps, ladybugs.',
        'risk_level': 'Moderate'
    },
    'armyworm': {
        'name': 'Armyworm',
        'details': 'Caterpillars that move in groups stripping leaves.',
        'cause': 'Grassy weeds; cool wet springs.',
        'control': 'Bacillus thuringiensis (Bt), trichogramma wasps.',
        'risk_level': 'High'
    },
    'beetle': {
        'name': 'Beetle',
        'details': 'Hard-shelled insects causing leaf skeletonization.',
        'cause': 'Overwintering in soil/debris.',
        'control': 'Hand-picking, floating row covers, nematodes.',
        'risk_level': 'Moderate'
    },
    'bollworm': {
        'name': 'Bollworm',
        'details': 'Larvae that bore into fruit and pods.',
        'cause': 'Moth migration; monocultures.',
        'control': 'Pheromone traps, resistant crop varieties.',
        'risk_level': 'High'
    },
    'catterpillar': {
        'name': 'Caterpillar',
        'details': 'Leaf chewers causing irregular holes.',
        'cause': 'Butterflies/moths laying eggs.',
        'control': 'Bt spray, hand-picking, bird encouragement.',
        'risk_level': 'Moderate'
    },
    'earthworms': {
        'name': 'Earthworm',
        'details': 'Beneficial soil aerators.',
        'cause': 'Healthy soil ecosystem.',
        'control': 'None required. They are friends!',
        'risk_level': 'None'
    },
    'grasshopper': {
        'name': 'Grasshopper',
        'details': 'Jumping insects that consume foliage.',
        'cause': 'Dry weather; nearby uncultivated land.',
        'control': 'Nolo Bait, tilling in fall.',
        'risk_level': 'High'
    },
    'mites': {
        'name': 'Spider Mites',
        'details': 'Tiny arachnids causing yellow stippling.',
        'cause': 'Hot, dusty, dry conditions.',
        'control': 'Misting, predatory mites, horticultural oil.',
        'risk_level': 'Moderate'
    },
    'moth': {
        'name': 'Moth',
        'details': 'Adult stage of larvae pests.',
        'cause': 'Attracted to light; open access.',
        'control': 'Light traps, netting.',
        'risk_level': 'Low (Directly)'
    },
    'sawfly': {
        'name': 'Sawfly Larvae',
        'details': 'Wasp-like larvae that skeletonize leaves.',
        'cause': 'Spring emergence from soil.',
        'control': 'Insecticidal soap (NOT Bt).',
        'risk_level': 'Moderate'
    },
    'stem_borer': {
        'name': 'Stem Borer',
        'details': 'Larvae boring internally into stems.',
        'cause': 'Poor field sanitation.',
        'control': 'Destroy infected stems, crop rotation.',
        'risk_level': 'Severe'
    },
    'wasp': {
        'name': 'Wasp',
        'details': 'Flying stinging insects. Often predatory.',
        'cause': 'Nesting sites nearby.',
        'control': 'Traps only if safety hazard.',
        'risk_level': 'Low (to crops)'
    },
    'weevil': {
        'name': 'Weevil',
        'details': 'Snouted beetles damaging grains/roots.',
        'cause': 'Infested stored seeds.',
        'control': 'Sanitation, diatomaceous earth.',
        'risk_level': 'High'
    }
}
# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model (This happens only once)..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
def preprocess_image(image_data):
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array
def create_download_report(pest_name, score, details, advice):
    report_text = f"""
    AGRIGUARD DIAGNOSTIC REPORT
    ---------------------------
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
   
    DETECTED PEST: {pest_name}
    CONFIDENCE: {score:.1%}
   
    DETAILS:
    {details}
   
    RECOMMENDED ACTION:
    {advice}
   
    ---------------------------
    Generated by AgriGuard AI
    """
    return report_text
# -----------------------------------------------------------------------------
# 4. APP LAYOUT
# -----------------------------------------------------------------------------
def main():
    # Session State for History
    if 'history' not in st.session_state:
        st.session_state.history = []
    # -- Sidebar --
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
        st.title("AgriGuard Pro")
        st.caption("v2.0.1 | AI-Powered AgTech")
       
        st.markdown("---")
       
        input_mode = st.radio("Input Source", ["📸 Camera", "📂 Upload Image", "🎥 Upload Video"])
       
        st.markdown("---")
        st.subheader("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05, help="Minimum score to consider a detection valid.")
       
        st.markdown("---")
        if st.session_state.history:
            st.subheader("Recent Detections")
            for item in st.session_state.history[-5:]:
                st.text(f"• {item}")
    # -- Main Content --
    col_hero1, col_hero2 = st.columns([3, 1])
    with col_hero1:
        st.title("Plant Health Diagnostics")
        st.markdown("Upload an image or use your camera to detect pests and get immediate control recommendations.")
    with col_hero2:
        # Simulated weather widget
        st.info(f"📍 Local Conditions: 28°C, Humidity 65%")
    model = get_model()
    if not model:
        st.stop()
    # Input Logic
    processed_image = None
   
    if input_mode == "📸 Camera":
        img_file = st.camera_input("Take a clear picture of the pest")
        if img_file:
            processed_image = Image.open(img_file).convert("RGB")
           
    elif input_mode == "📂 Upload Image":
        img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if img_file:
            processed_image = Image.open(img_file).convert("RGB")
        # New feature: Fetch random 7 test images from GitHub
        if not processed_image:
            st.subheader("Or choose a test pest for testing")
            if 'random_pests' not in st.session_state:
                st.session_state.random_pests = random.sample(CLASS_NAMES, min(7, len(CLASS_NAMES)))
            if 'test_images' not in st.session_state:
                st.session_state.test_images = {}
                for pest in st.session_state.random_pests:
                    class_url = f"https://api.github.com/repos/blurerjr/Explainable-Pest-Detection/contents/test/?ref=8eabae8d2e82acbc7507a2a41e2bec28ac1ab097"
                    resp = requests.get(class_url)
                    if resp.status_code == 200:
                        files = resp.json()
                        images = [f['name'] for f in files if isinstance(f, dict) and f.get('type') == 'file' and f['name'].lower().endswith(('.jpg', '.png', '.jpeg'))]
                        if images:
                            st.session_state.test_images[pest] = random.choice(images)
            if 'selected_test_url' in st.session_state:
                resp = requests.get(st.session_state.selected_test_url)
                if resp.status_code == 200:
                    processed_image = Image.open(BytesIO(resp.content)).convert("RGB")
            cols = st.columns(7)
            for i, pest in enumerate(st.session_state.random_pests):
                with cols[i]:
                    img_name = st.session_state.test_images.get(pest)
                    if img_name:
                        raw_url = f"https://raw.githubusercontent.com/blurerjr/Explainable-Pest-Detection/8eabae8d2e82acbc7507a2a41e2bec28ac1ab097/test/{img_name}"
                        st.image(raw_url, caption=pest.capitalize(), use_column_width=True)
                        if st.button("Select", key=f"select_{pest}_{i}"):
                            st.session_state.selected_test_url = raw_url
           
    elif input_mode == "🎥 Upload Video":
        video_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            ret, frame = cap.read() # Take first frame for simplicity/speed in demo
            if ret:
                processed_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            os.remove(tfile.name)
    # -- Analysis Section --
    if processed_image:
        st.markdown("---")
       
        # Two-column layout for results
        col_img, col_data = st.columns([1, 1.5], gap="large")
       
        display_img, input_arr = preprocess_image(processed_image)
       
        with st.spinner("Analyzing biological signatures..."):
            predictions = model.predict(input_arr)
            time.sleep(0.5) # UX smoother
       
        # Process Results
        top_indices = np.argsort(predictions[0])[::-1][:3]
        top_scores = predictions[0][top_indices]
        top_classes = [CLASS_NAMES[i] for i in top_indices]
       
        main_score = top_scores[0]
        main_class = top_classes[0]
        pest_data = PEST_INFO.get(main_class, {})
        # -- Left Column (Image) --
        with col_img:
            st.image(processed_image, caption="Analyzed Specimen", use_container_width=True)
           
            # Risk Level Badge
            risk = pest_data.get('risk_level', 'Unknown')
            risk_color = "red" if risk in ['High', 'Severe'] else "orange" if risk == 'Moderate' else "green"
            st.markdown(f"**Risk Level:** :{risk_color}[{risk}]")
        # -- Right Column (Data & Insights) --
        with col_data:
            if main_score < confidence_threshold:
                st.warning(f"⚠️ Low Confidence Detection ({main_score:.1%}). The model is unsure. Please try a clearer image.")
            else:
                # Add to history
                if main_class not in st.session_state.history:
                    st.session_state.history.append(main_class)
               
                # Header
                st.markdown(f"### Detected: **{pest_data.get('name', main_class).upper()}**")
               
                # Probability Chart (Altair)
                df_probs = pd.DataFrame({
                    'Pest': top_classes,
                    'Probability': top_scores
                })
               
                chart = alt.Chart(df_probs).mark_bar().encode(
                    x=alt.X('Probability', axis=alt.Axis(format='%')),
                    y=alt.Y('Pest', sort='-x'),
                    color=alt.condition(
                        alt.datum.Pest == main_class,
                        alt.value('#2e7d32'), # Highlight winner
                        alt.value('#a5d6a7') # Others
                    ),
                    tooltip=['Pest', alt.Tooltip('Probability', format='.1%')]
                ).properties(height=150)
               
                st.altair_chart(chart, use_container_width=True)
                # Tabs for Details
                tab1, tab2, tab3 = st.tabs(["📋 Description", "🛡️ Treatment", "🩺 Causes"])
               
                with tab1:
                    st.write(pest_data.get('details'))
               
                with tab2:
                    st.success(f"**Action Plan:** {pest_data.get('control')}")
               
                with tab3:
                    st.info(f"**Root Cause:** {pest_data.get('cause')}")
                # Download Report Button
                report = create_download_report(
                    pest_data.get('name', main_class),
                    main_score,
                    pest_data.get('details'),
                    pest_data.get('control')
                )
               
                st.download_button(
                    label="📄 Download Diagnostic Report",
                    data=report,
                    file_name=f"AgriGuard_Report_{main_class}.txt",
                    mime="text/plain"
                )
if __name__ == "__main__":
    main()
