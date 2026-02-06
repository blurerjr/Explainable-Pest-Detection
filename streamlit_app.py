import os
import numpy as np
import streamlit as st
import tensorflow as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import tempfile
from PIL import Image, ImageOps
import json
import shutil
from datetime import datetime


# ============= CONFIGURATION =============
class Config:
    """Configuration settings."""
    MODEL_URL = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
    MODEL_PATH = "best_pest_model.keras"
    PEST_DATA_DIR = "/path/to/kaggle/pest-dataset/pest/test"  # Update with actual path
    CLASS_NAMES = [
        'aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar',
        'earthworms', 'grasshopper', 'mites', 'moth', 'sawfly',
        'stem_borer', 'wasp', 'weevil'
    ]
    COLOR_MAP = {
        'healthy': '#2E7D32',  # Green
        'minor': '#F9A825',      # Orange
        'severe': '#C62828'   # Red
    }


# ============= MODEL LOADING =============
def download_if_needed(model_path, model_url):
    """Downloads model if file doesn't exist."""
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            try:
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Model download failed: {e}")
                return None
    return model_path


def load_model_with_validation(model_path, test_images=None):
    """Loads model and validates with test images."""
    model = load_model(model_path)
    
    # Prepare validation setup
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # If test images provided, validate immediately
    if test_images:
        images, labels = prepare_validation_set(test_images)
        model.evaluate(images, labels, verbose=0)
    
    return model


# ============= DATA HANDLING =============
class PestDataset:
    """Manages pest image dataset with indexing."""
    
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.pest_images = []
        self.pest_classes = []
        self._load_directory()
    
    def _load_directory(self):
        """Scans directory for pest subfolders."""
        for pest_name in os.listdir(self.dir_path):
            pest_path = os.path.join(self.dir_path, pest_name)
            if os.path.isdir(pest_path):
                self._collect_pest(pest_name, pest_path)
        
        if not self.pest_images:
            st.warning("No pest images found. Verify directory structure and permissions.")
    
    def _collect_pest(self, pest_name, directory_path):
        """Collects images for a single pest type."""
        self.pest_images.extend([
            (pest_name, os.path.relpath(img_path), img_path)
            for img_path in os.listdir(directory_path)
            if os.path.isfile(img_path)
        ])
        # Add class name
        self.pest_classes.append(pest_name)
    
    @property
    def total_images(self):
        """Total number of images in the dataset."""
        return len(self.pest_images)
    
    def get_pest_types(self):
        """Returns available pest types with image counts."""
        return {
            pest_name: sum(1 for _, _, path in self.pest_images 
                          if os.path.isfile(path) if pest_name == os.path.relpath(path, directory_path)
            ) for pest_name in self.pest_classes
        }


# ============= IMAGE PREPROCESSING =============
def preprocess_for_inference(image_PIL):
    """Preprocess image for final inference."""
    # Resize to model input size
    image = ImageOps.fit(
        image_PIL, 
        (224, 224), 
        Image.Resampling.LANCZOS,
        fill='luminance'
    )
    return img_to_array(image) / 255.0


# =============
# 2. MAIN APPLICATION
# =============

def main(app_state):
    """Main application interface."""
    
    # ============= HEADER =============
    st.markdown("# 🌿 AgriGuard | AI Pest Detection", 
               unsafe_allow_html=True)
    st.markdown("---")
    
    # ============= TABS =============
    tab1, tab2 = st.tabs(["Real-Time Detection", "Manual Testing"])
    with tab1:
        real_time_mode()
    with tab2:
        manual_testing_mode(app_state)


def real_time_mode():
    """Real-time detection interface."""
    # (Existing implementation remains mostly the same)
    # File upload logic
    # Detection and results display
    
    # ============= RESULTS ANALYSIS =============
    if detected_class and score:
        # (Existing result display code)
        
        # New: Comparison mode toggle
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div style="background:#f9f9f9; padding:15px; border-radius:10px">
                <h4>Before Classification</h4>
                <p>Original image for analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div style="background:#f9f1d9; padding:15px; border-radius:10px">
                <h4>After Classification</h4>
                <p>Visualizing pest distribution</p>
            </div>
            """, unsafe_allow_html=True)


def manual_testing_mode(app_state):
    """Manual testing interface."""
    # ============= SETUP =============
    st.header(" MANUAL PEST TESTING ")
    st.info(
        "Select a pest type to view its representative images. "
        "Then click 'Test All' to run inference on all images of that pest."
    )
    
    # ============= DATASET CONFIG =============
    with st.sidebar.expandable_collapsed(
        "Configure Testing Parameters", 
        expanded=False
    ):
        # (Parameter configuration remains)
    
    # ============= IMAGE SELECTION =============
    pest_dataset = PestDataset(Config.PEST_DATA_DIR)
    
    if not pest_dataset.total_images:
        st.warning("No images found. Please verify the dataset path and permissions.")
        return
    
    # Display pest options
    with st.sidebar:
        st.subheader("Pest Selection")
        pest_options = pest_dataset.get_pest_types()
        
        selected_pest = st.selectbox("Select Pest Type", list(pest_options.keys()))
        pest_folder = pest_dataset.pest_classes[pest_options.index(selected_pest)]
        
        st.markdown(f"— Images found: {len(pest_dataset.pest_images)}")
    
    # ============= TEST EXECUTION =============
    if st.button("Test All", key="test_button"):
        with st.spinner(f"Testing {selected_pest} images..."):
            # Prepare images for testing
            test_images = []
            test_labels = []
            
            for pest_name, relative_path, img_path in pest_dataset.pest_images:
                if pest_name == pest_folder:
                    image_PIL = Image.open(img_path)
                    test_images.append(preprocess_for_inference(image_PIL))
                    test_labels.append(0)  # Assuming all same class
            
            # Run inference
            if test_images:
                predictions = model.predict(np.array(test_images))
                # Display results per image
                results = []
                for i, (img_path, pred) in enumerate(zip(test_images, predictions)):
                    confidence = pred[np.argmax(pred)]
                    class_idx = np.argmax(pred)
                    results.append({
                        'image': img_path,
                        'class': CLASS_NAMES[class_idx],
                        'confidence': f"{confidence:.2%}",
                        'prediction': np.argmax(predictions[i])
                    })
            
            # ============= RESULTS DISPLAY =============
            st.success(f"Testing complete! Results: {len(results)} images tested.")
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="background:#f0f8ff; padding:10px; border-radius:5px">
                    <b>Average Confidence:</b> {results[0]['confidence']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background:#f0f8ff; padding:10px; border-radius:5px">
                    <b>Target Class:</b> {selected_pest.title()}
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background:#fff0f0; padding:10px; border-radius:5px">
                    <b>Images Processed:</b> {len(results)}
                </div>
                """, unsafe_allow_html=True)
            
            # ============= IMAGE GRID =============
            num_cols = min(5, len(results))
            st.markdown(f"### Results ({len(results)} images)")
            for i in range(0, len(results), num_cols):
                with st.expander(f"Image {i+1} - {results[i]['class']}")
                    col_grid = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i + j
                        if idx < len(results):
                            with col_grid[j]:
                                st.image(
                                    results[idx]['image'],
                                    use_container_width=True,
                                    caption=f"{results[idx]['class']} ({results[idx]['confidence']})"
                                )

# =============
# 3. UTILITY FUNCTIONS
# =============

def prepare_validation_set(images_paths):
    """Prepares images for validation set."""
    images, labels = [], []
    for path in images_paths:
        img = Image.open(path).convert('RGB')
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        # Labels need to be inferred from path naming
        label = os.path.basename(path).split('.')[0]
        labels.append(label)
    return np.array(images), np.array(labels)


# =============
# 4. APPLICATION ENTRY POINT
# =============

if __name__ == "__main__":
    # Initialize app state
    app_state = st.session_state.get('pest_app_state', {})
    
    model_path = Config.MODEL_PATH
    # Try to download if doesn't exist
    if not os.path.exists(model_path):
        model_path = download_if_needed(Config.MODEL_URL, model_path)
    
    if model_path is None:
        st.stop()
    
    # Attempt to load and validate model
    model = load_model_with_validation(model_path, None)
    if model is None:
        st.stop()
    
    st.session_state['model'] = model
    
    main(app_state)
