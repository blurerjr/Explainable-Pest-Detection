import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

# Page configuration
st.set_page_config(
    page_title="Pest Detection System",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .pest-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Define pest classes
PEST_CLASSES = ['aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar', 
                'earthworms', 'grasshopper', 'mites', 'moth', 'sawfly', 
                'stem_borer', 'wasp', 'weevil']

IMAGE_SIZE = (224, 224)
MODEL_URL = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"

@st.cache_resource
def load_model(model_url):
    """Load the pre-trained model"""
    try:
        # Download model from URL
        response = requests.get(model_url, timeout=30)
        response.raise_for_status()
        
        # Save temporarily
        temp_model_path = "temp_model.keras"
        with open(temp_model_path, 'wb') as f:
            f.write(response.content)
        
        # Load model
        model = tf.keras.models.load_model(temp_model_path)
        
        # Clean up temp file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize(IMAGE_SIZE)
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_pest(model, image):
    """Make prediction on the image"""
    if model is None:
        return None, None
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = PEST_CLASSES[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]

def display_results(predicted_class, confidence, all_predictions):
    """Display prediction results"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Detected Pest", predicted_class.upper(), 
                 delta=f"{confidence*100:.2f}% confidence")
    
    with col2:
        st.metric("Confidence Score", f"{confidence*100:.2f}%")
    
    # Display all predictions as a bar chart
    st.subheader("All Class Probabilities")
    
    # Create dataframe for visualization
    predictions_df = {
        'Pest Class': PEST_CLASSES,
        'Probability': all_predictions
    }
    
    st.bar_chart(
        data={
            'Pest Class': PEST_CLASSES,
            'Probability': all_predictions
        }
    )
    
    # Show top 5 predictions
    st.subheader("Top 5 Predictions")
    top_5_idx = np.argsort(all_predictions)[-5:][::-1]
    
    for rank, idx in enumerate(top_5_idx, 1):
        prob = all_predictions[idx]
        st.write(f"{rank}. **{PEST_CLASSES[idx].replace('_', ' ').title()}**: {prob*100:.2f}%")

# Main app
def main():
    st.title("🐛 Pest Detection and Classification System")
    st.markdown("---")
    
    st.write("""
    This application uses a deep learning model to detect and classify agricultural pests 
    from images. Simply upload an image or take a photo of a pest, and the system will 
    identify what type of pest it is.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(MODEL_URL)
    
    if model is None:
        st.error("Failed to load the model. Please check your internet connection and try again.")
        return
    
    st.success("✅ Model loaded successfully!")
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio("Choose input method:", 
                           ["📤 Upload Image", "📷 Capture from Camera"],
                           horizontal=True)
    
    image = None
    
    if input_method == "📤 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image of a pest",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    
    else:  # Camera input
        camera_image = st.camera_input("Take a photo of the pest")
        if camera_image is not None:
            image = Image.open(camera_image).convert('RGB')
    
    # Process image
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, all_predictions = predict_pest(model, image)
            
            if predicted_class is not None:
                # Display results
                st.success("Analysis complete!")
                st.markdown("---")
                display_results(predicted_class, confidence, all_predictions)
                
                # Additional information
                st.markdown("---")
                st.info(f"""
                **Detected Pest:** {predicted_class.replace('_', ' ').title()}
                
                **Confidence:** {confidence*100:.2f}%
                
                Consider consulting with an agricultural expert for treatment recommendations.
                """)
            else:
                st.error("Failed to make prediction. Please try again.")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About This App")
        st.markdown("""
        ### Supported Pest Classes
        """)
        
        # Display pest classes in columns
        cols = st.columns(2)
        for idx, pest in enumerate(PEST_CLASSES):
            with cols[idx % 2]:
                st.write(f"• {pest.replace('_', ' ').title()}")
        
        st.markdown("---")
        st.markdown("""
        ### How to Use
        1. Choose to upload an image or take a photo
        2. Select a clear image of the pest
        3. Wait for the model to analyze
        4. View the results and confidence score
        
        ### Tips for Best Results
        - Use clear, well-lit images
        - Ensure the pest is the main focus
        
