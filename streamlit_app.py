import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# App title and description
st.set_page_config(
    page_title="Pest Detection System",
    page_icon="🐛",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .result-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .pest-info {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .control-methods {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        height: 20px;
        margin: 5px 0;
    }
    .confidence-fill {
        background-color: #4caf50;
        border-radius: 5px;
        height: 100%;
        transition: width 0.5s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Pest information dictionary
pest_info = {
    'aphids': {
        'description': 'Small sap-sucking insects that can be green, black, or brown. They reproduce quickly and can cause stunted growth and curled leaves.',
        'control': [
            'Introduce natural predators like ladybugs or lacewings',
            'Use insecticidal soap or neem oil',
            'Remove heavily infested plants',
            'Use reflective mulches to deter aphids'
        ]
    },
    'armyworm': {
        'description': 'Caterpillars that feed on leaves and can cause significant defoliation. They are typically green or brown with stripes.',
        'control': [
            'Apply biological pesticides containing Bacillus thuringiensis (Bt)',
            'Use pheromone traps to monitor and reduce populations',
            'Encourage natural predators like birds and parasitic wasps',
            'Rotate crops to disrupt their life cycle'
        ]
    },
    'beetle': {
        'description': 'Hard-shelled insects with chewing mouthparts. Many species feed on leaves, stems, or roots.',
        'control': [
            'Handpick and remove beetles from plants',
            'Use floating row covers to protect plants',
            'Apply diatomaceous earth around plants',
            'Introduce predatory insects like ground beetles'
        ]
    },
    'bollworm': {
        'description': 'Larvae of moths that bore into buds and fruits, causing significant damage to crops like cotton and tomatoes.',
        'control': [
            'Use Bt (Bacillus thuringiensis) based insecticides',
            'Implement crop rotation',
            'Monitor with pheromone traps',
            'Encourage natural predators like parasitic wasps'
        ]
    },
    'catterpillar': {
        'description': 'Larval stage of butterflies and moths that feed on leaves, often causing skeletonization of foliage.',
        'control': [
            'Handpick and remove caterpillars',
            'Use Bt (Bacillus thuringiensis) sprays',
            'Encourage natural predators like birds',
            'Apply neem oil to deter feeding'
        ]
    },
    'earthworms': {
        'description': 'Beneficial soil organisms that improve soil structure and nutrient cycling. Typically not considered pests.',
        'control': [
            'No control needed - earthworms are beneficial',
            'Maintain organic matter in soil to support earthworm populations',
            'Avoid excessive tilling which can harm earthworms'
        ]
    },
    'grasshopper': {
        'description': 'Jumping insects that feed on leaves, often causing significant defoliation in large numbers.',
        'control': [
            'Use floating row covers to protect plants',
            'Apply diatomaceous earth to plant surfaces',
            'Encourage natural predators like birds and spiders',
            'Use baits containing Nosema locustae (a natural pathogen)'
        ]
    },
    'mites': {
        'description': 'Tiny arachnids that suck plant juices, causing stippling, discoloration, and webbing on leaves.',
        'control': [
            'Spray plants with water to dislodge mites',
            'Use insecticidal soap or neem oil',
            'Introduce predatory mites',
            'Maintain proper humidity levels'
        ]
    },
    'moth': {
        'description': 'Adult stage of many pest caterpillars. While adults don't typically damage plants, their larvae can be destructive.',
        'control': [
            'Use pheromone traps to disrupt mating',
            'Apply Bt (Bacillus thuringiensis) to target larvae',
            'Encourage natural predators like bats and birds',
            'Use light traps to monitor populations'
        ]
    },
    'sawfly': {
        'description': 'Larvae resemble caterpillars but are actually wasp relatives. They feed on leaves, often skeletonizing them.',
        'control': [
            'Handpick and remove larvae',
            'Use insecticidal soap or horticultural oil',
            'Encourage natural predators like birds and parasitic wasps',
            'Apply spinosad-based insecticides'
        ]
    },
    'stem_borer': {
        'description': 'Larvae that bore into stems, causing wilting and plant death. Common in crops like maize and sugarcane.',
        'control': [
            'Use stem injections with insecticides',
            'Practice crop rotation',
            'Remove and destroy infested plant material',
            'Use resistant crop varieties'
        ]
    },
    'wasp': {
        'description': 'Some wasp species can be beneficial predators, while others may damage fruits or be aggressive.',
        'control': [
            'Use traps with protein baits in spring',
            'Remove nests when found',
            'Encourage natural predators like birds',
            'Use insecticidal dusts in nest entrances'
        ]
    },
    'weevil': {
        'description': 'Beetles with elongated snouts that feed on plants, often causing distinctive notching on leaf edges.',
        'control': [
            'Use beneficial nematodes for soil-dwelling larvae',
            'Apply diatomaceous earth',
            'Use pheromone traps',
            'Practice crop rotation'
        ]
    }
}

# Load the model
@st.cache_resource
def load_model():
    model_url = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
    response = requests.get(model_url)
    model = tf.keras.models.load_model(BytesIO(response.content))
    return model

model = load_model()

# Pest classes
classes = ['aphids', 'armyworm', 'beetle', 'bollworm',
           'catterpillar', 'earthworms', 'grasshopper', 'mites',
           'moth', 'sawfly', 'stem_borer', 'wasp', 'weevil']

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Main app
def main():
    st.title("🐛 Pest Detection System")
    st.markdown("""
    Upload an image of a plant pest, and our AI will identify it and provide information about the pest and control methods.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess and predict
            img_array = preprocess_image(img)
            predictions = model.predict(img_array)
            predicted_class = classes[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100

    with col2:
        if uploaded_file is not None:
            st.subheader("Detection Results")

            # Display prediction
            st.write(f"### Detected Pest: **{predicted_class.replace('_', ' ').title()}**")
            st.write(f"### Confidence: **{confidence:.2f}%**")

            # Confidence bar
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%"></div>
            </div>
            """, unsafe_allow_html=True)

            # Pest information
            if predicted_class in pest_info:
                st.markdown(f"""
                <div class="pest-info">
                    <h3>About {predicted_class.replace('_', ' ').title()}</h3>
                    <p>{pest_info[predicted_class]['description']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Control methods
                st.markdown(f"""
                <div class="control-methods">
                    <h3>Control Methods</h3>
                    <ul>
                        {"".join([f"<li>{method}</li>" for method in pest_info[predicted_class]['control']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Information about this pest is not available in our database.")

if __name__ == "__main__":
    main()
