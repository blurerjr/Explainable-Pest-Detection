import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# -------------------------------
# Model Loading (use caching for efficiency)
# -------------------------------
@st.cache_resource
def load_pest_model():
    url = "https://github.com/blurerjr/Explainable-Pest-Detection/releases/download/model/best_pest_model.keras"
    try:
        response = requests.get(url)
        response.raise_for_status()
        model_bytes = BytesIO(response.content)
        model = tf.keras.models.load_model(model_bytes)
        return model
    except Exception as e:
        st.error(f"Error loading model from URL: {e}")
        st.stop()

model = load_pest_model()

# -------------------------------
# Pest Classes (in the exact order your model expects)
# -------------------------------
classes = [
    'aphids', 'armyworm', 'beetle', 'bollworm', 'catterpillar', 'earthworms',
    'grasshopper', 'mites', 'moth', 'sawfly', 'stem_borer', 'wasp', 'weevil'
]

# -------------------------------
# Pest Information Dictionary (sourced from agricultural extension resources)
# -------------------------------
pest_info = {
    'aphids': {
        'details': 'Small, soft-bodied insects (plant lice) that cluster on new growth, stems, and under leaves. They have piercing-sucking mouthparts.',
        'causes': 'Often introduced via infested transplants, weeds, or wind. Thrive in warm, dry conditions; high nitrogen fertilizers increase susceptibility.',
        'control': 'Encourage natural enemies (ladybugs, lacewings). Use strong water sprays to dislodge them. Apply insecticidal soaps, neem oil, or horticultural oils. For severe cases, use targeted insecticides like flonicamid or acetamiprid. Avoid broad-spectrum insecticides to protect beneficial insects.'
    },
    'armyworm': {
        'details': 'Larvae (caterpillars) of certain moths that feed aggressively in groups, causing defoliation. They chew on leaves, creating ragged edges.',
        'causes': 'Moths migrate and lay eggs on crops like maize, rice, sorghum, vegetables. Warm, moist conditions favor outbreaks.',
        'control': 'Scout regularly. Use Bacillus thuringiensis (Bt) for young larvae. Apply insecticides like chlorantraniliprole or lambda-cyhalothrin when thresholds are met. Promote early planting to avoid peak infestation periods.'
    },
    'beetle': {
        'details': 'Various species (e.g., leaf beetles, flea beetles) that chew holes in leaves or bore into stems/fruit.',
        'causes': 'Overwinter in soil or crop residue; emerge in spring. Weeds serve as hosts.',
        'control': 'Crop rotation, remove crop residue. Use row covers. Apply insecticides like carbaryl or pyrethroids when damage appears. Encourage predatory insects.'
    },
    'bollworm': {
        'details': 'Larvae (e.g., cotton bollworm / corn earworm) that bore into fruit, bolls, ears, causing direct damage and secondary infections.',
        'causes': 'Moths lay eggs on flowering crops. Warm weather accelerates development.',
        'control': 'Use Bt varieties where available. Scout for eggs/larvae. Apply targeted insecticides (e.g., indoxacarb, spinosad). Release Trichogramma wasps for biological control.'
    },
    'catterpillar': {
        'details': 'General term for moth/butterfly larvae that chew leaves, bore into stems, or roll leaves with silk.',
        'causes': 'Eggs laid on host plants; multiple generations per season in warm climates.',
        'control': 'Hand-pick when possible. Use Bt sprays. Apply targeted insecticides. Encourage birds and parasitic wasps.'
    },
    'earthworms': {
        'details': 'Usually beneficial soil engineers, but some species can damage seedlings or lawns in high numbers.',
        'causes': 'Excessive moisture, organic matter, or over-irrigation.',
        'control': 'Typically not needed. Reduce overwatering. Use carbaryl baits only if severe seedling damage occurs (rare).'
    },
    'grasshopper': {
        'details': 'Chewing insects that defoliate leaves, clip flowers/pods. Large populations cause major losses.',
        'causes': 'Dry conditions favor outbreaks. Weeds/grasses serve as breeding sites.',
        'control': 'Early season tillage to destroy eggs. Apply insecticides (e.g., diflubenzuron) around field borders. Use grasshopper baits. Encourage natural predators (birds, robber flies).'
    },
    'mites': {
        'details': 'Tiny arachnids (e.g., spider mites) that suck plant sap, causing stippling, yellowing, webbing.',
        'causes': 'Hot, dry weather; pesticide overuse killing predators.',
        'control': 'Increase humidity. Use miticides like abamectin or bifenazate. Apply insecticidal soap/oil. Release predatory mites (Phytoseiulus persimilis).'
    },
    'moth': {
        'details': 'Adult stage often not damaging; larvae (caterpillars) cause harm (see armyworm, bollworm, etc.).',
        'causes': 'Attracted to lights, flowering plants.',
        'control': 'Pheromone traps for monitoring. Focus on larval control (Bt, targeted insecticides).'
    },
    'sawfly': {
        'details': 'Larvae resemble caterpillars but are fly relatives; they chew leaves or bore into stems.',
        'causes': 'Overwinter in soil; emerge in spring.',
        'control': 'Remove infested parts. Apply insecticides like spinosad or carbaryl. Encourage natural enemies.'
    },
    'stem_borer': {
        'details': 'Larvae bore into stems, causing wilting, dead hearts, reduced yield.',
        'causes': 'Eggs laid on leaves; larvae tunnel inside. Multiple crops affected (rice, maize, sugarcane).',
        'control': 'Use resistant varieties. Apply systemic insecticides at early infestation. Destroy crop residue. Release parasitic wasps (e.g., Cotesia).'
    },
    'wasp': {
        'details': 'Most are beneficial (parasitic wasps control pests); some (e.g., gall wasps) cause plant galls.',
        'causes': 'Diverse habitats; some attracted to certain plants.',
        'control': 'Usually not needed. Protect beneficial species. Remove galls if aesthetic issue.'
    },
    'weevil': {
        'details': 'Snout beetles; adults chew leaves, larvae bore into seeds, stems, roots (e.g., boll weevil, grain weevil).',
        'causes': 'Overwinter in residue; emerge in warm weather.',
        'control': 'Crop rotation. Use pheromone traps. Apply insecticides (e.g., malathion for stored grain). Destroy infested material.'
    }
}

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="Pest Detection App", page_icon="🐛", layout="wide")

st.title("🐛 Pest Detection & Management Advisor")
st.markdown("Upload an image of a pest-affected plant or insect. The model will detect the pest type and provide detailed control recommendations.")

# Main layout: two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the pest or affected plant part."
    )

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            img_resized = img.resize((224, 224))
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize

            # Predict
            with st.spinner("Detecting pest..."):
                prediction = model.predict(img_array)
                predicted_class_idx = np.argmax(prediction)
                confidence = float(np.max(prediction)) * 100
                predicted_pest = classes[predicted_class_idx]

            st.success("Detection Complete!")
            st.metric(label="Detected Pest", value=predicted_pest.capitalize(), delta=f"{confidence:.2f}% Confidence")

        except Exception as e:
            st.error(f"Error processing image: {e}")

with col2:
    st.subheader("Pest Information & Control")
    
    if 'predicted_pest' in locals():
        if predicted_pest in pest_info:
            info = pest_info[predicted_pest]
            
            st.markdown(f"### {predicted_pest.capitalize()}")
            with st.expander("Details", expanded=True):
                st.write(info['details'])
            with st.expander("Probable Causes"):
                st.write(info['causes'])
            with st.expander("How to Control (Efficient Methods)"):
                st.write(info['control'])
            
            st.info("**Tip**: Always scout fields regularly, use integrated pest management (IPM), and consult local extension services for region-specific advice.")
        else:
            st.warning("No detailed information available for this class.")
    else:
        st.info("Upload an image to see pest details and management strategies.")

# Footer
st.markdown("---")
st.caption("Model: Custom-trained Keras model | Input size: 224×224 | Classes: 13 common pests")
st.caption("Built with Streamlit • For educational and agricultural advisory use")
