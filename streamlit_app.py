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
    page_title="AI Pest Detection and Diagnosis | Using CNN and XAI",
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
        'details': 'Small (1-6mm) sap-sucking hemipterans belonging to scales and greenfly families. Colonies develop rapidly on new shoots, concentrating on juvenile leaves and flower buds. Multiple generations per year in warm climates; winged morphs disperse in response to plant stress, chemical control, or population peaks. Common host families include Rosaceae, Solanaceae, Fabaceae. Greenfly scales colonize stems and branches, leaving dense, sticky honeydew that can girdle twigs.',
        'cause': 'Population explosions triggered by nitrogen surplus (0.5-2% N in leaf tissue indicates deficiency, but 2-5% indicates surplus that favors aphid growth). Lack of natural predators occurs when: (1) Summer dormancy insects are killed by heat/drought, (2) Multiple generations deplete specialist parasitoids before their host stage completes, (3) Biological control is disrupted by pesticides targeting parasitoids or predators (daemons, ladybugs, lacebugs). Alternative host plants in wind pollinated crops (corn, soy) provide refuges for aphid reproduction.',
        'control': 'Monitor with sweep net or sticky traps; act at first node expansion. Apply oils or soaps at dawn when aphids are concentrated on lower leaves but active on stems. Use neem oil (2-4 tsp/gal) every 7-10 days through host season. Insecticidal soaps (2-4 tsp/gal) effective at population peaks if applied early. Apply to new growth before significant expansion. Rotate control methods to prevent resistance. Introduce parasitoids (for scales) or predators (ladybugs at 10-25/lb) for long-term suppression. Avoid pesticides that target parasitoid development (3-7 day-old target).',
        'risk_level': 'Moderate'
    },
    'armyworm': {
        'name': 'Armyworm',
        'details': 'Larvae (3-15mm) of nocturnal moths (family Noctuidae). Move in coordinated columns, stripping foliage entirely and sometimes girdling stems. Mature in soil or leaf litter; some complete development in one season. Distinguished by continuous regeneration: each death creates replacement larva from egg. Major agricultural pests in temperate regions; some species have tropical origins and colonize temperate areas via wind dispersal.',
        'cause': 'Development favored by: (1) Grass-covered fields with native grasses (preferred oviposition sites), (2) Cool, wet springs that favor egg hatching and early larval development. Population dynamics involve multiple generations per year, with larvae overwintering in soil or plant debris. Agricultural practices like monoculture planting and pesticide use select for resilient populations. Wind-driven dispersal introduces species to new regions, establishing permanent or cyclical populations.',
        'control': 'Monitor with sweep nets or yellow sticky traps. Apply entomicides to seed or seed coats before emergence. Use insecticidal sprays targeting new growth. Apply targeted control at population thresholds. Inject soil-applied pesticides at seedling line. For large infestations, apply systemic seed treatment or pre-emergence applications. Utilize cultural practices like crop rotation and maintaining clean field edges to interrupt life cycle. Consider using beneficial insects like *Trichogramma* species for biological control.',
        'risk_level': 'High'
    },
    'beetle': {
        'name': 'Beetle',
        'details': 'Hard-shelled (elytrine) insects from order Coleoptera. Adults and larvae (caterpillars) feed on foliage, stems, flowers, and fruits. Leaf skeletonizers leave characteristic "window" patterns. Larval feeding often begins beneath soil line or within plant debris. Larvae of some species are root feeders in crops. Adults can be green, black, red, or colorful species mimicking other insects.',
        'cause': 'Annual or multi-year populations established through consistent overwintering habitats. Larvae develop in soil, crop residue, or within plant tissues. Soil temperature and moisture critically influence development and survival. Cold stress limits population establishment, while optimal temperatures (15-25°C/59-77°F) and moderate moisture favor development. Crop rotation and residue management help disrupt life cycles.',
        'control': 'Early-season intervention most effective. Manually remove adults from leaves and stems. Apply floating row covers at planting to exclude adults. Use insecticidal sprays targeting new growth as population increases. Apply soil-applied insecticides at planting. For root-feeders, inject soil applications targeting larval development. Utilize beneficial nematodes (*Steinernema*, *Heterorhabditis* species) - apply at 200-500 worms per square foot. Rotate to non-susceptible crops. Consider integrated approaches combining mechanical, biological, and chemical methods.',
        'risk_level': 'Moderate'
    },
    'bollworm': {
        'name': 'Bollworm',
        'details': 'Tetramonellid moth larvae (3-12mm) damaging cotton, soy, and other leguminous crops. Larvae burrow into bolls, feeding on seeds and developing fruits. Single larval stage percesent in crops; adults complete metamorphosis and seek mates. Cotton bollworm (*Heliothis* spp.) and tomato tomato hornworm (*Manduca* spp.) most damaging in U.S. agricultural regions.',
        'cause': 'Continuous populations maintained by: (1) Multiple generations per year in warm climates, (2) Monoculture practices limiting natural predator diversity, (3) Wind dispersal introducing new populations to non-native regions, (4) Pesticide use selecting for resistant morphs. Alternative host plants expand population before crop-specific damage occurs.',
        'control': 'Monitor with pheromone traps targeting male release. Apply targeted insecticide applications at population thresholds. Utilize resistant crop varieties through certified seed selection. Consider non-chemical strategies: crop rotation, maintaining diverse native plant buffers, timely planting to minimize vulnerable growth stages. For cotton, integrate vertical row spacing to reduce boll accessibility. Explore biological controls using *Bacillus thuringiensis* targeting larval stages. Apply insecticide sprays focusing on first-stage boll expansion when larvae are small and vulnerable.',
        'risk_level': 'High'
    },
    'catterpillar': {
        'name': 'Caterpillar',
        'details': 'Holometabolous insect larval stage preceding pupation and metamorphosis. Moths (order Lepidoptera) lay eggs on host plants; larvae hatch and consume foliage, growing through molts. Different species attack various plants, leaving irregular holes or continuous skeletonization. Some species have specialized feeding behaviors like leaf-tunneling or stem-boring.',
        'cause': 'Eggs laid on host plants; hatched larvae consume developing leaves. Population dynamics depend on host availability, climate conditions, and food source continuity. Multiple generations may occur in single growing season under favorable conditions.',
        'control': 'Apply *Bacillus thuringiensis* (Bt) sprays targeting caterpillar larvae. Use insecticidal soaps for direct leaf feeding. Manual removal effective for small infestations. Encourage natural predators like birds, parasitic wasps, and beneficial insects. Utilize row covers to prevent egg-laying. Consider crop rotation to disrupt life cycles and reduce population establishment.',
        'risk_level': 'Moderate'
    },
    'earthworms': {
        'name': 'Earthworm',
        'details': 'Ecosystem engineers critical for soil health. Enhance aeration, water infiltration, and nutrient cycling through continuous surface-subsurface movement. Excretion (castes) releases organic matter decomposition bacteria, boosting soil fertility. Different species specialize in surface (lumbricoides), burrowing (eiseni), or marine environments, with similar engineering functions across ecosystems.',
        'cause': 'Natural soil inhabitants requiring specific habitat conditions. Successful populations need: undisturbed soil, adequate moisture, appropriate temperature, organic matter availability, and suitable host vegetation. Human activities like over-tillage, deforestation, urbanization, and pesticide use can disrupt earthworm communities.',
        'control': 'No intervention required; actively manage to support earthworm populations. Practice minimal soil disturbance, maintain moisture balance, incorporate organic amendments, and protect vegetation cover. Avoid excessive chemical interventions that can negatively impact earthworm health and population dynamics.',
        'risk_level': 'None'
    },
    'grasshopper': {
        'name': 'Grasshopper',
        'details': 'Short-horned grasshoppers (family Acrididae) with robust bodies and large hind legs for jumping. Adults and nymphs feed on grasses and herbaceous vegetation, consuming foliage entirely. Some species damage agricultural crops, turfgrass, and native ranges.',
        'cause': 'Population surges triggered by: (1) Prolonged drought followed by rain saturating grasses, (2) Vegetation cover expansion in agricultural periphery zones, (3) Agricultural practices like monoculture and residue retention that favor grassy habitats. Cool, wet springs favor egg hatching and early larval development.',
        'control': 'Monitor with sweep nets; act at first significant vegetation damage. Apply topical insecticides to target adults. Utilize granular applications in field periphery areas. Implement biological controls: apply *Nosema* spp. spores, introduce parasitic wasps (*Trichogramma*), or use *Bacillus* spp. bacteria. Apply insecticidal baits like Nolo in fall before vegetation expansion. Conduct field tillage to disrupt soil-bound populations.',
        'risk_level': 'High'
    },
    'mites': {
        'name': 'Spider Mites',
        'details': 'Microscopic arachnids (family Tetranychidae) causing significant plant damage despite small size. Yellow stippling results from leaf tissue expansion between mineralized salt crystals deposited by feeding mites. Infestations often invisible to unaided eye due to minuscule body size (100-400μm).',
        'cause': 'Population establishment favored by: (1) Elevated temperatures (optimal 25-35°C/77-95°F), (2) Low humidity (<40-50% RH), (3) Intensive agriculture with row spacing limiting microclimate modification, (4) Sequential crop selection allowing population establishment and reproduction.',
        'control': 'Increase environmental humidity via misting or strategic irrigation. Apply horticultural oils or insecticidal sprays targeting mature adults. Introduce predatory mites (*Neoseiulus*, *Panonyssus* species) as biological control. Avoid broad-spectrum pesticides that eliminate natural predators and can accelerate population growth. Rotate control methods to prevent resistance development.',
        'risk_level': 'Moderate'
    },
    'moth': {
        'name': 'Moth',
        'details': 'Adult stage of holometabolous insects (order Lepidoptera). Female moths disperse via flight to locate host plants, releasing pheromones that attract male counterparts. Adult moths may feed on nectar, decaying vegetation, or specialized sources like apple tree sap. Some species have adapted to feed on specific plant types or developmental stages.',
        'cause': 'Continuous population cycles depend on host availability and developmental conditions. Multiple generations may occur in temperate regions, with some species having extended life cycles spanning multiple years.',
        'control': 'Utilize yellow sticky traps positioned near host plants to capture adult moths. Apply pheromone traps to interfere with mating processes. Install protective netting around valuable growth areas. Consider broad-spectrum pesticides as last resort, targeting active growth stages when population is manageable.',
        'risk_level': 'Low (Directly)'
    },
    'sawfly': {
        'name': 'Sawfly Larvae',
        'details': 'Wasps (Hymenoptera) with caterpillar-like appearance. Larvae feed on leaves, stems, or developing fruits, causing skeletonization or other tissue damage. Distinguishable by: (1) Sharp, saw-like ovipositor used to penetrate host tissue, (2) Elaborate head structures resembling caterpillar prolegs, (3) Diverse host plant preferences across multiple ecosystems.',
        'cause': 'Spring emergence occurs as larvae complete developmental stage in soil or protected areas. Environmental conditions like temperature and moisture influence timing and population dynamics. Agricultural practices such as forest edge management can affect habitat suitability.',
        'control': 'Apply insecticidal sprays targeting new growth to prevent leaf penetration. Utilize horticultural oils carefully, ensuring coverage while minimizing crop damage. Introduce natural predators like parasitic wasps. Rotate crops annually to disrupt continuous population cycles.',
        'risk_level': 'Moderate'
    },
    'stem_borer': {
        'name': 'Stem Borer',
        'details': 'Larval stage of noctuid moths (family Noctuidae). Pupate in soil or protected areas, emerging as adult moths that disperse to locate host plants. Larval feeding begins by tunneling through soil into stems, where feeding continues and growth occurs.',
        'cause': 'Development favored by: (1) Continuous crop fields with minimal interruption, (2) Soil conditions supporting prolonged larval development (moisture, temperature, organic content), (3) Host plant continuity without crop removal or disturbance.',
        'control': 'Emergence interception: apply insecticides at soil line as adults begin active development. Utilize pheromone traps to monitor population levels and time interventions. Apply targeted sprays targeting new growth. Manually remove and destroy infested plant parts. Implement strict field sanitation practices. Rotate to non-susceptible crop species.',
        'risk_level': 'Severe'
    },
    'wasp': {
        'name': 'Wasp',
        'details': 'Hymenopteran insects including both beneficial pollinators and destructive pests. Diverse ecological roles: some species provide ecosystem services through pollination; others attack insect nests, preying on caterpillars and other insects. Human interaction ranges from minimal (mostly avoidant) to highly destructive (nest defense responses).',
        'cause': 'Natural habitat determination by species. Attractant species seek specific nesting conditions like protected areas, minimal competition, and suitable climate. Human interaction occurs through habitat proximity, landscape management practices, and agricultural ecosystem dynamics.',
        'control': 'Preventative for agricultural settings: maintain garden distance from human areas, utilize protective row covers, select non-attractive plant species. Manage existing populations: apply targeted entomicide applications around nests when safe to do so. Encourage natural predator populations including birds, bats, and parasitic wasps. Avoid broad-spectrum pesticides that disrupt ecosystem balance.',
        'risk_level': 'Low (to crops)'
    },
    'weevil': {
        'name': 'Weevil',
        'details': 'Snouted beetles (family Curculionidae) with distinct labral projection (snout). Larvae are often called "roundworms" due to cylindrical appearance; they feed on plant tissues, stored products, or other organic materials. Diverse habitat preferences across agricultural, garden, and ecosystem boundaries.',
        'cause': 'Population establishment depends on: (1) Continuous food source availability, (2) Suitable environmental conditions supporting developmental stages, (3) Landscape features supporting overwintering or pupal development, (4) Agricultural practices supporting population persistence like crop residue retention or seed contamination.',
        'control': 'Source reduction: implement strict crop residue management, utilize clean seed selection, apply thermal treatment to stored products. Utilize diatomaceous earth as physical barrier. Apply targeted insecticides targeting adult emergence. Support biological control through natural predator integration and minimal pesticide use.',
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
    Developed by Kwasu  Student
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
        st.title("Explainable Pest Detection")
        st.caption("v2.0 | AI-Powered Pest Detection")
       
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
        st.title("Plant Pest Detection and Diagnostics")
        st.markdown("Upload an image or use your camera to detect pests and get immediate control recommendations.")
    with col_hero2:
        # Simulated weather widget
        st.info(f"📍 Local Conditions: 29°C, Humidity 67%")
    model = get_model()
    if not model:
        st.stop()
    # Input Logic
    processed_image = None
   
    if input_mode == "📸 Camera":
        img_file = st.camera_input("Take a clear picture of the pest to get detail informations")
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
                st.session_state.random_pests = random.sample(CLASS_NAMES, min(10, len(CLASS_NAMES)))
            if 'test_images' not in st.session_state:
                st.session_state.test_images = {}
                for pest in st.session_state.random_pests:
                    class_url = f"https://api.github.com/repos/blurerjr/Explainable-Pest-Detection/contents/test/?ref=f4cca1b404e409c763658409ac2d36250bdcddb2"
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
            cols = st.columns(10)
            for i, pest in enumerate(st.session_state.random_pests):
                with cols[i]:
                    img_name = st.session_state.test_images.get(pest)
                    if img_name:
                        raw_url = f"https://raw.githubusercontent.com/blurerjr/Explainable-Pest-Detection/f4cca1b404e409c763658409ac2d36250bdcddb2/test/{img_name}"
                        st.image(raw_url, use_column_width=True)
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
