import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
import sys
import os
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import resize_image, extract_green_channel, apply_clahe
from feature_harris import extract_harris_features, draw_harris_corners
from feature_fast import extract_fast_features, draw_fast_keypoints
from feature_orb import extract_orb_descriptors, draw_orb_keypoints
from bovw import compute_bovw_histogram, load_bovw_model
import joblib

# --- Page Config ---
st.set_page_config(page_title="AI Glaucoma Dashboard", page_icon="👁️", layout="wide")

# --- Custom Premium CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit default menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Typography and background */
    body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Image Cards styling */
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        transition: transform 0.2s ease-in-out;
    }
    .stImage > img:hover {
        transform: scale(1.01);
    }
    
    /* Headers */
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling (Targeting by attribute) */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Tabs Styling */
    [data-baseweb="tab"] {
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_results():
    path = Path("outputs/reports/results.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

def load_ml_model(feature_name, model_type):
    model_path = Path(f"outputs/models/{model_type}_{feature_name}.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None

config = load_config()
df_results = load_results()

# --- SIDEBAR (Settings) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004383.png", width=80) # Eye icon
    st.markdown("### ⚙️ System Config")
    
    with st.container(border=True):
        feature_method = st.selectbox(
            "1. Feature Extraction Method:",
            ("Harris Corners", "FAST Keypoints", "ORB Keypoints")
        )
        
        model_type = st.selectbox(
            "2. Machine Learning Model:",
            ("RandomForest", "SVM", "KNN", "GNB")
        )
    
    # Show training accuracy if available
    feat_name_key = "Harris" if feature_method == "Harris Corners" else "FAST" if feature_method == "FAST Keypoints" else "ORB_BoVW"
    
    if df_results is not None:
        model_perf = df_results[(df_results['Model'] == model_type) & (df_results['Feature'] == feat_name_key)]
        if not model_perf.empty:
            f1_score = model_perf.iloc[0]['F1-Score'] * 100
            with st.container(border=True):
                st.metric(label="🎯 Training F1-Score", value=f"{f1_score:.2f}%")
        else:
            st.warning("⚠️ No training data found for this combination.")
            
    st.markdown("---")
    with st.expander("ℹ️ About Methods", expanded=True):
        st.markdown("- **Harris/FAST:** Good for corner and vessel edge detection.\n- **ORB:** Scale and rotation invariant.")

# --- MAIN DASHBOARD ---
st.markdown('<div class="main-header">👁️ Glaucoma Diagnosis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated fundus image analysis using Computer Vision & Machine Learning.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a Fundus Image", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")

if uploaded_file is None:
    # --- EMPTY STATE ---
    st.info("👋 **Welcome to the Glaucoma Diagnosis Dashboard!**\n\nPlease follow these steps to analyze an image:\n1. Open the sidebar on the left and select your preferred Feature Extraction method and Machine Learning model.\n2. Click 'Browse files' above to upload a fundus image.\n3. Wait a moment for the AI to process and return the diagnostic result.")
else:
    # Process uploaded image
    with st.spinner('Processing image and extracting features...'):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        img_bgr_resized = resize_image(image_bgr, tuple(config['preprocessing']['image_size']))
        green_channel = extract_green_channel(img_bgr_resized)
        clahe_img = apply_clahe(green_channel)
        
        # Feature Extraction
        feature_vector = None
        num_keypoints = 0
        
        if feature_method == "Harris Corners":
            vis_img = draw_harris_corners(clahe_img)
            feature_vector = extract_harris_features(clahe_img, **{k:v for k,v in config['features']['harris'].items() if k != 'enabled'})
            num_keypoints = int(feature_vector[0])
        elif feature_method == "FAST Keypoints":
            vis_img = draw_fast_keypoints(clahe_img)
            feature_vector = extract_fast_features(clahe_img, **{k:v for k,v in config['features']['fast'].items() if k != 'enabled'})
            num_keypoints = int(feature_vector[0])
        elif feature_method == "ORB Keypoints":
            vis_img = draw_orb_keypoints(clahe_img, nfeatures=config['features']['orb']['nfeatures'])
            kp, des = extract_orb_descriptors(clahe_img, nfeatures=config['features']['orb']['nfeatures'])
            num_keypoints = len(kp)
            bovw_path = Path("outputs/models/orb_bovw_kmeans.joblib")
            if bovw_path.exists():
                kmeans = load_bovw_model(bovw_path)
                feature_vector = compute_bovw_histogram(des, kmeans)
                
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # PREDICTION RESULT (Highlight)
        st.markdown("### 🏆 Diagnostic Result")
        model = load_ml_model(feat_name_key, model_type)
        
        if feature_vector is None or (feature_method == "ORB Keypoints" and not bovw_path.exists()):
            st.error("❌ Failed to extract features or missing BoVW model.")
        elif model is None:
            st.error(f"❌ Model {model_type} ({feat_name_key}) not found. Please run the training pipeline first.")
        else:
            # Predict
            X_input = feature_vector.reshape(1, -1)
            pred = model.predict(X_input)[0]
            
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                confidence = proba[pred] * 100
                
            with st.container(border=True):
                col_res1, col_res2 = st.columns([1, 1])
                with col_res1:
                    if pred == 0:
                        st.success("🟢 **CONCLUSION: NORMAL**")
                    else:
                        st.error("🔴 **CONCLUSION: GLAUCOMA**")
                
                with col_res2:
                    if confidence:
                        st.caption(f"AI Confidence Score: **{confidence:.2f}%**")
                        st.progress(int(confidence) / 100)
    
        st.markdown("---")
        
        # ANALYSIS INTERFACE (TABS)
        tab1, tab2, tab3 = st.tabs(["🩻 Preprocessing", "🎯 Keypoint Analysis", "📊 Feature Extraction"])
        
        with tab1:
            with st.container(border=True):
                st.markdown("#### Contrast Enhancement & Filtering")
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.image(cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB), caption="1. Original (Resized)", use_container_width=True)
                with col_t2:
                    st.image(green_channel, caption="2. Green Channel", use_container_width=True)
                with col_t3:
                    st.image(clahe_img, caption="3. CLAHE Enhanced", use_container_width=True)
    
        with tab2:
            with st.container(border=True):
                col_k1, col_k2 = st.columns([2, 1])
                with col_k1:
                    st.image(vis_img_rgb, caption=f"Detection Algorithm: {feature_method}", use_container_width=True)
                with col_k2:
                    st.metric(label="Total Keypoints Detected", value=f"{num_keypoints:,}")
                    st.info("These keypoints represent structural changes in the fundus (e.g., optic disc, vessel edges). The Machine Learning model uses the distribution of these points to make its decision.")
    
        with tab3:
            if feature_vector is not None:
                with st.container(border=True):
                    st.markdown(f"#### Feature Vector Chart ({len(feature_vector)} dimensions)")
                    
                    # Visualize the feature array
                    chart_data = pd.DataFrame(feature_vector, columns=["Value"])
                    st.bar_chart(chart_data)
                    
                    with st.expander("View Raw Array Data"):
                        st.dataframe(chart_data.T)
