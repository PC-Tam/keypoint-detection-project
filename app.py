"""
app.py
------
Streamlit demo application for Blood Cell Classification using
Harris Corner Detection, FAST, and ORB keypoint methods.

[!]️ Academic demonstration only — NOT for clinical/medical diagnosis.
"""

import sys
import os
import time
import numpy as np
import cv2
import joblib
import yaml
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

# -- Path setup -----------------------------------------------------------------
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import preprocess_cell_image, load_cell_image, resize_image, to_grayscale, apply_clahe
from segmentation_classical import create_cell_mask
from transformations import create_transformation_cases
from feature_harris import detect_harris_corners, draw_harris_corners, extract_harris_features
from feature_fast import detect_fast_keypoints, draw_fast_keypoints, extract_fast_features
from feature_orb import extract_orb_keypoints_descriptors, draw_orb_keypoints, extract_orb_statistical_features
from feature_morphology import extract_morphology_features
from feature_color_texture import extract_color_texture_features

# -- Page config ----------------------------------------------------------------
st.set_page_config(
    page_title="Blood Cell Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Premium CSS ----------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.warning-box {
    background: linear-gradient(135deg, #ffecd2, #fcb69f);
    border-left: 4px solid #ff6b6b;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #7f1d1d;
    font-weight: 500;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #475569;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #7dd3fc;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    border-bottom: 2px solid #334155;
    padding-bottom: 6px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# -- Config & Helpers -----------------------------------------------------------
@st.cache_data
def load_config():
    """Load project configuration."""
    for p in ["config.yaml", "config.yml"]:
        if Path(p).exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {}


@st.cache_data
def load_results():
    """Load training results CSV if available."""
    p = Path("outputs/reports/results.csv")
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def load_label_map():
    """Load label to class name mapping."""
    p = Path("data/processed/label_map.csv")
    if p.exists():
        df = pd.read_csv(p)
        return dict(zip(df["label"], df["class_name"]))
    return {}


def list_available_models():
    """List all available model files."""
    models_dir = Path("outputs/models")
    if not models_dir.exists():
        return []
    # Skip the kmeans vocabulary
    models = [f.name for f in models_dir.glob("*.joblib") if f.name != "orb_bovw_kmeans.joblib"]
    # Put best_model first if it exists
    if "best_model.joblib" in models:
        models.remove("best_model.joblib")
        models.insert(0, "best_model.joblib")
    return models


def load_selected_model(model_filename):
    """Load the user-selected model."""
    p = Path(f"outputs/models/{model_filename}")
    if p.exists():
        try:
            return joblib.load(str(p)), model_filename
        except Exception:
            return None, None
    return None, None


def load_bovw_kmeans():
    """Load BoVW KMeans vocabulary if available."""
    p = Path("outputs/models/orb_bovw_kmeans.joblib")
    if p.exists():
        try:
            return joblib.load(str(p))
        except Exception:
            return None
    return None


def bgr_to_rgb(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bytes_to_bgr(file_bytes):
    """Convert uploaded file bytes to OpenCV BGR image."""
    arr = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# -- Main App --------------------------------------------------------------------
config = load_config()
df_results = load_results()
label_map = load_label_map()
img_size = tuple(config.get("image_size", [256, 256]))
orb_n = config.get("orb_nfeatures", 500)
harris_bs = config.get("harris_block_size", 2)
harris_ks = config.get("harris_ksize", 3)
harris_k = config.get("harris_k", 0.04)
harris_tr = config.get("harris_threshold_ratio", 0.01)
fast_thresh = config.get("fast_threshold", 20)

# -- Header ---------------------------------------------------------------------
st.markdown('<div class="main-title">🔬 Blood Cell Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Harris Corner Detection · FAST · ORB — Traditional CV + Machine Learning</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="warning-box">[!]️ Academic demonstration only. NOT intended for clinical or medical diagnosis.</div>',
    unsafe_allow_html=True,
)

# -- Sidebar --------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    available_models = list_available_models()
    if available_models:
        selected_model_filename = st.selectbox(
            "🧠 ML Classification Model", 
            available_models,
            help="Select the model to use for prediction"
        )
    else:
        st.warning("No models found.")
        selected_model_filename = None

    with st.container():
        algo = st.selectbox(
            "🔑 Keypoint Algorithm",
            ["Harris Corner Detection", "FAST", "ORB"],
            help="Select keypoint detection algorithm to visualize",
        )
        transform_case = st.selectbox(
            "🔄 Transformation Case",
            [
                "Original",
                "Rotate 15°",
                "Gaussian Noise",
                "Change Brightness/Contrast",
                "Gaussian Blur",
            ],
        )

    st.markdown("---")

    if df_results is not None:
        st.markdown("### 📊 Model Performance")
        
        target_row = None
        if selected_model_filename:
            if selected_model_filename == "best_model.joblib":
                target_row = df_results.iloc[0] if len(df_results) > 0 else None
            elif "__" in selected_model_filename:
                f_set, m_name = selected_model_filename.replace('.joblib', '').split('__')
                matches = df_results[(df_results['feature_set'] == f_set) & (df_results['model'] == m_name)]
                if len(matches) > 0:
                    target_row = matches.iloc[0]
                    
        if target_row is not None:
            model_type = target_row.get('model', '?')
            st.success(f"**Model:** {model_type}")
            st.info(
                f"**Feature:** `{target_row.get('feature_set','?')}`\n\n"
                f"**Accuracy:** {target_row.get('accuracy',0):.2%}\n\n"
                f"**Macro F1:** {target_row.get('f1_macro',0):.2%}\n\n"
                f"**Precision:** {target_row.get('precision_macro',0):.2%}\n\n"
                f"**Recall:** {target_row.get('recall_macro',0):.2%}"
            )
            
            # Show a brief description of the ML algorithm
            ml_descriptions = {
                "SVM_RBF": "**SVM (RBF Kernel)**: Thuật toán Máy Vector Hỗ Trợ với nhân phi tuyến (đường cong). Rất mạnh trong việc tìm ra các ranh giới phân loại phức tạp giữa các loại tế bào máu.",
                "SVM_Linear": "**SVM (Linear)**: Phân loại bằng một đường thẳng cắt ngang. Tốc độ rất nhanh và hiệu quả khi các đặc trưng của tế bào có sự phân biệt rõ ràng.",
                "RandomForest": "**Random Forest**: Tập hợp của nhiều Cây quyết định. Chống nhiễu cực tốt và xử lý hoàn hảo việc kết hợp trộn lẫn nhiều loại đặc trưng (như hình thái + điểm chính).",
                "KNN": "**K-Nearest Neighbors (KNN)**: Phân loại một tế bào dựa trên 'K' tế bào có đặc điểm giống nó nhất trong tập dữ liệu. Đơn giản, trực quan nhưng có thể hơi chậm khi dự đoán.",
                "GaussianNB": "**Gaussian Naive Bayes**: Thuật toán xác suất thống kê dựa trên định lý Bayes. Tốc độ tính toán cực kỳ nhanh nhưng đôi khi kém chính xác với các cấu trúc tế bào phức tạp.",
                "LogisticRegression": "**Logistic Regression**: Mô hình toán học tuyến tính dự đoán xác suất. Là một nền tảng cơ bản cực tốt giúp đánh giá mức độ quan trọng của từng đặc trưng tế bào."
            }
            if model_type in ml_descriptions:
                st.caption(ml_descriptions[model_type])
                
        else:
            st.warning("No performance data available for this model.")
    else:
        st.warning("No trained models found.\nRun `python src/run_pipeline.py` first.")

    st.markdown("---")
    st.markdown("### ℹ️ About Algorithms")
    with st.expander("Harris"):
        st.markdown("""
Corner detector based on **intensity gradient matrix** (second-moment matrix).
- High response at corners
- Scale-invariant? ❌
- Rotation-invariant? ✅ (for small angles)
""")
    with st.expander("FAST"):
        st.markdown("""
**F**eatures from **A**ccelerated **S**egment **T**est.
- Compares pixel ring intensity
- Very fast (~ real-time)
- No descriptor built-in
""")
    with st.expander("ORB"):
        st.markdown("""
**O**riented FAST + **R**otated **B**RIEF.
- Full keypoint + binary descriptor
- Scale & rotation invariant
- Efficient matching with Hamming distance
""")

# -- Main content ---------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload a Blood Cell Image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Upload a microscopic blood cell image",
)

if uploaded_file is None:
    st.info(
        "👋 **Welcome!**\n\n"
        "Upload a blood cell microscopy image to begin analysis.\n\n"
        "The app will:\n"
        "- Preprocess and segment the cell\n"
        "- Apply the selected keypoint detection algorithm\n"
        "- Show detailed metrics\n"
        "- Predict the cell type (if a trained model is available)"
    )

    # Show sample benchmark results if available
    if df_results is not None and len(df_results) > 0:
        st.markdown("---")
        st.markdown("### 📈 Training Results Summary")
        st.dataframe(
            df_results[["feature_set", "model", "accuracy", "f1_macro", "f1_weighted"]]
            .head(15)
            .style.background_gradient(subset=["f1_macro"], cmap="Greens"),
            use_container_width=True,
        )
else:
    # -- Process uploaded image -------------------------------------------------
    with st.spinner("Processing image..."):
        try:
            img_bgr = bytes_to_bgr(uploaded_file)
            if img_bgr is None:
                st.error("Could not decode the uploaded image.")
                st.stop()

            img_bgr = resize_image(img_bgr, img_size)
            gray = to_grayscale(img_bgr)
            gray_clahe = apply_clahe(gray)

            # Apply transformation
            t_map = {
                "Original": "case1_original",
                "Rotate 15°": "case2_rotated",
                "Gaussian Noise": "case3_noisy",
                "Change Brightness/Contrast": "case4_brightness",
                "Gaussian Blur": "case5_blurred",
            }
            cases = create_transformation_cases(gray_clahe)
            selected_gray = cases[t_map[transform_case]]

            # Segmentation
            mask, bbox, contour = create_cell_mask(img_bgr)

            # Keypoint detection
            t0 = time.perf_counter()
            if algo == "Harris Corner Detection":
                corners = detect_harris_corners(
                    selected_gray, harris_bs, harris_ks, harris_k, harris_tr
                )
                kp_vis = draw_harris_corners(selected_gray, corners)
                n_kp = len(corners)
                pts = np.array([[c[0], c[1]] for c in corners], dtype=np.float32) if corners else np.empty((0, 2))

            elif algo == "FAST":
                kps = detect_fast_keypoints(selected_gray, threshold=fast_thresh)
                kp_vis = draw_fast_keypoints(selected_gray, kps)
                n_kp = len(kps)
                pts = np.array([kp.pt for kp in kps], dtype=np.float32) if kps else np.empty((0, 2))

            else:  # ORB
                kps, descs = extract_orb_keypoints_descriptors(selected_gray, nfeatures=orb_n)
                kp_vis = draw_orb_keypoints(selected_gray, kps)
                n_kp = len(kps)
                pts = np.array([kp.pt for kp in kps], dtype=np.float32) if kps else np.empty((0, 2))

            runtime_ms = (time.perf_counter() - t0) * 1000

            # Cell-region keypoint ratio
            h_img, w_img = mask.shape[:2]
            n_in_mask = 0
            if len(pts) > 0:
                for x, y in pts:
                    xi = int(np.clip(x, 0, w_img - 1))
                    yi = int(np.clip(y, 0, h_img - 1))
                    if mask[yi, xi] > 0:
                        n_in_mask += 1
            cell_ratio = n_in_mask / n_kp if n_kp > 0 else 0.0

        except Exception as e:
            st.error(f"Image processing error: {e}")
            st.stop()

    # -- Tabs -------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩻 Preprocessing",
        "🎯 Keypoints",
        "📊 Analysis",
        "🤖 Classification",
    ])

    with tab1:
        st.markdown('<div class="section-header">Image Preprocessing Pipeline</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(bgr_to_rgb(img_bgr), caption="Original (Resized)", use_container_width=True)
        with col2:
            st.image(gray_clahe, caption="Grayscale + CLAHE", use_container_width=True, clamp=True)
        with col3:
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if contour is not None:
                cv2.drawContours(mask_vis, [contour], -1, (0, 255, 0), 2)
            st.image(bgr_to_rgb(mask_vis), caption="Cell Mask (Otsu+HSV)", use_container_width=True)

    with tab2:
        st.markdown(
            f'<div class="section-header">Keypoint Detection: {algo} | Case: {transform_case}</div>',
            unsafe_allow_html=True,
        )
        col_img, col_metrics = st.columns([2, 1])
        with col_img:
            st.image(bgr_to_rgb(kp_vis), caption=f"{algo} — {n_kp} keypoints detected", use_container_width=True)
        with col_metrics:
            st.markdown(f"""
<div class="metric-card" style="margin-bottom:12px">
  <div class="metric-value">{n_kp:,}</div>
  <div class="metric-label">Keypoints</div>
</div>
<div class="metric-card" style="margin-bottom:12px">
  <div class="metric-value">{runtime_ms:.2f} ms</div>
  <div class="metric-label">Runtime</div>
</div>
<div class="metric-card">
  <div class="metric-value">{cell_ratio:.1%}</div>
  <div class="metric-label">In Cell Region</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### All Transformation Cases")
        case_cols = st.columns(5)
        case_labels = ["Original", "Rotate 15°", "Noise", "Brightness", "Blur"]
        for i, (case_key, label) in enumerate(zip(cases.keys(), case_labels)):
            case_img = cases[case_key]
            with case_cols[i]:
                st.image(case_img, caption=label, use_container_width=True, clamp=True)

    with tab3:
        st.markdown('<div class="section-header">Feature Analysis</div>', unsafe_allow_html=True)

        grid_size = tuple(config.get("grid_size", [4, 4]))
        resp_bins = config.get("response_hist_bins", 10)

        try:
            if algo == "Harris Corner Detection":
                feat_vec = extract_harris_features(
                    selected_gray, mask=mask, contour=contour,
                    block_size=harris_bs, ksize=harris_ks, k=harris_k,
                    threshold_ratio=harris_tr,
                    grid_size=grid_size, response_hist_bins=resp_bins,
                )
                feat_labels = (
                    ["n_kp", "mean_r", "max_r", "std_r", "density"]
                    + [f"grid_{i}" for i in range(grid_size[0] * grid_size[1])]
                    + [f"resp_hist_{i}" for i in range(resp_bins)]
                    + ["n_in_mask", "ratio_in_mask", "n_near_contour"]
                )
            elif algo == "FAST":
                feat_vec = extract_fast_features(
                    selected_gray, mask=mask, contour=contour,
                    threshold=fast_thresh, grid_size=grid_size,
                    response_hist_bins=resp_bins,
                )
                feat_labels = (
                    ["n_kp", "mean_r", "max_r", "std_r", "density"]
                    + [f"grid_{i}" for i in range(grid_size[0] * grid_size[1])]
                    + [f"resp_hist_{i}" for i in range(resp_bins)]
                    + ["n_in_mask", "ratio_in_mask", "n_near_contour"]
                )
            else:
                feat_vec = extract_orb_statistical_features(
                    selected_gray, mask=mask, contour=contour,
                    nfeatures=orb_n, grid_size=grid_size,
                    response_hist_bins=resp_bins,
                )
                feat_labels = (
                    ["n_kp", "mean_r", "max_r", "std_r", "density",
                     "mean_angle", "std_angle", "mean_size", "std_size"]
                    + [f"grid_{i}" for i in range(grid_size[0] * grid_size[1])]
                    + [f"resp_hist_{i}" for i in range(resp_bins)]
                    + ["n_in_mask", "ratio_in_mask", "n_near_contour"]
                )

            col_bar, col_morph = st.columns(2)
            with col_bar:
                st.markdown("**Keypoint Feature Vector**")
                df_feat = pd.DataFrame({"Feature": feat_labels[:len(feat_vec)], "Value": feat_vec})
                st.bar_chart(df_feat.set_index("Feature")["Value"])

            with col_morph:
                st.markdown("**Morphological Features**")
                morph_feat = extract_morphology_features(mask)
                morph_names = [
                    "area", "perimeter", "circularity", "eccentricity",
                    "aspect_ratio", "extent", "solidity", "equiv_diam",
                    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                ] + [f"hu_{i}" for i in range(7)] + ["cx", "cy", "mean_dist", "std_dist"]
                df_morph = pd.DataFrame({
                    "Feature": morph_names[:len(morph_feat)],
                    "Value": morph_feat
                })
                st.dataframe(df_morph.set_index("Feature"), use_container_width=True)

        except Exception as e:
            st.warning(f"Feature extraction failed: {e}")

    with tab4:
        st.markdown('<div class="section-header">Cell Type Classification</div>', unsafe_allow_html=True)

        if not selected_model_filename:
            st.warning(
                "No trained model found.\n\n"
                "Run the full pipeline first:\n"
                "```\npython src/run_pipeline.py\n```"
            )
        else:
            model, model_name = load_selected_model(selected_model_filename)
            if model is None:
                st.error("Failed to load selected model.")
            else:
                try:
                    # Assemble feature vector same as in feature_engineering
                    # Feature assembly for combined_all_traditional
    
                    # Try combined_all_traditional feature bundle if model expects it
                    # Build the feature on-the-fly
                    h_feat = extract_harris_features(
                        gray_clahe, mask=mask, contour=contour,
                        block_size=harris_bs, ksize=harris_ks, k=harris_k,
                        threshold_ratio=harris_tr, grid_size=grid_size, response_hist_bins=resp_bins,
                    )
                    f_feat = extract_fast_features(
                        gray_clahe, mask=mask, contour=contour,
                        threshold=fast_thresh, grid_size=grid_size, response_hist_bins=resp_bins,
                    )
                    o_stat = extract_orb_statistical_features(
                        gray_clahe, mask=mask, contour=contour,
                        nfeatures=orb_n, grid_size=grid_size, response_hist_bins=resp_bins,
                    )
    
                    kmeans = load_bovw_kmeans()
                    from feature_orb import extract_orb_keypoints_descriptors
                    _, o_desc = extract_orb_keypoints_descriptors(gray_clahe, nfeatures=orb_n)
                    n_clusters = config.get("bovw_clusters", 100)
                    if kmeans is not None:
                        from bovw import compute_bovw_histogram
                        o_bovw = compute_bovw_histogram(o_desc, kmeans)
                    else:
                        o_bovw = np.zeros(n_clusters, dtype=np.float32)
    
                    morph = extract_morphology_features(mask)
                    ct_feat = extract_color_texture_features(img_bgr, gray_clahe, mask=mask)
    
                    # Determine feature set from filename
                    feature_set = "combined_all_traditional" # Default
                    if "__" in selected_model_filename:
                        feature_set = selected_model_filename.split("__")[0]
                    elif selected_model_filename == "best_model.joblib":
                        # We know best_model was combined_all_traditional from results
                        feature_set = "combined_all_traditional"
                    
                    # Assemble the correct feature vector
                    if feature_set == "harris_only":
                        X = h_feat
                    elif feature_set == "fast_only":
                        X = f_feat
                    elif feature_set == "orb_stats_only":
                        X = o_stat
                    elif feature_set == "orb_bovw":
                        X = o_bovw
                    elif feature_set == "harris_morphology":
                        X = np.hstack([h_feat, morph])
                    elif feature_set == "fast_morphology":
                        X = np.hstack([f_feat, morph])
                    elif feature_set == "orb_bovw_morphology":
                        X = np.hstack([o_bovw, morph])
                    elif feature_set == "combined_harris_fast_orb":
                        X = np.hstack([h_feat, f_feat, o_stat])
                    else: # combined_all_traditional
                        X = np.hstack([h_feat, f_feat, o_stat, o_bovw, morph, ct_feat])
                    
                    X = X.reshape(1, -1)
    
                    try:
                        pred_label = int(model.predict(X)[0])
                        pred_class = label_map.get(pred_label, f"class_{pred_label}")
    
                        col_pred, col_proba = st.columns([1, 2])
                        with col_pred:
                            st.success(f"### Predicted Class\n# {pred_class.replace('_', ' ').title()}")
                            st.caption(f"Model: `{model_name}`")
    
                        with col_proba:
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(X)[0]
                                class_names = [label_map.get(i, str(i)) for i in range(len(proba))]
                                df_proba = pd.DataFrame({
                                    "Class": class_names,
                                    "Probability": proba,
                                }).sort_values("Probability", ascending=False)
                                st.markdown("**Class Probabilities**")
                                st.bar_chart(df_proba.set_index("Class")["Probability"])
    
                    except Exception as e:
                        # Model may expect a specific feature bundle with different size
                        st.warning(
                            f"Prediction skipped: feature mismatch.\n\n"
                            f"The loaded best_model may expect a specific feature bundle "
                            f"with a different number of dimensions.\n\nError: {e}"
                        )
    
                except Exception as e:
                    st.error(f"Classification error: {e}")

        st.markdown("---")
        st.caption(
            "[!]️ This is an academic demonstration project. "
            "Results are based on traditional computer vision (no deep learning). "
            "Do NOT use for medical diagnosis."
        )
