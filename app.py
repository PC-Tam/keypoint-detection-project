import streamlit as st
import pandas as pd
from PIL import Image
import os
import cv2
import numpy as np
from methods.harris_detector import detect_harris
from methods.fast_detector import detect_fast
from methods.orb_detector import detect_orb
from methods.preprocessing import auto_preprocess, analyze_image, format_analysis_report, preprocess_image, DEFAULT_PREPROCESS_OPTIONS
from src.matcher import match_orb

# CẤU HÌNH TRANG CHÍNH & KHỞI TẠO TABS
st.set_page_config(
    page_title="Keypoint & Matching Studio",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Keypoint & Product Matching Studio")
st.markdown("Giao diện kiểm thử hệ thống nhận diện nhãn sản phẩm/logo sử dụng OpenCV truyền thống.")
st.write("---")

# Khởi tạo 3 Tabs chính theo yêu cầu bài toán
tab1, tab2, tab3 = st.tabs([
    "Keypoint Detection", 
    "Logo/Product Matching", 
    "Evaluation"
])


# KHÔNG GIAN TAB 1 - KEYPOINT DETECTION
with tab1:
    st.header("🎯 Phát hiện Điểm Đặc Trưng")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Cấu hình Input")
        uploaded_file_t1 = st.file_uploader(
            "Chọn ảnh để trích xuất đặc trưng...", 
            type=["jpg", "jpeg", "png"], 
            key="t1_single_uploader"
        )
        
        algorithm = st.selectbox(
            "Chọn thuật toán:",
            ("Harris Corner Detection", "FAST", "ORB")
        )
        
        # --- Tiền xử lý ảnh ---
        st.write("---")
        preprocess_mode = st.radio(
            "🔧 Tiền xử lý ảnh:",
            ["Tự động (đề xuất)", "Thủ công", "Tắt"],
            index=0,
            key="t1_preprocess_mode",
            horizontal=True
        )
        
        # Tham số thủ công (chỉ hiển khi chọn Thủ công)
        use_gaussian = False
        gaussian_ksize = 5
        use_clahe = False
        clahe_clip = 2.0
        use_bilateral = False
        bilateral_d = 9
        bilateral_sigma = 75
        
        if preprocess_mode == "Thủ công":
            use_gaussian = st.checkbox("Gaussian Blur", value=True, key="t1_gaussian")
            if use_gaussian:
                gaussian_ksize = st.slider("  Kernel Size", 3, 15, 5, step=2, key="t1_gk")
            use_bilateral = st.checkbox("Bilateral Filter", value=False, key="t1_bilateral")
            if use_bilateral:
                bilateral_d = st.slider("  Diameter", 5, 15, 9, step=2, key="t1_bd")
                bilateral_sigma = st.slider("  Sigma", 25, 150, 75, step=25, key="t1_bs")
            use_clahe = st.checkbox("CLAHE", value=False, key="t1_clahe")
            if use_clahe:
                clahe_clip = st.slider("  Clip Limit", 1.0, 10.0, 2.0, step=0.5, key="t1_cc")
        
        # --- Tham số thuật toán ---
        st.write("---")
        st.caption("⚙️ Cấu hình tham số thuật toán:")
        if algorithm == "Harris Corner Detection":
            block_size = st.slider("Block Size", 2, 10, 2)
            ksize = st.slider("Aperture Parameter (ksize)", 3, 31, 3, step=2)
        elif algorithm == "FAST":
            threshold = st.slider("Threshold", 1, 100, 20)
        elif algorithm == "ORB":
            n_features = st.slider("Max Features (nfeatures)", 100, 5000, 1500, step=100)
            
        btn_run_t1 = st.button("Chạy Phát Hiện Keypoints", type="primary", key="btn_t1")

    with col2:
        st.subheader("Kết quả Hiển thị")
        if uploaded_file_t1 is not None:
            img_t1 = Image.open(uploaded_file_t1).convert("RGB")
            img_np = np.array(img_t1)
            
            # === Áp dụng tiền xử lý ===
            # Map tên thuật toán cho preprocessing module
            algo_map = {
                "Harris Corner Detection": "Harris",
                "FAST": "FAST",
                "ORB": "ORB"
            }
            
            if preprocess_mode == "Tự động (đề xuất)":
                img_processed, preprocess_report = auto_preprocess(
                    img_np, algorithm=algo_map[algorithm]
                )
            elif preprocess_mode == "Thủ công":
                img_processed = img_np.copy()
                manual_steps = []
                if use_gaussian:
                    img_processed = cv2.GaussianBlur(img_processed, (gaussian_ksize, gaussian_ksize), 0)
                    manual_steps.append(f"Gaussian (k={gaussian_ksize})")
                if use_bilateral:
                    img_processed = cv2.bilateralFilter(img_processed, bilateral_d, bilateral_sigma, bilateral_sigma)
                    manual_steps.append(f"Bilateral (d={bilateral_d})")
                if use_clahe:
                    lab = cv2.cvtColor(img_processed, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    img_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    manual_steps.append(f"CLAHE (clip={clahe_clip})")
                preprocess_report = {
                    "summary": " → ".join(manual_steps) if manual_steps else "Không áp dụng",
                    "analysis": analyze_image(img_np),
                    "steps_applied": manual_steps,
                }
            else:  # Tắt
                img_processed = img_np
                preprocess_report = {
                    "summary": "Tắt tiền xử lý",
                    "analysis": analyze_image(img_np),
                    "steps_applied": [],
                }
            
            # === Kiểm tra có thay đổi hay không ===
            has_preprocessing = not np.array_equal(img_np, img_processed)
            
            # === Hiển thị ===
            if has_preprocessing:
                out_col1, out_col2, out_col3 = st.columns(3)
            else:
                out_col1, out_col3 = st.columns(2)
            
            with out_col1:
                st.image(img_t1, caption="Ảnh Gốc", use_container_width=True)
            
            if has_preprocessing:
                with out_col2:
                    st.image(img_processed, caption="Sau Tiền Xử Lý", use_container_width=True)
            
            # Hiển thị báo cáo phân tích
            if preprocess_mode == "Tự động (đề xuất)":
                with st.expander("📊 Xem phân tích ảnh & bước tiền xử lý", expanded=False):
                    st.markdown(format_analysis_report(preprocess_report))
            
            with out_col3:
                if btn_run_t1:
                    st.info(f"Đang xử lý trích xuất bằng {algorithm}...")
                    
                    if algorithm == "Harris Corner Detection":
                        result = detect_harris(img_processed, block_size=block_size, ksize=ksize)
                    elif algorithm == "FAST":
                        result = detect_fast(img_processed, threshold=threshold)
                    elif algorithm == "ORB":
                        result = detect_orb(img_processed, nfeatures=n_features)
                    
                    result_rgb = cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption=f"Kết quả {algorithm}", use_container_width=True)
                    st.success(f"✅ Phát hiện **{result['keypoints_count']}** keypoints trong **{result['runtime_ms']:.2f} ms**")
                else:
                    st.warning("Nhấn nút 'Chạy Phát Hiện Keypoints' để xem kết quả.")
        else:
            st.info("Vui lòng upload ảnh ở cột bên trái.")


# KHÔNG GIAN TAB 2 - LOGO/PRODUCT MATCHING
with tab2:
    st.header("🧩 Khớp Nhãn Sản Phẩm & Logo")
    t2_col_left, t2_col_right = st.columns([1, 2])
    
    with t2_col_left:
        st.subheader("Dữ liệu Đầu Vào")
        template_file = st.file_uploader(
            "1. Upload ảnh Template (Ảnh gốc mẫu nhãn/logo)", 
            type=["jpg", "jpeg", "png"], 
            key="t2_template"
        )
        
        test_file = st.file_uploader(
            "2. Upload ảnh Test (Ảnh hiện trường thực tế)", 
            type=["jpg", "jpeg", "png"], 
            key="t2_test"
        )
        
        # ── Preprocessing Options ──────────────────────────────────
        st.write("---")
        st.caption("🔧 Tiền xử lý ảnh (Preprocessing):")
        
        t2_resize_max = st.slider(
            "Resize max size (px)", 256, 2048, 1024, step=128, key="t2_resize"
        )
        t2_use_clahe = st.checkbox("Enable CLAHE", value=True, key="t2_clahe")
        t2_clahe_clip = 2.0
        if t2_use_clahe:
            t2_clahe_clip = st.slider(
                "  CLAHE clipLimit", 1.0, 8.0, 2.0, step=0.5, key="t2_clahe_clip"
            )
        t2_use_gaussian = st.checkbox("Enable Gaussian Blur", value=False, key="t2_gaussian")
        t2_gaussian_k = 5
        if t2_use_gaussian:
            t2_gaussian_k = st.slider(
                "  Gaussian kernel", 3, 15, 5, step=2, key="t2_gk"
            )
        t2_use_median = st.checkbox("Enable Median Blur", value=False, key="t2_median")
        t2_median_k = 5
        if t2_use_median:
            t2_median_k = st.slider(
                "  Median kernel", 3, 9, 5, step=2, key="t2_mk"
            )
        t2_use_contrast = st.checkbox("Enable Contrast Stretching", value=False, key="t2_contrast")
        t2_use_sharpen = st.checkbox("Enable Sharpen", value=False, key="t2_sharpen")
        t2_sharpen_str = 0.5
        if t2_use_sharpen:
            t2_sharpen_str = st.slider(
                "  Sharpen strength", 0.1, 1.5, 0.5, step=0.1, key="t2_ss"
            )
        
        # ── ORB Parameters ─────────────────────────────────────────
        st.write("---")
        st.caption("⚙️ Tham số ORB:")
        t2_nfeatures = st.slider(
            "nfeatures", 500, 5000, 1500, step=100, key="t2_nfeat"
        )
        t2_scale = st.slider(
            "scaleFactor", 1.05, 2.0, 1.2, step=0.05, key="t2_scale"
        )
        
        # ── Matching Parameters ────────────────────────────────────
        st.write("---")
        st.caption("🎯 Tham số Matching:")
        t2_ratio = st.slider(
            "Ratio threshold (Lowe's)", 0.5, 0.95, 0.75, step=0.05, key="t2_ratio"
        )
        t2_min_matches = st.number_input(
            "Min good matches", min_value=4, max_value=100, value=10, key="t2_minmatch"
        )
        t2_min_inlier = st.slider(
            "Min inlier ratio", 0.1, 0.8, 0.25, step=0.05, key="t2_inlier"
        )
        
        btn_run_t2 = st.button("Tiến hành So Khớp Ảnh", type="primary", key="btn_t2")

    with t2_col_right:
        st.subheader("Kết quả Matching")
        if template_file and test_file:
            img_template = Image.open(template_file)
            img_test = Image.open(test_file)
            
            # Preview ảnh gốc
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.image(img_template, caption="Template gốc", width=200)
            with preview_col2:
                st.image(img_test, caption="Ảnh test gốc", width=200)
                
            st.write("---")
            
            if btn_run_t2:
                st.info("Đang chạy ORB Feature Matching + Preprocessing + RANSAC...")
                
                # Build preprocessing options từ UI
                pp_options = {
                    "resize": True,
                    "max_size": t2_resize_max,
                    "grayscale": True,
                    "clahe": t2_use_clahe,
                    "clahe_clip_limit": t2_clahe_clip,
                    "gaussian_blur": t2_use_gaussian,
                    "gaussian_ksize": t2_gaussian_k,
                    "median_blur": t2_use_median,
                    "median_ksize": t2_median_k,
                    "contrast_stretching": t2_use_contrast,
                    "sharpen": t2_use_sharpen,
                    "sharpen_strength": t2_sharpen_str,
                }
                
                result = match_orb(
                    img_template, img_test,
                    preprocessing_options=pp_options,
                    nfeatures=t2_nfeatures,
                    scaleFactor=t2_scale,
                    ratio_threshold=t2_ratio,
                    min_good_matches=t2_min_matches,
                    min_inlier_ratio=t2_min_inlier,
                )
                
                # ── Hiển thị ảnh trước/sau preprocessing ──────────
                pp_info = result.get("preprocessing_info", {})
                tpl_info = pp_info.get("template", {})
                qry_info = pp_info.get("query", {})
                
                if tpl_info.get("steps_applied") or qry_info.get("steps_applied"):
                    with st.expander("📊 Chi tiết Preprocessing", expanded=False):
                        pc1, pc2 = st.columns(2)
                        with pc1:
                            st.markdown("**Template:**")
                            tpl_steps = tpl_info.get("steps_applied", [])
                            st.markdown(" → ".join(tpl_steps) if tpl_steps else "Không có")
                        with pc2:
                            st.markdown("**Query:**")
                            qry_steps = qry_info.get("steps_applied", [])
                            st.markdown(" → ".join(qry_steps) if qry_steps else "Không có")
                
                # ── Kết quả chính ─────────────────────────────────
                if result["detected"]:
                    st.success(
                        f"✅ **DETECTED** — Tìm thấy đối tượng!  \n"
                        f"Confidence: **{result['confidence_score']:.1%}** | "
                        f"Good matches: **{result['good_matches']}** | "
                        f"Inliers: **{result['inlier_count']}** ({result['inlier_ratio']:.1%})"
                    )
                else:
                    st.error(
                        f"❌ **NOT DETECTED** — Không tìm thấy đối tượng.  \n"
                        f"Good matches: **{result['good_matches']}** "
                        f"(cần ≥ {t2_min_matches})"
                    )
                
                # ── Bảng metrics chi tiết ─────────────────────────
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("KP Template", result["keypoints_template"])
                mc2.metric("KP Query", result["keypoints_query"])
                mc3.metric("Good Matches", f"{result['good_matches']}/{result['total_matches']}")
                mc4.metric("Confidence", f"{result['confidence_score']:.1%}")
                
                mc5, mc6, mc7, mc8 = st.columns(4)
                mc5.metric("Inlier Count", result["inlier_count"])
                mc6.metric("Inlier Ratio", f"{result['inlier_ratio']:.1%}")
                mc7.metric("Runtime", f"{result['runtime_ms']:.1f} ms")
                mc8.metric("Kết luận", "✅ Có" if result["detected"] else "❌ Không")
                
                st.write("---")
                
                # ── Ảnh matching ──────────────────────────────────
                result_rgb = cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB)
                st.image(
                    result_rgb,
                    caption="Kết quả Feature Matching (ORB + Lowe's Ratio Test + RANSAC)",
                    use_container_width=True,
                )
                
                # ── Ảnh detected region ───────────────────────────
                if result["detected_region"] is not None:
                    region_rgb = cv2.cvtColor(result["detected_region"], cv2.COLOR_BGR2RGB)
                    st.image(
                        region_rgb,
                        caption="Vùng logo/nhãn phát hiện (Homography projection)",
                        use_container_width=True,
                    )
            else:
                st.warning("Nhấn nút 'Tiến hành So Khớp Ảnh' để chạy thuật toán.")
        else:
            st.info("Vui lòng upload ĐỦ cả ảnh Template và ảnh Test để thực hiện.")


# KHÔNG GIAN TAB 3 - EVALUATION (Đọc và phân tích file results.csv)
with tab3:
    st.header("📊 Kết quả Đánh giá Hiệu năng (Evaluation)")
    st.markdown("Đọc trực tiếp dữ liệu từ file thống kê hệ thống `results.csv`.")
    
    csv_path = "results.csv"
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            st.subheader("Chỉ số đo lường tổng quan")
            m1, m2, m3, m4 = st.columns(4)
            
            total_tested = len(df)
            avg_time = df['Processing_Time'].mean() if 'Processing_Time' in df.columns else 0.0
            
            m1.metric("Tổng số mẫu đã test", f"{total_tested} ảnh")
            m2.metric("Thời gian xử lý trung bình", f"{avg_time:.4f} s")
            
            if 'Prediction' in df.columns and 'Ground_Truth' in df.columns:
                correct = (df['Prediction'] == df['Ground_Truth']).sum()
                accuracy = (correct / total_tested) * 100
                m3.metric("Độ chính xác (Accuracy)", f"{accuracy:.1f}%")
            else:
                m3.metric("Độ chính xác (Accuracy)", "N/A")
                
            m4.metric("Trạng thái tệp dữ liệu", "Đã đồng bộ", delta="results.csv")
            
            st.write("---")
            st.subheader("Chi tiết kết quả theo cụm sản phẩm")
            
            if 'Logo_Group' in df.columns:
                logo_groups = ["Tất cả"] + list(df['Logo_Group'].unique())
                selected_group = st.selectbox("Lọc theo thư mục sản phẩm:", logo_groups)
                
                if selected_group != "Tất cả":
                    df_display = df[df['Logo_Group'] == selected_group]
                else:
                    df_display = df
            else:
                df_display = df
                st.caption("Gợi ý: Thêm cột `Logo_Group` vào tệp CSV để phân tách kết quả theo thư mục con.")
            
            st.dataframe(df_display, use_container_width=True)
            
            st.write("---")
            st.subheader("Biểu đồ Trực quan hóa")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                if 'Processing_Time' in df.columns:
                    st.markdown("**Thời gian xử lý của hệ thống qua từng ảnh (giây):**")
                    st.line_chart(df_display['Processing_Time'])
            with chart_col2:
                if 'Good_Matches' in df.columns:
                    st.markdown("**Số lượng điểm khớp hình học (Good Matches) tìm được:**")
                    st.bar_chart(df_display['Good_Matches'])
                    
        except Exception as e:
            st.error(f"Lỗi khi đọc hoặc xử lý tệp CSV: {e}")
            
    else:
        st.warning(f"⚠️ Không tìm thấy tệp `{csv_path}` tại thư mục hiện hành.")
        st.info("💡 Ứng dụng hiển thị Giao diện Mẫu (Mock Data). Biểu đồ sẽ tự động cập nhật dữ liệu thật khi có file csv.")
        
        mock_data = {
            "Logo_Group": ["Logo_01", "Logo_01", "Logo_01", "Logo_02", "Logo_02"],
            "Image_Name": ["pos_01.jpg", "pos_02.jpg", "neg_01.jpg", "pos_01.jpg", "neg_01.jpg"],
            "Algorithm": ["ORB", "ORB", "ORB", "ORB", "ORB"],
            "Good_Matches": [42, 55, 4, 38, 2],
            "Processing_Time": [0.042, 0.048, 0.035, 0.041, 0.039],
            "Prediction": ["Positive", "Positive", "Negative", "Positive", "Negative"],
            "Ground_Truth": ["Positive", "Positive", "Negative", "Positive", "Negative"]
        }
        df_mock = pd.DataFrame(mock_data)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Tổng số mẫu (Mock)", "5 ảnh")
        m2.metric("Thời gian xử lý TB (Mock)", "0.0410 s")
        m3.metric("Độ chính xác (Mock)", "100%")
        
        st.dataframe(df_mock, use_container_width=True)
        st.markdown("**Biểu đồ mẫu về số điểm khớp (Good Matches):**")
        st.bar_chart(df_mock['Good_Matches'])
