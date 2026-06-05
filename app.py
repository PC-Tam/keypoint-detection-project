import streamlit as st
import pandas as pd
from PIL import Image
import os

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
        
        st.caption("Cấu hình tham số (Dùng để map vào OpenCV sau này):")
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
            img_t1 = Image.open(uploaded_file_t1)
            out_col1, out_col2 = st.columns(2)
            
            with out_col1:
                st.image(img_t1, caption="Ảnh Gốc (Input)", use_container_width=True)
            with out_col2:
                if btn_run_t1:
                    st.info(f"Đang xử lý trích xuất bằng {algorithm}...")
                    st.image(img_t1, caption=f"Ảnh kết quả (Placeholder cho {algorithm})", use_container_width=True)
                    st.success("Xử lý hoàn tất!")
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
        
        st.write("---")
        st.caption("Cấu hình bộ lọc kết quả:")
        min_match_thresh = st.number_input("Số điểm khớp tối thiểu để nhận diện (Min Matches)", min_value=5, max_value=100, value=15)
        
        btn_run_t2 = st.button("Tiến hành So Khớp Ảnh", type="primary", key="btn_t2")

    with t2_col_right:
        st.subheader("Kết quả Matching")
        if template_file and test_file:
            img_template = Image.open(template_file)
            img_test = Image.open(test_file)
            
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.image(img_template, caption="Template đã chọn", width=180)
            with preview_col2:
                st.image(img_test, caption="Ảnh hiện trường đã chọn", width=180)
                
            st.write("---")
            
            if btn_run_t2:
                st.info("Đang chạy thuật toán ORB Feature Matching + RANSAC...")
                st.success(f"Kết quả (Giả lập): Tìm thấy đối tượng! (Số điểm khớp hình học > {min_match_thresh})")
                st.image(img_test, caption="Ảnh kết quả nối đặc trưng (Placeholder)", use_container_width=True)
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
