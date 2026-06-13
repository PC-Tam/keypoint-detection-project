# 🔬 Hệ Thống Phân Loại Tế Bào Máu (Classical Computer Vision + Machine Learning)

> **Dự án xây dựng hệ thống tự động phân loại tế bào máu từ ảnh hiển vi, sử dụng Giao diện Web (Streamlit) với khả năng "Plug and Play" (Tải về là chạy được ngay) mà không cần huấn luyện lại.**

---

## 🎯 Điểm Nổi Bật Của Dự Án

Dự án này tập trung vào việc giải quyết bài toán phân loại đa lớp (multi-class) trong Y Tế **bằng các phương pháp Thị giác máy tính truyền thống (Classical CV)**, hoàn toàn **KHÔNG sử dụng Deep Learning**.

1. **Giao diện Web Trực Quan:** Được xây dựng bằng Streamlit, cho phép người dùng tải ảnh lên, phân đoạn tế bào, trích xuất đặc trưng và dự đoán trực tiếp trên trình duyệt.
2. **Sẵn Sàng Sử Dụng (Pre-trained):** Kho lưu trữ đã bao gồm sẵn **10 mô hình tốt nhất** (SVM, Random Forest, Logistic Regression, KNN...) được huấn luyện trên 17,000 ảnh từ Kaggle. Bạn không cần tải bộ dữ liệu nặng nề hay train lại.
3. **Thuật Toán Sử Dụng:**
   - **Đặc trưng điểm (Keypoints):** Harris Corner, FAST, ORB kết hợp Bag of Visual Words (BoVW).
   - **Đặc trưng cơ bản:** Hình thái học (Morphology), Biểu đồ màu (Color Histogram), Cấu trúc bề mặt (Texture).

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh (Quick Start)

Do các mô hình trí tuệ nhân tạo đã được huấn luyện sẵn và tích hợp vào dự án, bạn chỉ mất chưa tới 2 phút để chạy thử hệ thống.

### Bước 1: Tải mã nguồn và Cài đặt thư viện

Mở Terminal / Command Prompt và chạy các lệnh sau:

```bash
# Clone dự án về máy
git clone https://github.com/PC-Tam/keypoint-detection-project.git
cd keypoint-detection-project

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Khởi chạy Ứng dụng Web

```bash
python -m streamlit run app.py
```

Trình duyệt sẽ tự động mở ra ở địa chỉ `http://localhost:8501`. 

### Bước 3: Trải nghiệm thử
Trong thư mục `sample_images/` của dự án đã có sẵn khoảng 16 tấm ảnh đại diện cho các loại tế bào máu khác nhau. 
Bạn hãy bấm nút **Browse files** trên web và chọn một tấm ảnh trong thư mục này để xem hệ thống cắt nền và dự đoán kết quả nhé!

---

## 🧠 Các Mô Hình Được Tích Hợp (Models)

Hệ thống cung cấp sẵn 10 mô hình Machine Learning kết hợp với các nhóm đặc trưng (Feature Bundles) khác nhau. Bạn có thể chọn đổi mô hình trực tiếp ở thanh công cụ bên trái (Sidebar) của Web App:

- `combined_all_traditional__SVM_RBF.joblib` (🌟 **Khuyên dùng** - Hiệu suất tốt nhất)
- `combined_all_traditional__RandomForest.joblib`
- `fast_morphology__RandomForest.joblib`
- `orb_bovw__RandomForest.joblib`
- ... cùng nhiều mô hình khác để đối chiếu so sánh.

---

## ⚠️ Ràng buộc cốt lõi của dự án

| Điều **ĐƯỢC** dùng | Điều **KHÔNG ĐƯỢC** dùng |
|---|---|
| OpenCV, NumPy, scikit-learn | TensorFlow, PyTorch, Keras |
| Harris / FAST / ORB | CNN, YOLO, ResNet, ViT |
| SVM, RF, KNN, GNB, LR | Bất kỳ mô hình deep learning nào |
| Morphological features, LBP, Color Histogram | Transfer learning |

---

## 🛠 Hướng dẫn Dành Cho Nhà Phát Triển (Tự Huấn Luyện Lại)

Nếu bạn muốn thay đổi mã nguồn, tự tải tập dữ liệu 17.000 ảnh và tự huấn luyện lại mô hình từ đầu, vui lòng làm theo các bước sau:

### 1. Tải Dataset (Từ Kaggle)

Bộ dữ liệu: [Blood Cells Image Dataset](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset)

**Cách 1: Dùng Kaggle API (Tự động)**
```bash
kaggle datasets download -d unclesamulus/blood-cells-image-dataset -p data/raw --unzip
```

**Cách 2: Tải thủ công**
Tải từ web Kaggle và giải nén sao cho cấu trúc thư mục là: `data/raw/basophil/`, `data/raw/eosinophil/`,...

### 2. Chạy Pipeline Đào tạo

Chạy tập lệnh sau để tự động trích xuất đặc trưng và train toàn bộ các thuật toán:

```bash
python src/run_pipeline.py
```

*Lưu ý: Quá trình này có thể tốn từ 1-2 tiếng tùy thuộc vào cấu hình máy tính của bạn.*

---

## ⚕️ Tuyên bố từ chối trách nhiệm

> Project này là **bài tập thực hành học thuật**. Hoàn toàn KHÔNG được sử dụng hệ thống này thay thế cho các thiết bị y tế hay dùng cho mục đích chẩn đoán y khoa thực tế trong bệnh viện.
