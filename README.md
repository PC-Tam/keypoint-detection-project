# 🔬 Blood Cell Classification using Keypoint Detection + Machine Learning

> **Xây dựng hệ thống tự động phân loại tế bào máu từ ảnh hiển vi bằng Harris Corner Detection, FAST và ORB kết hợp Machine Learning truyền thống.**

---

## 🎯 Mục tiêu

Phát triển pipeline phân loại tế bào máu (multi-class) từ ảnh hiển vi không sử dụng deep learning, thay vào đó dùng:

1. **Phát hiện điểm đặc trưng** (keypoint detection): Harris, FAST, ORB
2. **Trích xuất đặc trưng hình thái + màu sắc + texture truyền thống**
3. **Bag of Visual Words (BoVW)** với ORB descriptors
4. **Machine Learning**: SVM, Random Forest, KNN, Gaussian NB, Logistic Regression, XGBoost (optional)

---

## ⚠️ Ràng buộc quan trọng

| Điều **ĐƯỢC** dùng | Điều **KHÔNG ĐƯỢC** dùng |
|---|---|
| OpenCV, NumPy, scikit-learn | TensorFlow, PyTorch, Keras |
| Harris / FAST / ORB | CNN, YOLO, ResNet, ViT |
| SVM, RF, KNN, GNB, LR, XGBoost | Bất kỳ mô hình deep learning nào |
| Morphological features, LBP, Color Histogram | Transfer learning |

---

## 📦 Dataset

**Blood Cells Image Dataset** — Kaggle  
🔗 https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset

### Các lớp tế bào:

| Class | Description |
|---|---|
| `basophil` | Bạch cầu ưa kiềm |
| `eosinophil` | Bạch cầu ưa acid |
| `erythroblast` | Nguyên hồng cầu |
| `immature_granulocyte` | Bạch cầu hạt chưa trưởng thành |
| `lymphocyte` | Tế bào lympho |
| `monocyte` | Bạch cầu đơn nhân |
| `neutrophil` | Bạch cầu trung tính |
| `platelet` | Tiểu cầu |

---

## 🏗️ Cấu trúc Project

```
blood_cell_keypoint_ml/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/            ← Dataset gốc từ Kaggle
│   ├── processed/      ← metadata.csv, train/val/test.csv
│   └── features/       ← Feature bundles (.npy)
├── outputs/
│   ├── figures/        ← Confusion matrices, benchmark plots, visualizations
│   ├── models/         ← Trained models (.joblib)
│   ├── reports/        ← Classification reports, benchmark CSV
│   └── transformed_cases/  ← Transformation examples
├── src/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── preprocessing.py
│   ├── transformations.py
│   ├── segmentation_classical.py
│   ├── feature_harris.py
│   ├── feature_fast.py
│   ├── feature_orb.py
│   ├── feature_morphology.py
│   ├── feature_color_texture.py
│   ├── feature_engineering.py
│   ├── bovw.py
│   ├── feature_selection.py
│   ├── train_ml.py
│   ├── evaluate.py
│   ├── benchmark_keypoints.py
│   ├── visualize_keypoints.py
│   └── run_pipeline.py
└── app.py              ← Streamlit demo
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

> **XGBoost (optional):**
> ```bash
> pip install xgboost
> ```
> Pipeline tự động bỏ qua XGBoost nếu chưa được cài.

---

### 2. Tải Dataset

#### Cách 1: Kaggle API (tự động)

```bash
# Cấu hình kaggle.json trước (~/.kaggle/kaggle.json)
kaggle datasets download -d unclesamulus/blood-cells-image-dataset -p data/raw --unzip
```

#### Cách 2: Tải thủ công

1. Vào: https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset
2. Nhấn **Download**
3. Giải nén vào `data/raw/` (sao cho `data/raw/basophil/`, `data/raw/eosinophil/`, ... tồn tại)
4. Chạy lại pipeline

---

### 3. Chạy toàn bộ pipeline

```bash
python src/run_pipeline.py
```

Pipeline sẽ tự động:
1. Kiểm tra/tải dataset
2. Tạo metadata và chia train/val/test
3. Benchmark Harris/FAST/ORB (200 ảnh mẫu)
4. Visualize keypoints
5. Trích xuất tất cả features
6. Fit ORB BoVW vocabulary
7. Train tất cả ML models
8. Evaluate và lưu kết quả

---

### 4. Chạy demo Streamlit

```bash
streamlit run app.py
```

---

## 🧠 Giải thích thuật toán

### Harris Corner Detection

Harris phát hiện góc (corner) dựa trên ma trận gradient bậc hai (second-moment matrix):

```
M = Σ w(x,y) * [Ix²   IxIy]
                [IxIy  Iy²]

R = det(M) - k·trace²(M)
```

- R > 0: corner (góc)  
- R < 0: edge (cạnh)  
- R ≈ 0: flat region

**Tham số:**
- `block_size`: kích thước lân cận
- `ksize`: aperture Sobel operator
- `k`: Harris detector parameter (thường 0.04–0.06)

---

### FAST — Features from Accelerated Segment Test

FAST so sánh pixel trung tâm với 16 pixel trên đường tròn:

- Nếu ≥ N pixel liên tiếp sáng hơn (hoặc tối hơn) ngưỡng `t` → là keypoint
- Rất nhanh (≈ real-time), nhưng không có descriptor tích hợp
- Non-maximum suppression để giảm keypoint thừa

---

### ORB — Oriented FAST and Rotated BRIEF

ORB kết hợp:
- **Orientation FAST** để detect keypoints (với orientation tính từ intensity centroid)
- **Rotated BRIEF** để tạo binary descriptor 256-bit

ORB là rotation-invariant và scale-invariant (theo octave pyramid).  
Matching dùng **Hamming distance** — rất nhanh.

---

## 🔧 Feature Engineering

### 1. Keypoint Statistics (Harris / FAST / ORB)

Với mỗi thuật toán, tạo vector đặc trưng gồm:

| Feature | Mô tả |
|---|---|
| `n_keypoints` | Số điểm phát hiện được |
| `mean/max/std response` | Thống kê điểm mạnh |
| `keypoint_density` | Mật độ keypoint |
| `spatial histogram` | Phân bố theo lưới 4×4 |
| `response histogram` | Phân bố response (10 bins) |
| `n_in_mask / ratio_in_mask` | Keypoints trong vùng tế bào |
| `n_near_contour` | Keypoints gần biên tế bào |

Thêm với ORB: `mean/std angle`, `mean/std size`.

### 2. ORB Bag of Visual Words (BoVW)

```
ORB descriptors (N × 32) 
→ MiniBatchKMeans (100 clusters)  ← chỉ fit trên train
→ Histogram 100 chiều (L1 normalized)
```

> **Không bao giờ fit vocabulary trên val/test** để tránh data leakage.

### 3. Morphological Features (23 chiều)

Trích xuất từ mask phân đoạn tế bào:
- Area, perimeter, circularity, eccentricity
- Aspect ratio, extent, solidity
- Equivalent diameter
- **Hu moments** (7 values, log-transform)
- Centroid, mean/std distance contour → centroid

### 4. Color + Texture Features (88 chiều)

| Feature | Chiều |
|---|---|
| RGB mean/std | 6 |
| HSV mean/std | 6 |
| BGR color histogram | 48 |
| Gray histogram | 16 |
| LBP histogram (NumPy) | 10 |
| Edge density (Canny) | 1 |
| Mean intensity in mask | 1 |

### 5. Named Feature Bundles

| Bundle | Nội dung |
|---|---|
| `harris_only` | Harris stats |
| `fast_only` | FAST stats |
| `orb_stats_only` | ORB stats |
| `orb_bovw` | ORB BoVW 100-dim |
| `harris_morphology` | Harris + Morphology |
| `fast_morphology` | FAST + Morphology |
| `orb_bovw_morphology` | BoVW + Morphology |
| `combined_harris_fast_orb` | Harris + FAST + ORB stats |
| `combined_all_traditional` | Tất cả features |

---

## 📊 Chỉ số Benchmark Keypoint

### A. Number of Keypoints
Số điểm đặc trưng phát hiện được — phản ánh độ nhạy của detector.

### B. Runtime (ms)
Thời gian xử lý mỗi ảnh — quan trọng cho ứng dụng thực tế.

### C. Repeatability
Tỷ lệ keypoint từ ảnh gốc tìm lại được trong ảnh biến đổi:
```
Repeatability = matched_keypoints / original_keypoints
```
Dùng ngưỡng 5 pixel để tính match.

### D. Matching Rate
- **Harris/FAST**: Geometric matching rate (so khớp vị trí, không có descriptor)
- **ORB**: Descriptor matching rate (BFMatcher Hamming + ratio test 0.75)

### E. Registration Error
Sai số trung bình (pixels) khi ước lượng biến đổi hình học:
- Harris/FAST: từ các cặp điểm match theo vị trí
- ORB: từ `estimateAffinePartial2D` (RANSAC)

### F. Cell-region Keypoint Ratio
Tỷ lệ keypoint nằm trong vùng mask tế bào — đánh giá detector có tập trung vào tế bào hay bị nhiễu bởi nền.

---

## 📈 Chỉ số Classification

| Metric | Ý nghĩa |
|---|---|
| **Accuracy** | Tỷ lệ dự đoán đúng tổng thể |
| **Macro Precision** | Precision trung bình các lớp (equal weight) |
| **Macro Recall** | Recall trung bình các lớp |
| **Macro F1** | ⭐ Chỉ số chính — F1 trung bình (phù hợp cho multi-class, imbalanced) |
| **Weighted F1** | F1 trung bình theo tỷ lệ sample từng lớp |
| **Top-2 Accuracy** | Dự đoán đúng nếu lớp thật nằm trong top-2 dự đoán |
| **ROC-AUC (macro)** | One-vs-Rest ROC AUC trung bình |
| **Confusion Matrix** | Ma trận nhầm lẫn chi tiết |

---

## 📉 Mức kỳ vọng kết quả

> Pipeline truyền thống (không CNN) trên 8 lớp tế bào:

| Mức | Macro F1 | Đánh giá |
|---|---|---|
| Random baseline | ≈ 12.5% | Đoán ngẫu nhiên |
| Pipeline chạy đúng | — | Có visualization, benchmark, confusion matrix |
| Có tín hiệu học | > 50% | Model học được gì đó |
| Tốt cho traditional CV | > 60% | Kết quả ổn |
| Xuất sắc (traditional) | > 70% | Rất tốt cho non-DL |
| Deep learning thường đạt | > 85–95% | Ngoài phạm vi đề tài |

---

## 📂 Outputs

```
outputs/
├── reports/
│   ├── keypoint_benchmark.csv      ← Bảng benchmark Harris/FAST/ORB
│   ├── results.csv                 ← Kết quả tất cả models
│   ├── best_result.txt             ← Model tốt nhất
│   └── classification_report__*.txt
├── figures/
│   ├── keypoints_count_comparison.png
│   ├── runtime_comparison.png
│   ├── repeatability_comparison.png
│   ├── matching_rate_comparison.png
│   ├── registration_error_comparison.png
│   ├── cell_region_keypoint_ratio.png
│   ├── confusion_matrix__*.png
│   └── sample_visualizations/     ← Ảnh minh họa từng lớp
├── models/
│   ├── orb_bovw_kmeans.joblib
│   ├── best_model.joblib
│   └── <feature_set>__<model>.joblib
└── transformed_cases/              ← 5 case biến đổi ảnh
```

---

## 🔧 Cấu hình (config.yaml)

Chỉnh trong `config.yaml`:

```yaml
test_mode_limit: 300    # Giới hạn số ảnh (null = dùng toàn bộ)
orb_nfeatures: 500      # Số features ORB tối đa
bovw_clusters: 100      # Số visual words
feature_selection_k: 120  # Số features được chọn (SelectKBest)
image_size: [256, 256]  # Kích thước resize
```

---

## 📝 Ghi chú kỹ thuật

- Tất cả scaler/selector chỉ fit trên tập **train**, không bao giờ fit trên val/test.
- BoVW vocabulary chỉ được build từ tập **train ORB descriptors**.
- Pipeline hỗ trợ chế độ test nhanh qua `test_mode_limit`.
- Features lưu ra `.npy` để không phải extract lại.
- Dùng `class_weight="balanced"` cho SVM và Logistic Regression để xử lý lệch class.

---

## ⚕️ Tuyên bố từ chối trách nhiệm

> Project này là **demo học thuật**. Không sử dụng cho mục đích chẩn đoán y khoa thực tế.
