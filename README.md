<div align="center">

# Keypoint Detection Project

**Nghiên cứu, triển khai và so sánh các phương pháp phát hiện điểm đặc trưng ảnh**  
**theo 4 luồng tiếp cận: Hand-crafted → Real-time → ML Clustering → Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green.svg)](https://opencv.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-Latest-yellow.svg)](https://scikit-image.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-blueviolet.svg)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data--Analysis-lightgrey.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific--Computing-ff69b4.svg)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-Image--Processing-informational.svg)](https://python-pillow.org/)


</div>

---

## Giới thiệu | Introduction

**[VI]** Dự án nghiên cứu, triển khai và so sánh các phương pháp phát hiện điểm đặc trưng ảnh (Keypoint Detection) theo **4 luồng tiếp cận có hệ thống**: Hand-crafted Features (Harris, SIFT, LoG/DoH) → Real-time Features (FAST, ORB) → ML Clustering (Bag of Visual Words) → Deep Learning (CNN/MobileNetV2 + Grad-CAM). Kèm theo là ứng dụng web demo 3 tab cho phép so sánh trực tiếp kết quả của các phương pháp.

**[EN]** This project systematically studies, implements, and compares keypoint detection methods across **4 algorithmic streams**: Hand-crafted Features (Harris, SIFT, LoG/DoH) → Real-time Features (FAST, ORB) → ML Clustering (Bag of Visual Words) → Deep Learning (CNN/MobileNetV2 + Grad-CAM), with an interactive 3-tab Streamlit web application for live comparison.

---

##  Tính năng | Features

| Tính năng | Mô tả |
|:---|:---|
|  **4 luồng tiếp cận** | Từ Hand-crafted → Real-time → ML Clustering → Deep Learning — bám sát nội dung bài giảng |
|  **8 phương pháp** | Harris · SIFT · LoG/DoH · FAST · ORB · K-Means · BoVW · CNN/MobileNetV2 |
|  **So sánh định lượng** | Đo lường số keypoint, thời gian xử lý, phân bố không gian trên cùng bộ ảnh test |
|  **Web app 3 tab** | Detection · BoVW Image Matching · CNN Grad-CAM Visualization |
|  **Workflow chuẩn** | Toàn bộ sprint, issue, phân công quản lý trên GitHub Project Board |

---

##  Kiến trúc 4 luồng | 4-Stream Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KEYPOINT DETECTION PROJECT                   │
├──────────────────┬──────────────────┬──────────────┬───────────┤
│  LUỒNG 1         │  LUỒNG 2         │  LUỒNG 3     │  LUỒNG 4  │
│  Hand-crafted    │  Real-time       │  ML Cluster  │  Deep     │
│  Features        │  Features        │  (BoVW)      │  Learning │
├──────────────────┼──────────────────┼──────────────┼───────────┤
│  • Harris Corner │  • FAST          │  • K-Means   │  • CNN    │
│  • SIFT          │  • ORB           │  • BoVW      │  MobileV2 │
│  • LoG / DoH     │                  │  • K-NN      │  • Grad-  │
│                  │                  │              │    CAM    │
├──────────────────┼──────────────────┼──────────────┼───────────┤
│  Duy Quang       │  Chí Tâm         │  Trọng Nguyên│  Lan Thanh│
└──────────────────┴──────────────────┴──────────────┴───────────┘
         │                  │                │              │
         └──────────────────┴────────────────┴──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Streamlit Web App (3 tab)  │
                    └─────────────────────────────┘
```

---

##  Các phương pháp | Methods

### Luồng 1 — Hand-crafted Features *(Duy Quang)*

| # | Phương pháp | Vai trò | Thư viện | Tham số chính | Đặc điểm |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Harris Corner** | Phương pháp phụ / Nền tảng | `OpenCV` | `k=0.04` · `threshold=0.01` | Phát hiện góc — nền tảng lý thuyết cổ điển |
| 2 | **LoG / DoH** | Phương pháp phụ / Nền tảng | `scikit-image` | `max_sigma=30` | Blob detection — đạo hàm bậc 2, nền tảng cho DoG trong SIFT |
| 3 | **SIFT** |  Phương pháp chính | `OpenCV` | `nfeatures=500` | Bất biến tỉ lệ/góc xoay · Descriptor 128 chiều · Dùng DoG ≈ LoG |

> **Lưu ý:** Bằng sáng chế SIFT hết hạn tháng 3/2020 — dùng `opencv-python` bình thường, không cần `opencv-contrib`.

### Luồng 2 — Real-time Features *(Chí Tâm)*

| # | Phương pháp | Vai trò | Thư viện | Tham số chính | Đặc điểm |
|:---:|:---|:---:|:---:|:---:|:---|
| 4 | **FAST** | Phương pháp phụ | `OpenCV` | `threshold=10` · `nms=True` | Segment test 16 pixel · Tốc độ cao · Không có descriptor |
| 5 | **ORB** |  Phương pháp chính | `OpenCV` | `nfeatures=500` | FAST (detect) + BRIEF (describe) · Binary descriptor · Royalty-free |

### Luồng 3 — ML Clustering / BoVW *(Trọng Nguyên)*

| # | Phương pháp | Vai trò | Thư viện | Tham số chính | Đặc điểm |
|:---:|:---|:---:|:---:|:---:|:---|
| 6 | **K-Means** | Phương pháp phụ / Bổ trợ | `scikit-learn` | `n_clusters=50` | Unsupervised clustering · Tạo visual vocabulary |
| 7 | **K-NN** | Phương pháp phụ / Bổ trợ | `scikit-learn` | `n_neighbors=3` | Supervised · Image retrieval theo histogram similarity |
| 8 | **BoVW** |  Phương pháp chính | `scikit-learn` | `vocab_size=50` | SIFT descriptors → K-Means codebook → Histogram → K-NN matching |

### Luồng 4 — Deep Learning *(Lan Thanh)*

| # | Phương pháp | Vai trò | Thư viện | Tham số chính | Đặc điểm |
|:---:|:---|:---:|:---:|:---:|:---|
| 9 | **CNN / MobileNetV2** |  Phương pháp chính | `torchvision` | `pretrained=True` | Depthwise separable conv · Feature extractor 1280 chiều · Data-driven |
| 10 | **Grad-CAM** | Bổ trợ visualization | `torch` | — | Visualize vùng CNN tập trung ≈ "keypoint vùng" theo deep learning |

---

##  Ứng dụng Web | Web Application

### Tab 1 — Keypoint Detection

| Bước | Thao tác |
|:---:|:---|
| 1 | Upload ảnh bất kỳ (JPG, PNG — kể cả PNG có alpha channel) |
| 2 | Chọn một hoặc nhiều method: Harris · SIFT · ORB · FAST · LoG · DoH |
| 3 | Nhấn **"Phân tích ảnh"** — kết quả hiển thị song song theo grid |
| 4 | Xem bảng tổng hợp: method · số keypoint · thời gian xử lý (ms) |

### Tab 2 — BoVW Image Matching

| Bước | Thao tác |
|:---:|:---|
| 1 | Upload ảnh query |
| 2 | Nhấn **"Tìm ảnh tương tự"** |
| 3 | Xem top-3 ảnh tương tự nhất từ database kèm similarity score |
| 4 | Hiểu nguyên lý: SIFT descriptors → K-Means → Histogram → Cosine similarity |

### Tab 3 — CNN Feature Visualization

| Bước | Thao tác |
|:---:|:---|
| 1 | Upload ảnh bất kỳ |
| 2 | Nhấn **"Phân tích CNN"** |
| 3 | Xem song song: ảnh gốc \| Grad-CAM heatmap |
| 4 | Đọc thống kê: CNN feature dimension · thời gian xử lý · so sánh với SIFT |

---

##  Cấu trúc thư mục | Project Structure

```
keypoint-detection-project/
│
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── app/
│   ├── streamlit_app.py          # Web app chính (3 tab)
│   └── image_utils.py            # Tiền xử lý ảnh, xử lý edge cases
│
├── methods/
│   ├── base_detector.py          # Interface chuẩn cho toàn bộ methods
│   ├── harris.py                 # Harris Corner Detector          [Luồng 1]
│   ├── sift.py                   # SIFT Detector + match_sift()    [Luồng 1]
│   ├── blob_detector.py          # LoG + DoH Blob Detection        [Luồng 1]
│   ├── orb.py                    # ORB Detector                    [Luồng 2]
│   ├── fast.py                   # FAST Detector                   [Luồng 2]
│   ├── bovw.py                   # BoVW Pipeline (K-Means + K-NN)  [Luồng 3]
│   └── cnn_extractor.py          # MobileNetV2 + Grad-CAM          [Luồng 4]
│
├── models/
│   └── bovw_codebook.pkl         # K-Means codebook (tạo bằng bovw.py)
│
├── evaluation/
│   ├── compare.py                # Script đo lường 6 methods × 4 metrics
│   ├── visualize.py              # Figures so sánh side-by-side
│   ├── results.csv               # Kết quả đo lường thực tế
│   ├── charts/                   # Biểu đồ so sánh (PNG)
│   │   ├── keypoint_count.png
│   │   ├── processing_time.png
│   │   └── distribution.png
│   └── figures/                  # Figures visualize từng ảnh test
│       ├── compare_<image>.png   # 6 cột: ảnh gốc + 5 classical methods
│       ├── blob_vs_corner.png    # So sánh Harris vs LoG vs DoH
│       └── speed_ranking.png     # Xếp hạng tốc độ
│
├── notebooks/
│   ├── harris_demo.ipynb         # Demo Harris Corner
│   ├── sift_demo.ipynb           # Demo SIFT + Feature Matching
│   ├── blob_demo.ipynb           # Demo LoG / DoH
│   ├── orb_demo.ipynb            # Demo ORB
│   ├── fast_demo.ipynb           # Demo FAST (NMS on/off comparison)
│   ├── bovw_demo.ipynb           # Demo BoVW pipeline end-to-end
│   └── cnn_demo.ipynb            # Demo MobileNetV2 + Grad-CAM
│
├── data/
│   ├── general/                  # Ảnh chung — test Harris/SIFT/ORB/FAST/LoG/DoH
│   ├── blob/                     # Ảnh blob — test LoG/DoH (tế bào, bong bóng...)
│   ├── matching_pairs/           # Cặp ảnh — test BoVW matching
│   └── README.md                 # Nguồn gốc ảnh test
│
├── report/
│   ├── theory_draft.md           # Tóm tắt lý thuyết 4 luồng (Sprint 1)
│   ├── T13a_classical.md         # Lý thuyết Hand-crafted: Harris/SIFT/LoG/DoH
│   ├── T13b_cnn_evaluation.md    # Lý thuyết ORB/FAST/CNN + kết quả thực nghiệm
│   ├── T13c_bovw_webapp.md       # BoVW + mô tả web app 3 tab + kết luận
│   └── final_report.pdf          # Báo cáo hoàn chỉnh
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Cài đặt | Installation

> **Yêu cầu hệ thống:** Python 3.10+ · pip · Git

### Bước 1 — Clone repository

```bash
git clone https://github.com/Sunphuynx/keypoint-detection-project.git
cd keypoint-detection-project
```

### Bước 2 — Tạo môi trường ảo *(khuyến nghị)*

```bash
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Bước 3 — Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4 — Build BoVW Codebook *(chỉ cần làm 1 lần)*

```bash
python -c "from methods.bovw import build_vocabulary; build_vocabulary('data/general/')"
```

> File `models/bovw_codebook.pkl` sẽ được tạo tự động.  
> Cần có ảnh trong `data/general/` trước khi chạy bước này.

### Bước 5 — Chạy web app

```bash
streamlit run app/streamlit_app.py
```

Mở trình duyệt và truy cập: `http://localhost:8501`

---

##  Dependencies

| Package | Phiên bản | Mục đích |
|:---|:---:|:---|
| `opencv-python` | 4.9.0.80 | Xử lý ảnh · Harris · SIFT · ORB · FAST |
| `scikit-image` | 0.22.0 | Blob detection — LoG và DoH |
| `scikit-learn` | 1.4.0 | K-Means clustering · K-NN · BoVW pipeline |
| `torch` | 2.2.0 | PyTorch deep learning framework |
| `torchvision` | 0.17.0 | MobileNetV2 pretrained · Image transforms |
| `streamlit` | 1.33.0 | Web application framework |
| `numpy` | 1.26.4 | Tính toán ma trận và xử lý dữ liệu |
| `matplotlib` | 3.8.0 | Vẽ biểu đồ và visualization |
| `Pillow` | 10.2.0 | Đọc/ghi và convert định dạng ảnh |
| `pandas` | 2.2.0 | Lưu và phân tích kết quả đo lường |

---

##  Hướng dẫn sử dụng | Usage

### Luồng 1 & 2 — Classical + Real-time Methods

Tất cả các method tuân theo interface chuẩn từ `methods/base_detector.py`:

```python
from methods.harris import detect_harris
from methods.sift import detect_sift
from methods.blob_detector import detect_blobs
from methods.orb import detect_orb
from methods.fast import detect_fast

# Tất cả trả về cùng format:
# (keypoints_list, annotated_image_numpy, processing_time_ms)

keypoints, image, time_ms = detect_harris("data/general/general_01.jpg")
keypoints, image, time_ms = detect_sift("data/general/general_01.jpg")
keypoints, image, time_ms = detect_blobs("data/blob/blob_01.jpg", method="log")
keypoints, image, time_ms = detect_orb("data/general/general_01.jpg")
keypoints, image, time_ms = detect_fast("data/general/general_01.jpg", nms=True)
```

### Luồng 3 — BoVW Pipeline

```python
from methods.bovw import build_vocabulary, encode_image, find_similar_images
import pickle

# 1. Build vocabulary (chỉ làm 1 lần)
kmeans = build_vocabulary("data/general/", n_clusters=50)

# 2. Encode một ảnh thành histogram vector
histogram = encode_image("data/general/general_01.jpg", kmeans)
print(f"Histogram shape: {histogram.shape}")  # (50,)

# 3. Tìm ảnh tương tự
kmeans = pickle.load(open("models/bovw_codebook.pkl", "rb"))
results = find_similar_images(
    query_path="data/general/general_01.jpg",
    kmeans=kmeans,
    database_matrix=db_matrix,
    image_paths=image_paths,
    top_k=3
)
# results = [(similarity_score, image_path), ...]
```

### Luồng 4 — CNN + Grad-CAM

```python
from methods.cnn_extractor import extract_cnn_features, visualize_gradcam

# Trích xuất feature vector
feature_vector, time_ms = extract_cnn_features("data/general/general_01.jpg")
print(f"CNN feature dim: {feature_vector.shape}")  # (1280,)

# Visualize Grad-CAM
heatmap_image = visualize_gradcam("data/general/general_01.jpg")
# heatmap_image: BGR numpy array với heatmap overlay
```

### Chạy toàn bộ Evaluation

```bash
# Đo lường 6 methods → lưu results.csv
python evaluation/compare.py

# Tạo figures và biểu đồ → lưu charts/ và figures/
python evaluation/visualize.py
```

### Chạy Notebook Demo

```bash
jupyter notebook notebooks/
```

| Notebook | Nội dung |
|:---|:---|
| `harris_demo.ipynb` | Harris Corner trên ảnh đa dạng |
| `sift_demo.ipynb` | SIFT detect + Feature matching giữa 2 ảnh |
| `blob_demo.ipynb` | LoG vs DoH · So sánh Corner vs Blob |
| `orb_demo.ipynb` | ORB · So sánh binary vs float descriptor |
| `fast_demo.ipynb` | FAST · NMS=True vs NMS=False |
| `bovw_demo.ipynb` | BoVW pipeline end-to-end · K-Means · K-NN |
| `cnn_demo.ipynb` | MobileNetV2 features · Grad-CAM · So sánh CNN vs SIFT |

---

##  Kết quả | Results

> Xem báo cáo đầy đủ với số liệu thực nghiệm tại `report/final_report.pdf`

### Hướng dẫn chọn phương pháp theo usecase

| Usecase | Phương pháp đề xuất | Luồng | Lý do |
|:---|:---:|:---:|:---|
| Cần tốc độ tối đa, real-time | **FAST** | Real-time | Nhanh nhất — chỉ so sánh pixel trên vòng tròn |
| Feature matching, object retrieval (classical) | **SIFT** | Hand-crafted | Descriptor 128D mạnh, bất biến tỉ lệ/góc xoay |
| Nhẹ, không GPU, royalty-free | **ORB** | Real-time | FAST + BRIEF, binary descriptor, matching bằng Hamming |
| Phát hiện vùng đặc trưng (blob, tế bào) | **LoG / DoH** | Hand-crafted | Đạo hàm bậc 2 — phát hiện vùng, không phải góc |
| Image retrieval từ database | **BoVW** | ML Clustering | Encode ảnh thành histogram, scalable, không cần GPU |
| Hiểu ngữ nghĩa ảnh, data-driven | **CNN + Grad-CAM** | Deep Learning | Feature 1280D biểu diễn ngữ nghĩa cao, visualize được |
| Nền tảng lý thuyết, giảng dạy | **Harris** | Hand-crafted | Công thức tường minh, dễ hiểu, dễ phân tích |

---

##  Thành viên | Team

| Thành viên | Vai trò | Luồng phụ trách | Trách nhiệm chính |
|:---|:---:|:---:|:---|
| **Chí Tâm** | Tech Lead · Algorithm Engineer | Luồng 2 | Setup repo · `base_detector.py` · FAST · ORB · Tích hợp web app · Release |
| **Duy Quang** | Hand-crafted Features Engineer | Luồng 1 | Harris · SIFT · LoG/DoH · Visualization · Báo cáo T13a |
| **Lan Thanh** | Deep Learning Engineer · Evaluation Lead | Luồng 4 | CNN/MobileNetV2 · Grad-CAM · Evaluation metrics · Báo cáo T13b |
| **Trọng Nguyên** | ML Clustering Engineer · Web App · Report Lead | Luồng 3 | BoVW pipeline · Streamlit app 3 tab · Tổng hợp báo cáo T13c+d |

---

##  GitHub Project

Toàn bộ tiến độ, phân công và sprint được quản lý tại GitHub Project Board:

 **[Xem Project Board](https://github.com/Sunphuynx/keypoint-detection-project/projects)**

### Sprint Overview

| Sprint | Thời gian | Mục tiêu |
|:---:|:---|:---|
| Sprint 1 | 29/04 – 05/05 | Setup + Harris + Dataset |
| Sprint 2 | 06/05 – 12/05 | SIFT · LoG/DoH · ORB · FAST · CNN · BoVW |
| Sprint 3 | 13/05 – 19/05 | Evaluation + Web App 3 tab |
| Sprint 4 | 20/05 – 26/05 | Báo cáo + Cleanup + Release v1.0.0 |

### Quy ước đặt tên branch

```
feature/harris-implementation      ← tính năng mới
feature/bovw-pipeline              ← tính năng mới
fix/sift-scale-space-error         ← bug fix
docs/update-readme                 ← tài liệu
eval/comparison-metrics            ← evaluation
app/streamlit-3-tabs               ← web app
```

### Quy ước commit message

```
feat: add harris corner detector module
feat: implement bovw pipeline with kmeans clustering
feat: add mobilenetv2 gradcam visualization
fix: resolve orb binary descriptor matching issue
docs: update README with 4-stream architecture
eval: add processing time comparison chart
app: integrate bovw matching into tab 2
```

---

##  Tài liệu tham khảo chính | Key References

| Tác giả | Tên công trình | Năm | Liên quan |
|:---|:---|:---:|:---|
| Lowe, D.G. | "Object recognition from local scale-invariant features" | 1999 | SIFT |
| Lowe, D.G. | "Distinctive image features from scale-invariant keypoints" | 2004 | SIFT |
| Rublee et al. | "ORB: An efficient alternative to SIFT or SURF" | 2011 | ORB |
| Rosten & Drummond | "Machine learning for high-speed corner detection" | 2006 | FAST |
| Sivic & Zisserman | "Video Google: A text retrieval approach to object matching" | 2003 | BoVW |
| Sandler et al. | "MobileNetV2: Inverted Residuals and Linear Bottlenecks" | 2018 | CNN |
| Selvaraju et al. | "Grad-CAM: Visual explanations from deep networks" | 2017 | Grad-CAM |

---

##  License

Dự án được phân phối theo giấy phép **MIT License** — xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

<div align="center">
  <b>Keypoint Detection Project</b><br>
  <sub>
    Hand-crafted · Real-time · ML Clustering · Deep Learning<br>
    Thực hiện bởi nhóm sinh viên · Trường Đại học Kinh tế TP.HCM (UEH)<br>
    Môn học: Xử lý và Phân tích Hình ảnh · Năm học 2025 – 2026
  </sub>
</div>
