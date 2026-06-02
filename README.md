<div align="center">

# Keypoint Detection Project

**Ứng dụng Keypoint Detection trong nhận diện nhãn sản phẩm/logo**  
**với 3 phương pháp chính: Harris Corner Detection · FAST · ORB**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-blueviolet.svg)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data--Analysis-lightgrey.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific--Computing-ff69b4.svg)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-Image--Processing-informational.svg)](https://python-pillow.org/)

</div>

---

## 1. Giới thiệu | Introduction

Dự án tập trung nghiên cứu, triển khai và so sánh các phương pháp phát hiện điểm đặc trưng ảnh (*Keypoint Detection*) trong xử lý ảnh truyền thống.

Dự án sử dụng **3 phương pháp chính**:

1. **Harris Corner Detection**
2. **FAST — Features from Accelerated Segment Test**
3. **ORB — Oriented FAST and Rotated BRIEF**

Trong đó, Harris và FAST được dùng để phát hiện và so sánh các điểm đặc trưng trên ảnh. ORB được dùng cho cả phát hiện keypoints, trích xuất descriptors và xây dựng demo nhận diện nhãn sản phẩm/logo thông qua feature matching.

---

## 2. Bài toán ứng dụng | Application Problem

Bài toán cụ thể của dự án:

> **Nhận diện sự xuất hiện của một nhãn sản phẩm hoặc logo đã biết trong ảnh đầu vào bằng Keypoint Detection và Feature Matching.**

Cách hoạt động tổng quát:

1. Người dùng upload một ảnh mẫu chứa nhãn sản phẩm/logo cần nhận diện.
2. Người dùng upload một ảnh kiểm thử có thể chứa hoặc không chứa nhãn/logo đó.
3. Hệ thống dùng ORB để phát hiện keypoints và trích xuất descriptors trên cả hai ảnh.
4. Hệ thống so khớp descriptors bằng Brute-Force Matcher với Hamming Distance.
5. Nếu số lượng good matches vượt ngưỡng, hệ thống kết luận nhãn/logo có khả năng xuất hiện trong ảnh kiểm thử.
6. Nếu đủ điều kiện, hệ thống sử dụng Homography + RANSAC để vẽ vùng phát hiện trên ảnh kiểm thử.

---

## 3. Phạm vi dự án | Scope

| Nội dung | Vai trò |
|:---|:---|
| Harris Corner Detection | Phương pháp chính thứ nhất, baseline cổ điển |
| FAST | Phương pháp chính thứ hai, nhấn mạnh tốc độ |
| ORB | Phương pháp chính thứ ba, dùng cho nhận diện bằng matching |
| OpenCV | Xử lý ảnh, keypoint detection, feature matching |
| Streamlit | Web app demo chạy local |
| Matplotlib / Pandas / NumPy | Đánh giá, bảng kết quả, biểu đồ |
| GitHub | Quản lý mã nguồn, phân công, release |


---

## 4. Kiến trúc phương pháp | Method Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                KEYPOINT DETECTION PROJECT                    │
├────────────────────┬────────────────────┬────────────────────┤
│ Method 1           │ Method 2           │ Method 3           │
│ Harris Corner      │ FAST               │ ORB                │
├────────────────────┼────────────────────┼────────────────────┤
│ Classical corner   │ Real-time keypoint │ Keypoint +         │
│ detection          │ detection          │ descriptor         │
├────────────────────┼────────────────────┼────────────────────┤
│ Output:            │ Output:            │ Output:            │
│ keypoints + time   │ keypoints + time   │ keypoints,         │
│                    │                    │ descriptors,       │
│                    │                    │ matching result    │
└────────────────────┴────────────────────┴────────────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │ Streamlit Web App           │
                │ Tab 1: Detection            │
                │ Tab 2: ORB Matching         │
                │ Tab 3: Evaluation           │
                └────────────────────────────┘
```

---

## 5. Các phương pháp chính | Main Methods

### 5.1. Harris Corner Detection

Harris Corner Detection là phương pháp phát hiện góc cổ điển. Phương pháp này dựa trên sự thay đổi cường độ sáng của vùng ảnh khi dịch chuyển theo nhiều hướng khác nhau. Những vị trí có sự thay đổi mạnh theo cả hai hướng thường được xem là điểm góc hoặc điểm đặc trưng.

**Vai trò trong dự án:**

- Là phương pháp baseline truyền thống.
- Dùng để minh họa nguyên lý phát hiện điểm đặc trưng cổ điển.
- Dùng để so sánh với FAST và ORB về số lượng keypoints, thời gian xử lý và chất lượng trực quan.

**Kết quả đầu ra:**

- Ảnh có đánh dấu các điểm Harris.
- Số lượng điểm phát hiện được.
- Thời gian xử lý.

---

### 5.2. FAST — Features from Accelerated Segment Test

FAST là phương pháp phát hiện keypoints tốc độ cao. Phương pháp này kiểm tra một vòng tròn các pixel xung quanh điểm trung tâm để xác định xem điểm đó có phải là điểm đặc trưng hay không.

**Vai trò trong dự án:**

- Là phương pháp chính thứ hai.
- Đại diện cho nhóm real-time keypoint detection.
- Dùng để so sánh tốc độ với Harris và ORB.

**Kết quả đầu ra:**

- Ảnh có đánh dấu keypoints FAST.
- Số lượng keypoints.
- Thời gian xử lý.

---

### 5.3. ORB — Oriented FAST and Rotated BRIEF

ORB là phương pháp kết hợp FAST detector và BRIEF descriptor đã cải tiến. ORB vừa có thể phát hiện keypoints, vừa có thể tạo descriptors dạng nhị phân để phục vụ so khớp đặc trưng.

**Vai trò trong dự án:**

- Là phương pháp chính thứ ba.
- Là phương pháp quan trọng nhất trong phần demo nhận diện nhãn sản phẩm/logo.
- Dùng để phát hiện keypoints, trích xuất descriptors và so khớp đặc trưng giữa ảnh mẫu và ảnh kiểm thử.

**Kết quả đầu ra:**

- Ảnh có keypoints ORB.
- Số lượng keypoints.
- Thời gian xử lý.
- Ảnh matching giữa ảnh mẫu và ảnh kiểm thử.
- Số lượng good matches.
- Kết luận `Detected` hoặc `Not detected`.

---

## 6. Phương pháp phụ trợ | Supporting Techniques

Các kỹ thuật sau chỉ đóng vai trò hỗ trợ trong pipeline nhận diện, **không được tính là phương pháp chính**:

| Kỹ thuật | Vai trò |
|:---|:---|
| Brute-Force Matcher | So khớp ORB descriptors giữa hai ảnh |
| Hamming Distance | Đo khoảng cách giữa các binary descriptors |
| Ratio Test hoặc Distance Threshold | Lọc các matches kém |
| RANSAC | Loại bỏ matches sai khi tính Homography |
| Homography | Ước lượng vùng chứa logo/nhãn trong ảnh kiểm thử |
| Visualization | Vẽ keypoints, matches, bounding box, bảng và biểu đồ |

---

## 7. Tính năng | Features

| Tính năng | Mô tả |
|:---|:---|
| 3 phương pháp chính | Harris · FAST · ORB |
| Web app chạy local | Giao diện Streamlit đơn giản, không cần GPU |
| Keypoint visualization | Vẽ trực tiếp keypoints lên ảnh đầu vào |
| ORB feature matching | So khớp ảnh mẫu và ảnh kiểm thử |
| Product/logo detection demo | Nhận diện nhãn sản phẩm/logo đã biết |
| Evaluation table | Lưu kết quả thực nghiệm vào `results.csv` |
| Charts | So sánh số keypoints, runtime và kết quả matching |
| No Deep Learning | Không dùng CNN, MobileNetV2, Grad-CAM, PyTorch |

---

## 8. Ứng dụng Web | Web Application

Web app gồm 3 tab chính.

### Tab 1 — Keypoint Detection

| Bước | Thao tác |
|:---:|:---|
| 1 | Upload một ảnh đầu vào |
| 2 | Chọn phương pháp: Harris, FAST hoặc ORB |
| 3 | Nhấn nút phân tích |
| 4 | Xem ảnh có keypoints, số lượng keypoints và thời gian xử lý |

### Tab 2 — Logo/Product Matching

| Bước | Thao tác |
|:---:|:---|
| 1 | Upload ảnh mẫu chứa logo/nhãn sản phẩm |
| 2 | Upload ảnh kiểm thử |
| 3 | Chạy ORB Matching |
| 4 | Xem ảnh matches, số good matches và kết luận detected/not detected |
| 5 | Nếu đủ matches, xem vùng phát hiện được vẽ trên ảnh kiểm thử |

### Tab 3 — Evaluation

| Bước | Thao tác |
|:---:|:---|
| 1 | Đọc dữ liệu từ `results/results.csv` |
| 2 | Hiển thị bảng kết quả thực nghiệm |
| 3 | Vẽ biểu đồ so sánh số keypoints |
| 4 | Vẽ biểu đồ so sánh thời gian xử lý |
| 5 | Hiển thị nhận xét ngắn về ưu/nhược điểm của từng phương pháp |

---

## 9. Cấu trúc thư mục | Project Structure

```text
keypoint-detection-project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── harris_detector.py        # Harris Corner Detection
│   ├── fast_detector.py          # FAST Detection
│   ├── orb_detector.py           # ORB Detection + Descriptor
│   ├── matcher.py                # ORB Matching + Homography
│   ├── evaluation.py             # Run evaluation and export CSV
│   └── utils.py                  # Image loading, conversion, visualization
│
├── data/
│   ├── templates/                # Ảnh mẫu logo/nhãn sản phẩm
│   └── test_images/              # Ảnh kiểm thử positive/negative
│
├── results/
│   ├── results.csv               # Bảng kết quả thực nghiệm
│   ├── charts/                   # Biểu đồ đánh giá
│   └── figures/                  # Ảnh output minh họa
│
├── notebooks/
│   ├── 01_harris_demo.ipynb
│   ├── 02_fast_demo.ipynb
│   └── 03_orb_matching_demo.ipynb
│
└── report/
    ├── report_draft.md
    └── final_report.pdf
```

---

## 10. Cài đặt | Installation

> Yêu cầu hệ thống: Python 3.10+, pip, Git.

### Bước 1 — Clone repository

```bash
git clone https://github.com/Sunphuynx/keypoint-detection-project.git
cd keypoint-detection-project
```

### Bước 2 — Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường ảo:

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

### Bước 4 — Chạy web app

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại:

```text
http://localhost:8501
```

---

## 11. Kết quả kỳ vọng | Expected Results

| Phương pháp | Kết quả kỳ vọng |
|:---|:---|
| Harris | Phát hiện tốt các điểm góc rõ ràng, dễ giải thích về mặt lý thuyết |
| FAST | Chạy nhanh, phát hiện nhiều keypoints, phù hợp so sánh thời gian xử lý |
| ORB | Có thể phát hiện keypoints, tạo descriptors và matching giữa ảnh mẫu với ảnh kiểm thử |
| ORB Matching | Có thể nhận diện nhãn/logo trong điều kiện ảnh đủ rõ, có nhiều chi tiết và ít bị che khuất |

---

## 12. Hạn chế | Limitations

Hệ thống chỉ nhận diện một đối tượng đã biết bằng cách so khớp điểm đặc trưng giữa ảnh mẫu và ảnh kiểm thử.

Các trường hợp có thể làm kết quả kém:

- Logo/nhãn quá ít chi tiết hoặc quá trơn.
- Ảnh bị mờ mạnh, thiếu sáng hoặc lóa sáng.
- Góc chụp quá nghiêng.
- Vật thể bị che khuất nhiều.
- Ảnh mẫu và ảnh kiểm thử khác nhau quá lớn về tỉ lệ, ánh sáng hoặc chất lượng.

---

## 13. Phân công thành viên | Team Assignment

| Thành viên | Vai trò mới | Trách nhiệm chính |
|:---|:---|:---|
| Chí Tâm | ORB Matching · Integration | Thiết kế cấu trúc repo, thống nhất output format, implement ORB, implement feature matching, tích hợp Streamlit, cleanup code, README |
| Duy Quang | Harris + FAST Engineer | Implement Harris, implement FAST, tạo notebook demo, xuất hình minh họa keypoints, viết phần lý thuyết Harris và FAST |
| Trọng Nguyên | Dataset · Streamlit · Visualization | Chuẩn bị ảnh template/test, xây dựng web app, tạo tab Evaluation, vẽ biểu đồ, hỗ trợ tổng hợp kết quả |
| Lan Thanh | Evaluation · Report Lead | Thiết kế tiêu chí đánh giá, chạy test, tạo `results.csv`, viết phần evaluation, hạn chế, kết luận và tổng hợp báo cáo |

---

## 14. Tài liệu tham khảo chính | Key References

| Tác giả | Tên công trình | Năm | Liên quan |
|:---|:---|:---:|:---|
| Harris, C. & Stephens, M. | A Combined Corner and Edge Detector | 1988 | Harris Corner Detection |
| Rosten, E. & Drummond, T. | Machine Learning for High-Speed Corner Detection | 2006 | FAST |
| Rublee, E., Rabaud, V., Konolige, K. & Bradski, G. | ORB: An efficient alternative to SIFT or SURF | 2011 | ORB |
| Bradski, G. & Kaehler, A. | Learning OpenCV 3 | 2016 | OpenCV, feature detection |
| Gonzalez, R. C. & Woods, R. E. | Digital Image Processing | 2018 | Xử lý ảnh số |

---

<div align="center">
  <b>Keypoint Detection Project</b><br>
  <sub>
    Harris Corner Detection · FAST · ORB<br>
    Ứng dụng nhận diện nhãn sản phẩm/logo trong bài toán keypoint detection<br>
    Môn học: Xử lý và Phân tích Hình ảnh
  </sub>
</div>
