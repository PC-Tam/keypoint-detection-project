# Preprocessing Pipeline — Ghi chú kỹ thuật

## 1. Vì sao cần tiền xử lý?

Pipeline cũ **không có tiền xử lý** (chỉ chuyển grayscale) nên kết quả ORB matching rất tệ khi:

| Điều kiện | Vấn đề |
|:---|:---|
| Ảnh thiếu sáng / tối | ORB không phát hiện đủ keypoints |
| Ảnh tương phản thấp | Descriptors không đủ phân biệt |
| Ảnh quá lớn (4000×3000 px) | Chạy chậm, keypoints bị phân tán |
| Ảnh nhiễu hạt | FAST (bên trong ORB) detect nhiều false corners |
| Ảnh mờ | Keypoints yếu, descriptors không ổn định |

## 2. Pipeline mới

```text
Input (RGB) 
  │
  ├── 1. Resize giữ tỉ lệ    [MẶC ĐỊNH — max_size=1024]
  │     Giới hạn kích thước, tăng tốc xử lý
  │
  ├── 2. Grayscale             [MẶC ĐỊNH]
  │     Detector/descriptor cần ảnh 1 kênh
  │
  ├── 3. CLAHE                 [MẶC ĐỊNH — clipLimit=2.0, tile=8×8]
  │     Cải thiện tương phản cục bộ, hiệu quả với ảnh tối/sáng không đều
  │
  ├── 4. Contrast Stretching   [TÙY CHỌN]
  │     Kéo giãn histogram khi ảnh có dynamic range hẹp
  │
  ├── 5. Gaussian Blur         [TÙY CHỌN — ksize=5]
  │     Làm mịn nhẹ, giảm nhiễu Gaussian
  │
  ├── 6. Median Blur           [TÙY CHỌN — ksize=5]
  │     Giảm nhiễu muối tiêu (salt-and-pepper)
  │
  └── 7. Sharpen               [TÙY CHỌN — strength=0.5]
        Làm rõ cạnh bằng unsharp masking
  │
  ▼
Output (Grayscale uint8) → Detector / Matcher
```

## 3. Bước mặc định vs tùy chọn

| Bước | Mặc định? | Lý do |
|:---|:---:|:---|
| Resize | ✅ | Ảnh lớn chạy chậm, keypoints phân tán |
| Grayscale | ✅ | OpenCV detector cần grayscale |
| CLAHE | ✅ | Cải thiện rõ rệt cho ảnh thiếu sáng/tương phản thấp mà ít gây hại |
| Contrast Stretching | ❌ | Có thể thay đổi phân bố pixel không mong muốn |
| Gaussian Blur | ❌ | Có thể làm mờ chi tiết, ảnh hưởng descriptor |
| Median Blur | ❌ | Chỉ cần khi ảnh có nhiễu muối tiêu |
| Sharpen | ❌ | Có thể khuếch đại nhiễu |

## 4. Vì sao KHÔNG dùng Thresholding và Global Histogram Equalization?

### Thresholding (Otsu, Adaptive, v.v.)
- **Phá hủy texture**: ORB descriptor dựa trên so sánh cường độ pixel trong patch. Thresholding biến ảnh thành binary (0/255), làm mất hoàn toàn thông tin gradient.
- **Giảm keypoints**: Vùng ảnh trở nên phẳng sau threshold → ít corner/edge.
- **Chỉ phù hợp cho OCR hoặc segmentation**, không phải feature matching.

### Global Histogram Equalization (`cv2.equalizeHist`)
- **Khuếch đại nhiễu**: Equalize toàn cục tăng cường vùng tối (thường chứa nhiễu) lên mức quá mạnh.
- **Không adaptive**: Không phân biệt vùng sáng/tối cục bộ.
- **CLAHE tốt hơn**: CLAHE là phiên bản cục bộ của histogram equalization, giới hạn amplification bằng clipLimit.

## 5. Vai trò Harris / FAST / ORB sau cập nhật

| Phương pháp | Vai trò | Preprocessing |
|:---|:---|:---|
| **Harris Corner Detection** | Detector, so sánh keypoints/runtime | Tab 1: auto_preprocess() hoặc thủ công |
| **FAST** | Detector, so sánh keypoints/runtime | Tab 1: auto_preprocess() hoặc thủ công |
| **ORB** (detection) | Detector + descriptor, hiển thị keypoints | Tab 1: auto_preprocess() hoặc thủ công |
| **ORB** (matching) | **Phương pháp chính** cho logo matching | Tab 2: preprocess_image() với tùy chọn từ UI |

**Ghi chú**: Tab 1 dùng `auto_preprocess()` (tự động phân tích ảnh rồi chọn bước xử lý phù hợp). Tab 2 dùng `preprocess_image()` (người dùng chọn từ giao diện).

## 6. Thông số mặc định khuyến nghị

### Preprocessing
```python
DEFAULT_PREPROCESS_OPTIONS = {
    "resize": True,       "max_size": 1024,
    "grayscale": True,
    "clahe": True,        "clahe_clip_limit": 2.0, "clahe_tile_grid_size": 8,
    "gaussian_blur": False, "gaussian_ksize": 5,
    "median_blur": False,   "median_ksize": 5,
    "contrast_stretching": False,
    "sharpen": False,       "sharpen_strength": 0.5,
}
```

### ORB
```python
nfeatures = 1500       # Nhiều hơn default 500 → matching tốt hơn
scaleFactor = 1.2      # Mặc định ORB
nlevels = 8            # Mặc định ORB  
edgeThreshold = 31     # Mặc định ORB
patchSize = 31         # Mặc định ORB
fastThreshold = 20     # Mặc định ORB
```

### Matching
```python
ratio_threshold = 0.75          # Lowe's ratio test (0.7-0.8 phổ biến)
min_good_matches = 10           # Ngưỡng tối thiểu
ransac_reproj_threshold = 5.0   # Mặc định OpenCV
min_inlier_ratio = 0.25         # 25% inliers
```

## 7. ORB Matching Pipeline (sau cập nhật)

```text
Template image + Query image
  │
  ├── preprocess_image() cho cả 2 ảnh
  │     (resize → grayscale → CLAHE → [tùy chọn])
  │
  ├── ORB detectAndCompute trên mỗi ảnh
  │     ↳ Nếu descriptors = None → return Not Detected
  │
  ├── BFMatcher (NORM_HAMMING, crossCheck=False)
  │
  ├── KNN matching (k=2)
  │
  ├── Lowe's ratio test (d1 < ratio × d2)
  │     ↳ Nếu good_matches < min_good_matches → return Not Detected
  │
  ├── Homography + RANSAC
  │     ↳ Nếu H = None hoặc det(H) bất thường → return Not Detected
  │
  ├── Tính inlier ratio = inliers / good_matches
  │     ↳ Nếu inlier_ratio < min_inlier_ratio → return Not Detected
  │
  ├── Tính confidence = (good_matches / min(kp1, kp2)) × inlier_ratio
  │
  └── Vẽ bounding box + return kết quả
```
