import time
import cv2
import numpy as np

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _ensure_numpy(image) -> np.ndarray:
    """Chuyển PIL Image → numpy array RGB nếu cần, giữ nguyên nếu đã là ndarray."""
    if _PIL_AVAILABLE and isinstance(image, PILImage.Image):
        return np.array(image.convert("RGB"))
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}. Expected numpy array or PIL Image.")
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh về grayscale, hỗ trợ RGB / BGR / grayscale."""
    if image is None:
        raise ValueError("Image is None.")
    if len(image.shape) == 2:
        return image.copy()
    if len(image.shape) == 3:
        channels = image.shape[2]
        if channels == 3:
            # Streamlit trả về RGB → chuyển về BGR rồi sang gray
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh về BGR (3 channels) để vẽ màu."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        # Giả định RGB từ Streamlit → BGR
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image.copy()


def detect_harris(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
) -> dict:
    """
    Phát hiện Harris Corner trên ảnh đầu vào.

    Parameters
    ----------
    image          : np.ndarray hoặc PIL.Image — ảnh đầu vào (RGB, BGR, grayscale, hoặc PIL)
    block_size     : int         — kích thước cửa sổ lân cận (mặc định 2)
    ksize          : int         — kích thước kernel Sobel (mặc định 3, phải lẻ)
    k              : float       — hệ số Harris (thường 0.04–0.06)
    threshold_ratio: float       — ngưỡng = ratio × max(response), mặc định 0.01

    Returns
    -------
    dict với các key:
        method          : str        — "Harris"
        keypoints_count : int        — số corner phát hiện được
        runtime_ms      : float      — thời gian xử lý (ms)
        image           : np.ndarray — ảnh BGR đã vẽ corners (chấm đỏ)
        corners         : np.ndarray — tọa độ (x, y) các corners, shape (N, 2)
        response_map    : np.ndarray — Harris response map (float32, normalized)
    """
    # 1. Tiền xử lý
    image = _ensure_numpy(image)   # PIL Image → numpy RGB nếu cần
    gray = _to_grayscale(image)
    gray_f32 = np.float32(gray)
    output_bgr = _to_bgr(image)

    # 2. Tính Harris response 
    t_start = time.perf_counter()

    response = cv2.cornerHarris(gray_f32, block_size, ksize, k)

    # Dilate để làm nổi bật corners (chuẩn OpenCV)
    response_dilated = cv2.dilate(response, None)

    # Ngưỡng
    threshold = threshold_ratio * response_dilated.max()
    corner_mask = response_dilated > threshold

    t_end = time.perf_counter()
    runtime_ms = (t_end - t_start) * 1000.0

    # 3. Lấy tọa độ corners
    ys, xs = np.where(corner_mask)
    corners = np.column_stack((xs, ys)) if len(xs) > 0 else np.empty((0, 2), dtype=int)

    # 4. Vẽ corners lên ảnh 
    output_bgr[corner_mask] = [0, 0, 255]  # BGR đỏ

    # 5. Normalize response map để hiển thị
    response_norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
    response_norm = np.uint8(response_norm)

    return {
        "method": "Harris",
        "keypoints_count": int(len(corners)),
        "runtime_ms": round(runtime_ms, 3),
        "image": output_bgr,          # BGR — dùng cv2.imshow hoặc st.image(..., channels="BGR")
        "corners": corners,
        "response_map": response_norm,
    }


#  Demo / test standalone 
if __name__ == "__main__":
    import sys

    # Dùng ảnh từ argument hoặc tạo ảnh test tổng hợp
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"[ERROR] Không đọc được ảnh: {sys.argv[1]}")
            sys.exit(1)
        # imread trả về BGR → chuyển sang RGB để giả lập Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Tạo ảnh test: nền xám + hình chữ nhật trắng (nhiều góc)
        img_rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        img_rgb[:] = 80
        cv2.rectangle(img_rgb, (50, 50), (180, 160), (220, 220, 220), -1)
        cv2.rectangle(img_rgb, (220, 80), (350, 200), (200, 200, 200), -1)
        cv2.rectangle(img_rgb, (100, 200), (300, 270), (180, 180, 180), -1)

    result = detect_harris(img_rgb, block_size=2, ksize=3, k=0.04, threshold_ratio=0.01)

    print("=" * 45)
    print(f"  Method         : {result['method']}")
    print(f"  Keypoints      : {result['keypoints_count']}")
    print(f"  Runtime        : {result['runtime_ms']} ms")
    print(f"  Output shape   : {result['image'].shape}")
    print("=" * 45)

    cv2.imwrite("harris_output.jpg", result["image"])
    cv2.imwrite("harris_response.jpg", result["response_map"])
    print("Saved: harris_output.jpg, harris_response.jpg")
