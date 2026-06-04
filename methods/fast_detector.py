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
            # Streamlit trả về RGB → chuyển sang gray
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh về BGR (3 channels) để vẽ màu."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image.copy()


def detect_fast(
    image: np.ndarray,
    threshold: int = 10,
    nonmax_suppression: bool = True,
    type: int = cv2.FastFeatureDetector_TYPE_9_16,
) -> dict:
    """
    Phát hiện keypoints bằng thuật toán FAST.

    Parameters
    ----------
    image               : np.ndarray — ảnh đầu vào (RGB, BGR, hoặc grayscale)
    threshold           : int        — ngưỡng cường độ pixel (mặc định 10)
                                       Thấp → nhiều keypoints, cao → ít nhưng mạnh hơn
    nonmax_suppression  : bool       — bật Non-Maximum Suppression (mặc định True)
    type                : int        — loại FAST:
                                         TYPE_9_16  (mặc định, phổ biến nhất)
                                         TYPE_7_12
                                         TYPE_5_8

    Returns
    -------
    dict với các key:
        method              : str           — "FAST"
        keypoints_count     : int           — số keypoints phát hiện được
        runtime_ms          : float         — thời gian xử lý (ms)
        image               : np.ndarray   — ảnh BGR đã vẽ keypoints (chấm xanh lá)
        keypoints           : list[cv2.KeyPoint] — danh sách KeyPoint object
        keypoints_xy        : np.ndarray   — tọa độ (x, y) dạng array, shape (N, 2)
        params              : dict          — tham số đã dùng
    """
    # 1. Tiền xử lý 
    image = _ensure_numpy(image)   # PIL Image → numpy RGB nếu cần
    gray = _to_grayscale(image)
    output_bgr = _to_bgr(image)

    # 2. Khởi tạo FAST detector 
    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=nonmax_suppression,
        type=type,
    )

    # 3. Detect keypoints 
    t_start = time.perf_counter()
    keypoints = fast.detect(gray, None)
    t_end = time.perf_counter()

    runtime_ms = (t_end - t_start) * 1000.0

    #  4. Lấy tọa độ dạng array 
    if keypoints:
        keypoints_xy = np.array(
            [[int(kp.pt[0]), int(kp.pt[1])] for kp in keypoints], dtype=np.int32
        )
    else:
        keypoints_xy = np.empty((0, 2), dtype=np.int32)

    # 5. Vẽ keypoints lên ảnh 
    # Dùng drawKeypoints với flag DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # để hiển thị size của từng keypoint
    output_bgr = cv2.drawKeypoints(
        output_bgr,
        keypoints,
        None,
        color=(0, 255, 0),  # BGR xanh lá
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return {
        "method": "FAST",
        "keypoints_count": int(len(keypoints)),
        "runtime_ms": round(runtime_ms, 3),
        "image": output_bgr,          # BGR — dùng st.image(..., channels="BGR")
        "keypoints": keypoints,
        "keypoints_xy": keypoints_xy,
        "params": {
            "threshold": threshold,
            "nonmax_suppression": nonmax_suppression,
            "type": type,
        },
    }


# Demo / test standalone 
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"[ERROR] Không đọc được ảnh: {sys.argv[1]}")
            sys.exit(1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Tạo ảnh test: nền tối + hình chữ nhật + vòng tròn → nhiều cạnh
        img_rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        img_rgb[:] = 60
        cv2.rectangle(img_rgb, (40, 40), (180, 160), (230, 230, 230), 2)
        cv2.rectangle(img_rgb, (200, 60), (360, 200), (210, 210, 210), 2)
        cv2.circle(img_rgb, (120, 240), 45, (200, 200, 200), 2)
        cv2.line(img_rgb, (0, 0), (400, 300), (180, 180, 180), 1)

    # Test với threshold mặc định
    result = detect_fast(img_rgb, threshold=10, nonmax_suppression=True)

    print("=" * 45)
    print(f"  Method         : {result['method']}")
    print(f"  Keypoints      : {result['keypoints_count']}")
    print(f"  Runtime        : {result['runtime_ms']} ms")
    print(f"  Output shape   : {result['image'].shape}")
    print(f"  Params         : {result['params']}")
    print("=" * 45)

    # So sánh threshold thấp vs cao
    result_low = detect_fast(img_rgb, threshold=5)
    result_high = detect_fast(img_rgb, threshold=30)
    print(f"\n  threshold=5  → {result_low['keypoints_count']} keypoints")
    print(f"  threshold=10 → {result['keypoints_count']} keypoints")
    print(f"  threshold=30 → {result_high['keypoints_count']} keypoints")

    cv2.imwrite("fast_output.jpg", result["image"])
    print("\nĐã lưu: fast_output.jpg")
