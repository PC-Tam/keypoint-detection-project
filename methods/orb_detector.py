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


def detect_orb(image: np.ndarray, nfeatures: int = 500) -> dict:
    """
    Phát hiện keypoints và trích xuất descriptors bằng thuật toán ORB.

    Parameters
    ----------
    image     : np.ndarray — ảnh đầu vào (RGB, BGR, hoặc grayscale)
    nfeatures : int        — số lượng keypoints tối đa cần giữ lại (mặc định 500)

    Returns
    -------
    dict với các key:
        method          : str                — "ORB"
        keypoints_count : int                — số keypoints phát hiện được
        runtime_ms      : float              — thời gian xử lý (ms)
        image           : np.ndarray         — ảnh BGR đã vẽ keypoints (chấm xanh lá)
        keypoints       : list[cv2.KeyPoint] — danh sách KeyPoint object
        descriptors     : np.ndarray hoặc None — numpy array chứa descriptors
    """
    # 1. Tiền xử lý
    image = _ensure_numpy(image)
    gray = _to_grayscale(image)
    output_bgr = _to_bgr(image)

    # 2. Khởi tạo ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # 3. Detect và compute
    t_start = time.perf_counter()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    t_end = time.perf_counter()

    runtime_ms = (t_end - t_start) * 1000.0

    # 4. Xử lý trường hợp không tìm thấy keypoints / descriptors
    if keypoints is None:
        keypoints = []
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)

    # 5. Vẽ keypoints lên ảnh
    output_bgr = cv2.drawKeypoints(
        output_bgr,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return {
        "method": "ORB",
        "keypoints_count": len(keypoints),
        "runtime_ms": round(runtime_ms, 3),
        "image": output_bgr,
        "keypoints": keypoints,
        "descriptors": descriptors
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

    result = detect_orb(img_rgb, nfeatures=500)

    print("=" * 45)
    print(f"  Method         : {result['method']}")
    print(f"  Keypoints      : {result['keypoints_count']}")
    print(f"  Runtime        : {result['runtime_ms']} ms")
    print(f"  Output shape   : {result['image'].shape}")
    print(f"  Descriptors    : {result['descriptors'].shape}")
    print("=" * 45)

    cv2.imwrite("orb_output.jpg", result["image"])
    print("\nSaved: orb_output.jpg")
