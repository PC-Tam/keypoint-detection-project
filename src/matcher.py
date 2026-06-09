"""
ORB Feature Matching — src/matcher.py
======================================
So khớp ảnh template với ảnh test bằng ORB descriptors + BFMatcher (NORM_HAMMING).

Pipeline mới (v2):
  1. Tiền xử lý cả template và test (resize, CLAHE, v.v.)
  2. ORB detectAndCompute
  3. BFMatcher + KNN matching (k=2)
  4. Lowe's ratio test → good matches
  5. Homography + RANSAC nếu đủ good matches
  6. Tính confidence score
  7. Vẽ bounding box + detected region

An toàn khi chạy trong Streamlit (không crash nếu descriptors = None).
"""

import time
import cv2
import numpy as np

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Import preprocessing
try:
    from methods.preprocessing import preprocess_image, DEFAULT_PREPROCESS_OPTIONS
except ImportError:
    # Fallback nếu import từ methods không được (chạy standalone)
    preprocess_image = None
    DEFAULT_PREPROCESS_OPTIONS = None


# ══════════════════════════════════════════════════════════════════════════════
#  Tiền xử lý cơ bản (dùng khi methods.preprocessing không available)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_numpy(image) -> np.ndarray:
    """Chuyển PIL Image → numpy array RGB; giữ nguyên nếu đã là ndarray."""
    if _PIL_AVAILABLE and isinstance(image, PILImage.Image):
        return np.array(image.convert("RGB"))
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            f"Expected numpy array or PIL Image."
        )
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh về grayscale (hỗ trợ RGB / RGBA / grayscale)."""
    if image is None:
        raise ValueError("Image is None.")
    if len(image.shape) == 2:
        return image.copy()
    if len(image.shape) == 3:
        c = image.shape[2]
        if c == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if c == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh về BGR 3-channel để vẽ màu."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image.copy()


def _build_empty_result(bgr_template, bgr_test, runtime_ms, reason=""):
    """Tạo kết quả rỗng khi matching thất bại sớm."""
    h1, w1 = bgr_template.shape[:2]
    h2, w2 = bgr_test.shape[:2]
    h_max = max(h1, h2)
    vis = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = bgr_template
    vis[:h2, w1:w1 + w2] = bgr_test

    label = reason if reason else "No descriptors found"
    cv2.putText(
        vis, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
    )

    return {
        "method": "ORB",
        "keypoints_template": 0,
        "keypoints_query": 0,
        "total_matches": 0,
        "good_matches": 0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "confidence_score": 0.0,
        "runtime_ms": round(runtime_ms, 3),
        "detected": False,
        "image": vis,
        "detected_region": None,
        "preprocessing_info": {},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Hàm chính: ORB Feature Matching (v2 — có preprocessing + KNN ratio test)
# ══════════════════════════════════════════════════════════════════════════════

def match_orb(
    template_image,
    test_image,
    # --- Preprocessing options ---
    preprocessing_options: dict = None,
    # --- ORB parameters ---
    nfeatures: int = 1500,
    scaleFactor: float = 1.2,
    nlevels: int = 8,
    edgeThreshold: int = 31,
    patchSize: int = 31,
    fastThreshold: int = 20,
    # --- Matching parameters ---
    ratio_threshold: float = 0.75,
    min_good_matches: int = 10,
    ransac_reproj_threshold: float = 5.0,
    min_inlier_ratio: float = 0.25,
) -> dict:
    """
    So khớp ảnh *template* (logo / nhãn) với ảnh *test* bằng ORB + BFMatcher.

    Pipeline:
        template + test → preprocess → ORB detectAndCompute
        → BFMatcher KNN (k=2) → Lowe's ratio test → good matches
        → Homography + RANSAC → inlier ratio → confidence score

    Parameters
    ----------
    template_image        : np.ndarray | PIL.Image — ảnh mẫu (logo / nhãn)
    test_image            : np.ndarray | PIL.Image — ảnh cần kiểm tra
    preprocessing_options : dict | None — tùy chọn preprocessing (None = mặc định)
    nfeatures             : int   — số ORB features tối đa (default 1500)
    scaleFactor           : float — scale pyramid (default 1.2)
    nlevels               : int   — số pyramid levels (default 8)
    edgeThreshold         : int   — edge threshold (default 31)
    patchSize             : int   — patch size cho descriptor (default 31)
    fastThreshold         : int   — FAST threshold (default 20)
    ratio_threshold       : float — Lowe's ratio test threshold (default 0.75)
    min_good_matches      : int   — ngưỡng tối thiểu good matches (default 10)
    ransac_reproj_threshold : float — RANSAC reprojection threshold (default 5.0)
    min_inlier_ratio      : float — tỷ lệ inlier tối thiểu (default 0.25)

    Returns
    -------
    dict:
        method              : str        — "ORB"
        keypoints_template  : int        — số keypoints ảnh mẫu
        keypoints_query     : int        — số keypoints ảnh test
        total_matches       : int        — tổng số matches trước ratio test
        good_matches        : int        — số good matches sau ratio test
        inlier_count        : int        — số inliers từ RANSAC
        inlier_ratio        : float      — inlier_count / good_matches
        confidence_score    : float      — điểm tin cậy tổng hợp (0.0 - 1.0)
        runtime_ms          : float      — tổng thời gian xử lý (ms)
        detected            : bool       — True nếu nhận diện thành công
        image               : np.ndarray — ảnh matching (BGR)
        detected_region     : np.ndarray | None — ảnh test có vẽ bounding box (BGR)
        preprocessing_info  : dict       — thông tin preprocessing đã áp dụng
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    #  1. Chuyển đổi input
    # ------------------------------------------------------------------
    template_np = _ensure_numpy(template_image)
    test_np = _ensure_numpy(test_image)

    # ------------------------------------------------------------------
    #  2. Tiền xử lý ảnh
    # ------------------------------------------------------------------
    preprocess_info = {"template": {}, "query": {}}

    if preprocess_image is not None:
        # Dùng preprocessing module
        gray_template, info_tpl = preprocess_image(template_np, preprocessing_options)
        gray_test, info_tst = preprocess_image(test_np, preprocessing_options)
        preprocess_info["template"] = info_tpl
        preprocess_info["query"] = info_tst
    else:
        # Fallback: chỉ grayscale
        gray_template = _to_grayscale(template_np)
        gray_test = _to_grayscale(test_np)

    # BGR cho visualization (dùng ảnh đã resize nếu có)
    # Resize ảnh gốc theo cùng kích thước với gray đã xử lý
    if gray_template.shape[:2] != template_np.shape[:2]:
        h, w = gray_template.shape[:2]
        template_resized = cv2.resize(template_np, (w, h), interpolation=cv2.INTER_AREA)
    else:
        template_resized = template_np

    if gray_test.shape[:2] != test_np.shape[:2]:
        h, w = gray_test.shape[:2]
        test_resized = cv2.resize(test_np, (w, h), interpolation=cv2.INTER_AREA)
    else:
        test_resized = test_np

    bgr_template = _to_bgr(template_resized)
    bgr_test = _to_bgr(test_resized)

    # ------------------------------------------------------------------
    #  3. Kiểm tra ảnh hợp lệ
    # ------------------------------------------------------------------
    if gray_template.size == 0 or gray_test.size == 0:
        runtime_ms = (time.perf_counter() - t_start) * 1000.0
        return _build_empty_result(
            bgr_template, bgr_test, runtime_ms,
            reason="Image too small or invalid"
        )

    # ------------------------------------------------------------------
    #  4. Detect ORB keypoints & descriptors
    # ------------------------------------------------------------------
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=scaleFactor,
        nlevels=nlevels,
        edgeThreshold=edgeThreshold,
        patchSize=patchSize,
        fastThreshold=fastThreshold,
    )

    kp_template, des_template = orb.detectAndCompute(gray_template, None)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    n_kp_template = len(kp_template) if kp_template else 0
    n_kp_test = len(kp_test) if kp_test else 0

    # ------------------------------------------------------------------
    #  5. Kiểm tra descriptors = None → detected = False
    # ------------------------------------------------------------------
    if des_template is None or des_test is None:
        runtime_ms = (time.perf_counter() - t_start) * 1000.0
        result = _build_empty_result(
            bgr_template, bgr_test, runtime_ms,
            reason=f"No descriptors (kp_tpl={n_kp_template}, kp_test={n_kp_test})"
        )
        result["keypoints_template"] = n_kp_template
        result["keypoints_query"] = n_kp_test
        result["preprocessing_info"] = preprocess_info
        return result

    # ------------------------------------------------------------------
    #  6. BFMatcher + KNN matching (k=2) + Lowe's ratio test
    # ------------------------------------------------------------------
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        raw_matches = bf.knnMatch(des_template, des_test, k=2)
    except cv2.error:
        # Fallback nếu KNN thất bại (ví dụ quá ít descriptors)
        runtime_ms = (time.perf_counter() - t_start) * 1000.0
        result = _build_empty_result(
            bgr_template, bgr_test, runtime_ms,
            reason="KNN matching failed"
        )
        result["keypoints_template"] = n_kp_template
        result["keypoints_query"] = n_kp_test
        result["preprocessing_info"] = preprocess_info
        return result

    total_matches = len(raw_matches)

    # Lowe's ratio test
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    # Sort theo distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    num_good = len(good_matches)

    # ------------------------------------------------------------------
    #  7. Homography + RANSAC
    # ------------------------------------------------------------------
    detected = False
    inlier_count = 0
    inlier_ratio = 0.0
    confidence_score = 0.0
    detected_region_img = None

    if num_good >= min_good_matches and num_good >= 4:
        src_pts = np.float32(
            [kp_template[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp_test[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold
        )

        if mask is not None:
            inlier_count = int(mask.sum())
            inlier_ratio = inlier_count / num_good if num_good > 0 else 0.0

        # Tính confidence score
        # = (good_matches / min(kp_tpl, kp_test)) * inlier_ratio
        min_kp = min(n_kp_template, n_kp_test) if min(n_kp_template, n_kp_test) > 0 else 1
        match_ratio = min(num_good / min_kp, 1.0)
        confidence_score = round(match_ratio * inlier_ratio, 4)

        # Vẽ bounding box nếu Homography hợp lệ + đủ inlier
        if H is not None and inlier_ratio >= min_inlier_ratio:
            # Kiểm tra Homography không bị suy biến
            det_H = np.linalg.det(H[:2, :2])
            if 0.05 < abs(det_H) < 20.0:  # Loại bỏ biến dạng quá mạnh
                detected = True

                h_t, w_t = gray_template.shape[:2]
                corners = np.float32(
                    [[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]
                ).reshape(-1, 1, 2)

                try:
                    projected = cv2.perspectiveTransform(corners, H)
                    projected = np.int32(projected)

                    # Tạo ảnh detected region
                    detected_region_img = bgr_test.copy()

                    # Vẽ bounding box (xanh lá, dày 3px)
                    cv2.polylines(
                        detected_region_img, [projected], isClosed=True,
                        color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA,
                    )

                    # Ghi nhãn
                    x_min = projected[:, 0, 0].min()
                    y_min = projected[:, 0, 1].min()
                    label_text = f"DETECTED ({confidence_score:.0%})"
                    cv2.putText(
                        detected_region_img, label_text,
                        (max(x_min, 5), max(y_min - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )

                    # Cũng vẽ lên bgr_test cho ảnh matching
                    cv2.polylines(
                        bgr_test, [projected], isClosed=True,
                        color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        bgr_test, label_text,
                        (max(x_min, 5), max(y_min - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                except cv2.error:
                    # perspectiveTransform có thể fail với H suy biến
                    detected = False

    # ------------------------------------------------------------------
    #  8. Vẽ ảnh matching
    # ------------------------------------------------------------------
    # Chọn màu dựa trên kết quả
    match_color = (0, 255, 0) if detected else (0, 100, 255)

    match_img = cv2.drawMatches(
        bgr_template, kp_template,
        bgr_test, kp_test,
        good_matches, None,
        matchColor=match_color,
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    runtime_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "method": "ORB",
        "keypoints_template": n_kp_template,
        "keypoints_query": n_kp_test,
        "total_matches": total_matches,
        "good_matches": num_good,
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "confidence_score": confidence_score,
        "runtime_ms": round(runtime_ms, 3),
        "detected": detected,
        "image": match_img,
        "detected_region": detected_region_img,
        "preprocessing_info": preprocess_info,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Demo / test standalone
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python matcher.py <template_path> <test_path>")
        sys.exit(1)

    tpl = cv2.imread(sys.argv[1])
    tst = cv2.imread(sys.argv[2])
    if tpl is None:
        print(f"[ERROR] Không đọc được template: {sys.argv[1]}")
        sys.exit(1)
    if tst is None:
        print(f"[ERROR] Không đọc được test image: {sys.argv[2]}")
        sys.exit(1)

    # OpenCV imread trả về BGR → chuyển sang RGB (giả lập input từ Streamlit)
    tpl_rgb = cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB)
    tst_rgb = cv2.cvtColor(tst, cv2.COLOR_BGR2RGB)

    result = match_orb(tpl_rgb, tst_rgb)

    print("=" * 60)
    print(f"  Method            : {result['method']}")
    print(f"  KP Template       : {result['keypoints_template']}")
    print(f"  KP Query          : {result['keypoints_query']}")
    print(f"  Total matches     : {result['total_matches']}")
    print(f"  Good matches      : {result['good_matches']}")
    print(f"  Inlier count      : {result['inlier_count']}")
    print(f"  Inlier ratio      : {result['inlier_ratio']}")
    print(f"  Confidence score  : {result['confidence_score']}")
    print(f"  Runtime           : {result['runtime_ms']} ms")
    print(f"  Detected          : {result['detected']}")
    print(f"  Output shape      : {result['image'].shape}")
    print("=" * 60)

    cv2.imwrite("match_output.jpg", result["image"])
    print("\nSaved: match_output.jpg")

    if result["detected_region"] is not None:
        cv2.imwrite("detected_region.jpg", result["detected_region"])
        print("Saved: detected_region.jpg")
