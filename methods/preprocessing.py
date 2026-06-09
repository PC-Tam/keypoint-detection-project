"""
Smart Image Preprocessing — methods/preprocessing.py
=====================================================
Pipeline tiền xử lý ảnh TỰ ĐỘNG cho keypoint detection.
Phân tích đặc tính ảnh (nhiễu, tương phản, độ sáng) rồi tự chọn
bước xử lý phù hợp — người dùng KHÔNG cần chỉnh thông số.

Tham khảo best practices:
  - Bilateral Filter thay Gaussian Blur (giữ cạnh, loại nhiễu)
  - CLAHE trên kênh L (LAB) để cải thiện contrast cục bộ
  - Adaptive theo từng thuật toán (FAST nhạy nhiễu, Harris nhạy contrast)
"""

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  1. PHÂN TÍCH ẢNH — Tự đánh giá đặc tính ảnh
# ═══════════════════════════════════════════════════════════════════════

def analyze_image(image_rgb: np.ndarray) -> dict:
    """
    Phân tích các đặc tính của ảnh để quyết định tiền xử lý.

    Returns
    -------
    dict:
        noise_level    : float — ước lượng mức nhiễu (0 = sạch, >30 = rất nhiễu)
        contrast_score : float — độ tương phản (0 = phẳng, >60 = tốt)
        brightness     : float — độ sáng trung bình (0-255)
        is_noisy       : bool  — True nếu nhiễu cao
        is_low_contrast: bool  — True nếu tương phản thấp
        is_dark        : bool  — True nếu ảnh quá tối
        is_bright      : bool  — True nếu ảnh quá sáng
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # --- Ước lượng mức nhiễu (Laplacian + Median Absolute Deviation) ---
    # Phương pháp robust: dùng MAD của Laplacian trên vùng phẳng
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # MAD (Median Absolute Deviation) — robust hơn variance
    noise_level = float(np.median(np.abs(laplacian)) * 1.4826)

    # --- Đánh giá tương phản ---
    # Dùng percentile range thay vì min/max (robust với outlier)
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    contrast_score = float(p95 - p5)

    # --- Độ sáng trung bình ---
    brightness = float(np.mean(gray))

    return {
        "noise_level": round(noise_level, 2),
        "contrast_score": round(contrast_score, 2),
        "brightness": round(brightness, 2),
        "is_noisy": noise_level > 15,
        "is_low_contrast": contrast_score < 80,
        "is_dark": brightness < 60,
        "is_bright": brightness > 200,
    }


# ═══════════════════════════════════════════════════════════════════════
#  2. TIỀN XỬ LÝ TỰ ĐỘNG — Adaptive pipeline
# ═══════════════════════════════════════════════════════════════════════

def auto_preprocess(
    image_rgb: np.ndarray,
    algorithm: str = "ORB",
) -> tuple[np.ndarray, dict]:
    """
    Tự động phân tích và tiền xử lý ảnh dựa trên đặc tính ảnh
    VÀ thuật toán keypoint detection sẽ dùng.

    Parameters
    ----------
    image_rgb : np.ndarray — ảnh RGB đầu vào
    algorithm : str        — "Harris", "FAST", hoặc "ORB"

    Returns
    -------
    (processed_image, report) : tuple
        processed_image : np.ndarray — ảnh RGB đã tiền xử lý
        report          : dict       — báo cáo chi tiết những gì đã làm
    """
    analysis = analyze_image(image_rgb)
    steps_applied = []
    params_used = {}
    img = image_rgb.copy()

    # ------------------------------------------------------------------
    #  Bước 1: GIẢM NHIỄU — Bilateral Filter (giữ cạnh)
    # ------------------------------------------------------------------
    # FAST rất nhạy nhiễu → luôn denoise nếu dùng FAST
    # Các thuật toán khác → chỉ denoise khi nhiễu cao
    should_denoise = analysis["is_noisy"] or algorithm == "FAST"

    if should_denoise:
        # Chọn tham số bilateral theo mức nhiễu
        noise = analysis["noise_level"]
        if noise > 30:
            # Nhiễu rất cao → filter mạnh
            d, sigma_color, sigma_space = 9, 90, 90
        elif noise > 15:
            # Nhiễu trung bình
            d, sigma_color, sigma_space = 7, 75, 75
        else:
            # Nhiễu nhẹ (chỉ áp dụng cho FAST)
            d, sigma_color, sigma_space = 5, 50, 50

        img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        steps_applied.append("Bilateral Filter")
        params_used["bilateral"] = {
            "d": d, "sigma_color": sigma_color,
            "sigma_space": sigma_space, "reason": f"noise={noise:.1f}"
        }

    # ------------------------------------------------------------------
    #  Bước 2: CẢI THIỆN TƯƠNG PHẢN — CLAHE trên kênh L (LAB)
    # ------------------------------------------------------------------
    # Harris nhạy với intensity → luôn cần contrast tốt
    # Ảnh tối/sáng quá → cũng cần CLAHE
    should_clahe = (
        analysis["is_low_contrast"]
        or analysis["is_dark"]
        or analysis["is_bright"]
        or algorithm == "Harris"
    )

    if should_clahe:
        # Clip limit adaptive: contrast thấp → clip cao hơn
        contrast = analysis["contrast_score"]
        if contrast < 40:
            clip_limit = 4.0  # Rất ít contrast → tăng mạnh
        elif contrast < 80:
            clip_limit = 3.0  # Trung bình
        else:
            clip_limit = 2.0  # Chỉ cải thiện nhẹ

        # Áp dụng CLAHE trên kênh L của LAB (giữ nguyên màu)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        steps_applied.append("CLAHE")
        params_used["clahe"] = {
            "clip_limit": clip_limit, "tile_size": 8,
            "reason": f"contrast={contrast:.1f}, brightness={analysis['brightness']:.0f}"
        }

    # ------------------------------------------------------------------
    #  Bước 3: CHUẨN HÓA ĐỘ SÁNG — nếu ảnh quá tối hoặc quá sáng
    # ------------------------------------------------------------------
    if analysis["is_dark"] or analysis["is_bright"]:
        # Dùng gamma correction
        brightness = analysis["brightness"]
        if brightness < 60:
            # Ảnh tối → tăng gamma (sáng lên)
            gamma = 0.6
        elif brightness > 200:
            # Ảnh quá sáng → giảm gamma (tối lại)
            gamma = 1.6
        else:
            gamma = 1.0

        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype("uint8")
            img = cv2.LUT(img, table)

            steps_applied.append("Gamma Correction")
            params_used["gamma"] = {
                "gamma": gamma,
                "reason": f"brightness={brightness:.0f}"
            }

    # ------------------------------------------------------------------
    #  Bước 4: LÀM MỊN NHẸ cuối cùng (chỉ cho FAST)
    # ------------------------------------------------------------------
    # FAST detect corner trên cường độ pixel trực tiếp
    # → Gaussian blur nhẹ cuối cùng để loại micro-noise còn sót
    if algorithm == "FAST" and analysis["noise_level"] > 10:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        steps_applied.append("Gaussian Blur nhẹ (3×3)")
        params_used["gaussian_final"] = {"ksize": 3, "reason": "FAST final smoothing"}

    # ------------------------------------------------------------------
    #  Tổng hợp report
    # ------------------------------------------------------------------
    if not steps_applied:
        steps_applied.append("Không cần tiền xử lý (ảnh đã đạt chất lượng)")

    report = {
        "analysis": analysis,
        "steps_applied": steps_applied,
        "params_used": params_used,
        "algorithm": algorithm,
        "summary": " → ".join(steps_applied),
    }

    return img, report


# ═══════════════════════════════════════════════════════════════════════
#  3. TIỀN XỬ LÝ TỔNG QUÁT — preprocess_image()
# ═══════════════════════════════════════════════════════════════════════

# Thông số mặc định (dễ import / override)
DEFAULT_PREPROCESS_OPTIONS = {
    # --- Resize ---
    "resize": True,
    "max_size": 1024,           # Giới hạn chiều dài/rộng tối đa (px)

    # --- Grayscale ---
    "grayscale": True,          # Luôn chuyển grayscale cho detector

    # --- CLAHE ---
    "clahe": True,              # Mặc định BẬT — cải thiện tương phản cục bộ
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": 8,

    # --- Gaussian Blur (tùy chọn) ---
    "gaussian_blur": False,
    "gaussian_ksize": 5,

    # --- Median Blur (tùy chọn) ---
    "median_blur": False,
    "median_ksize": 5,

    # --- Contrast Stretching (tùy chọn) ---
    "contrast_stretching": False,
    "contrast_low_percentile": 2,
    "contrast_high_percentile": 98,

    # --- Sharpen (tùy chọn) ---
    "sharpen": False,
    "sharpen_strength": 0.5,    # 0.0 = không sharpen, 1.0 = mạnh
}


def _resize_keep_aspect(image: np.ndarray, max_size: int) -> np.ndarray:
    """
    Resize ảnh giữ nguyên tỉ lệ sao cho chiều lớn nhất <= max_size.
    Nếu ảnh đã nhỏ hơn max_size thì giữ nguyên.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(
    image: np.ndarray,
    options: dict = None,
) -> tuple:
    """
    Tiền xử lý ảnh tổng quát cho keypoint detection / matching.

    Pipeline (theo thứ tự):
        1. Resize giữ tỉ lệ        (mặc định BẬT)
        2. Chuyển grayscale          (mặc định BẬT)
        3. CLAHE                     (mặc định BẬT)
        4. Contrast Stretching       (tùy chọn)
        5. Gaussian Blur             (tùy chọn)
        6. Median Blur               (tùy chọn)
        7. Sharpen                   (tùy chọn)

    KHÔNG dùng mặc định:
        - Global Histogram Equalization (khuếch đại nhiễu)
        - Thresholding (phá texture/descriptor cho ORB)

    Parameters
    ----------
    image   : np.ndarray — ảnh đầu vào (RGB hoặc grayscale)
    options : dict | None — dict tùy chọn, None = dùng DEFAULT_PREPROCESS_OPTIONS

    Returns
    -------
    (processed_image, info) : tuple
        processed_image : np.ndarray — ảnh grayscale đã xử lý (uint8)
        info            : dict — thông tin các bước đã áp dụng
            steps_applied  : list[str]
            original_size  : (h, w)
            resized_size   : (h, w)
            options_used   : dict
    """
    # Merge options với defaults
    opts = DEFAULT_PREPROCESS_OPTIONS.copy()
    if options is not None:
        opts.update(options)

    steps_applied = []
    img = image.copy()
    original_h, original_w = img.shape[:2]

    # ── 1. Resize giữ tỉ lệ ──────────────────────────────────────────
    if opts["resize"] and opts["max_size"] > 0:
        img = _resize_keep_aspect(img, opts["max_size"])
        new_h, new_w = img.shape[:2]
        if (new_h, new_w) != (original_h, original_w):
            steps_applied.append(
                f"Resize {original_w}×{original_h} → {new_w}×{new_h}"
            )

    # ── 2. Grayscale ──────────────────────────────────────────────────
    if opts["grayscale"]:
        if len(img.shape) == 3:
            channels = img.shape[2]
            if channels == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif channels == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]
            steps_applied.append("Grayscale")
        else:
            gray = img.copy()
    else:
        # Nếu không chuyển grayscale, vẫn cần gray cho CLAHE
        if len(img.shape) == 3:
            channels = img.shape[2]
            if channels == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif channels == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]
        else:
            gray = img.copy()

    # ── 3. CLAHE ──────────────────────────────────────────────────────
    if opts["clahe"]:
        clip = opts["clahe_clip_limit"]
        tile = opts["clahe_tile_grid_size"]
        clahe = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(tile, tile),
        )
        gray = clahe.apply(gray)
        steps_applied.append(f"CLAHE (clip={clip}, tile={tile})")

    # ── 4. Contrast Stretching ────────────────────────────────────────
    if opts["contrast_stretching"]:
        low_p = opts["contrast_low_percentile"]
        high_p = opts["contrast_high_percentile"]
        p_low = np.percentile(gray, low_p)
        p_high = np.percentile(gray, high_p)
        if p_high > p_low:
            gray = np.clip(
                (gray.astype(np.float32) - p_low) / (p_high - p_low) * 255,
                0, 255,
            ).astype(np.uint8)
            steps_applied.append(
                f"Contrast Stretching (p{low_p}-p{high_p})"
            )

    # ── 5. Gaussian Blur ──────────────────────────────────────────────
    if opts["gaussian_blur"]:
        ksize = opts["gaussian_ksize"]
        # Đảm bảo ksize lẻ
        if ksize % 2 == 0:
            ksize += 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        steps_applied.append(f"Gaussian Blur (k={ksize})")

    # ── 6. Median Blur ────────────────────────────────────────────────
    if opts["median_blur"]:
        ksize = opts["median_ksize"]
        if ksize % 2 == 0:
            ksize += 1
        gray = cv2.medianBlur(gray, ksize)
        steps_applied.append(f"Median Blur (k={ksize})")

    # ── 7. Sharpen ────────────────────────────────────────────────────
    if opts["sharpen"]:
        strength = opts["sharpen_strength"]
        # Unsharp masking: sharp = original + strength * (original - blurred)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        gray = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
        steps_applied.append(f"Sharpen (strength={strength})")

    # ── Tổng hợp info ────────────────────────────────────────────────
    if not steps_applied:
        steps_applied.append("Không có tiền xử lý")

    resized_h, resized_w = gray.shape[:2]

    info = {
        "steps_applied": steps_applied,
        "summary": " → ".join(steps_applied),
        "original_size": (original_h, original_w),
        "resized_size": (resized_h, resized_w),
        "options_used": opts,
    }

    return gray, info


# ═══════════════════════════════════════════════════════════════════════
#  4. TIỆN ÍCH — Format report cho hiển thị
# ═══════════════════════════════════════════════════════════════════════

def format_analysis_report(report: dict) -> str:
    """Tạo chuỗi mô tả chi tiết từ report để hiển thị trên UI."""
    a = report["analysis"]
    lines = []
    lines.append(f"📊 **Phân tích ảnh:**")
    lines.append(f"- Mức nhiễu: **{a['noise_level']:.1f}** {'⚠️ Cao' if a['is_noisy'] else '✅ Thấp'}")
    lines.append(f"- Tương phản: **{a['contrast_score']:.0f}** {'⚠️ Yếu' if a['is_low_contrast'] else '✅ Tốt'}")
    lines.append(f"- Độ sáng: **{a['brightness']:.0f}/255** "
                 f"{'⚠️ Tối' if a['is_dark'] else '⚠️ Quá sáng' if a['is_bright'] else '✅ Bình thường'}")
    lines.append(f"")
    lines.append(f"🔧 **Đã áp dụng:** {report['summary']}")
    return "\n".join(lines)
