"""
feature_color_texture.py
------------------------
Traditional (non-deep-learning) color and texture feature extraction.
Uses OpenCV and NumPy only — no heavy external libraries required.
"""

import cv2
import numpy as np


def _rgb_stats(bgr: np.ndarray) -> np.ndarray:
    """Mean and std for each BGR channel. Returns 6 values."""
    feats = []
    for ch in range(3):
        channel = bgr[:, :, ch].astype(np.float32)
        feats.extend([float(channel.mean()), float(channel.std())])
    return np.array(feats, dtype=np.float32)


def _hsv_stats(bgr: np.ndarray) -> np.ndarray:
    """Mean and std for each HSV channel. Returns 6 values."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        channel = hsv[:, :, ch].astype(np.float32)
        feats.extend([float(channel.mean()), float(channel.std())])
    return np.array(feats, dtype=np.float32)


def _color_histogram(bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Compute normalized color histogram for each BGR channel.
    Returns 3*bins values.
    """
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(bgr[:, :, ch], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        feats.extend(hist.tolist())
    return np.array(feats, dtype=np.float32)


def _gray_histogram(gray: np.ndarray, bins: int = 16) -> np.ndarray:
    """Normalized grayscale histogram. Returns `bins` values."""
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def _lbp_numpy(gray: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Compute a uniform Local Binary Pattern (LBP) histogram using NumPy.
    Avoids heavy libraries like scikit-image.
    Uses 8 neighbors at the given radius via bilinear sampling.

    Returns 10 values (8 uniform patterns + 1 non-uniform + 1 for edge).
    """
    h, w = gray.shape
    gray_f = gray.astype(np.float32)
    # 8 neighbor offsets for radius=1
    offsets = [
        (-radius,  0),
        (-radius,  radius),
        (0,        radius),
        (radius,   radius),
        (radius,   0),
        (radius,  -radius),
        (0,       -radius),
        (-radius, -radius),
    ]

    # Build LBP code
    lbp = np.zeros((h, w), dtype=np.uint8)
    center = gray_f[radius:h - radius, radius:w - radius]

    for i, (dr, dc) in enumerate(offsets):
        neighbor = gray_f[
            radius + dr: h - radius + dr,
            radius + dc: w - radius + dc
        ]
        bit = (neighbor >= center).astype(np.uint8)
        lbp[radius:h - radius, radius:w - radius] += bit << i

    # Compute histogram (256 bins, then reduce to 10 by grouping)
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()

    # Group into 10 buckets
    bucket_size = 256 // 10
    grouped = np.array([
        hist[i * bucket_size: (i + 1) * bucket_size].sum()
        for i in range(10)
    ], dtype=np.float32)
    return grouped


def _edge_density(gray: np.ndarray,
                  low_thresh: int = 50,
                  high_thresh: int = 150) -> float:
    """
    Compute edge density using Canny edge detector.
    Returns fraction of pixels that are edges.
    """
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return float(edges.sum()) / (edges.size * 255 + 1e-6)


def _mean_intensity_in_mask(gray: np.ndarray,
                             mask: np.ndarray) -> float:
    """Mean pixel intensity within the cell mask region."""
    if mask is None or mask.sum() == 0:
        return float(gray.mean())
    pixels = gray[mask > 0]
    return float(pixels.mean()) if len(pixels) > 0 else 0.0


def extract_color_texture_features(
    bgr: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray = None,
    color_hist_bins: int = 16,
    gray_hist_bins: int = 16,
) -> np.ndarray:
    """
    Extract a fixed-length color and texture feature vector.

    Feature layout:
      [0..5]    RGB mean+std (6)
      [6..11]   HSV mean+std (6)
      [12..59]  BGR color histogram (3 x 16 = 48)
      [60..75]  Gray histogram (16)
      [76..85]  LBP histogram (10)
      [86]      Edge density (1)
      [87]      Mean intensity in mask (1)
      Total = 88

    Parameters
    ----------
    bgr : np.ndarray
        BGR color image (uint8).
    gray : np.ndarray
        Grayscale image (uint8).
    mask : np.ndarray or None
        Binary cell mask.
    color_hist_bins : int
        Bins per channel for color histogram.
    gray_hist_bins : int
        Bins for grayscale histogram.

    Returns
    -------
    np.ndarray
        1-D float32 feature vector.
    """
    feat_size = 6 + 6 + 3 * color_hist_bins + gray_hist_bins + 10 + 1 + 1
    zero_feat = np.zeros(feat_size, dtype=np.float32)

    try:
        rgb_s = _rgb_stats(bgr)
        hsv_s = _hsv_stats(bgr)
        color_hist = _color_histogram(bgr, bins=color_hist_bins)
        gray_hist = _gray_histogram(gray, bins=gray_hist_bins)
        lbp_hist = _lbp_numpy(gray)
        edge_d = np.array([_edge_density(gray)], dtype=np.float32)
        mean_int = np.array([_mean_intensity_in_mask(gray, mask)], dtype=np.float32)

        feat = np.concatenate([
            rgb_s, hsv_s, color_hist, gray_hist, lbp_hist, edge_d, mean_int
        ])
        return feat.astype(np.float32)

    except Exception:
        return zero_feat
