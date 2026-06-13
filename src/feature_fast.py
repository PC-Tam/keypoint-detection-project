"""
feature_fast.py
---------------
FAST (Features from Accelerated Segment Test) keypoint detection and
feature extraction for blood cell classification.
"""

import cv2
import numpy as np


# -- Internal helpers ---------------------------------------------------------

def _safe_stats(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, max, std); all zeros if empty."""
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(arr)), float(np.max(arr)), float(np.std(arr))


def detect_fast_keypoints(
    gray: np.ndarray,
    threshold: int = 20,
    nonmax_suppression: bool = True,
) -> list:
    """
    Detect FAST keypoints in a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    threshold : int
        FAST intensity difference threshold.
    nonmax_suppression : bool
        Whether to apply non-maximum suppression.

    Returns
    -------
    list of cv2.KeyPoint
        Detected keypoints.
    """
    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=nonmax_suppression,
    )
    return fast.detect(gray, None)


def draw_fast_keypoints(
    image: np.ndarray,
    keypoints: list,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """
    Draw FAST keypoints on an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image.
    keypoints : list of cv2.KeyPoint
        Keypoints to draw.
    color : tuple
        BGR color.

    Returns
    -------
    np.ndarray
        BGR image with keypoints drawn.
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    return cv2.drawKeypoints(vis, keypoints, None, color=color,
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)


def extract_fast_features(
    gray: np.ndarray,
    mask: np.ndarray = None,
    contour=None,
    threshold: int = 20,
    nonmax_suppression: bool = True,
    grid_size: tuple = (4, 4),
    response_hist_bins: int = 10,
) -> np.ndarray:
    """
    Extract a fixed-length FAST feature vector from a grayscale image.

    Feature layout (total = 5 + grid_rows*grid_cols + response_hist_bins + 3):
      [0]   number_of_keypoints
      [1]   mean_response
      [2]   max_response
      [3]   std_response
      [4]   keypoint_density  (keypoints / image_area)
      [5 .. 5+G*G-1]   spatial histogram (grid cells)
      [5+G*G .. 5+G*G+B-1]  response histogram (bins)
      [-3]  n_keypoints_in_mask
      [-2]  ratio_keypoints_in_mask
      [-1]  n_keypoints_near_contour

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    mask : np.ndarray or None
        Binary mask for the cell region.
    contour : array or None
        Cell contour.
    threshold : int
        FAST threshold.
    nonmax_suppression : bool
        Enable non-max suppression.
    grid_size : tuple
        Grid (rows, cols) for spatial histogram.
    response_hist_bins : int
        Bins for response histogram.

    Returns
    -------
    np.ndarray
        1-D feature vector (float32).
    """
    h, w = gray.shape[:2]
    n_grid = grid_size[0] * grid_size[1]
    feat_size = 5 + n_grid + response_hist_bins + 3
    zero_feat = np.zeros(feat_size, dtype=np.float32)

    keypoints = detect_fast_keypoints(gray, threshold, nonmax_suppression)
    if not keypoints:
        return zero_feat

    xs = np.array([kp.pt[0] for kp in keypoints], dtype=np.float32)
    ys = np.array([kp.pt[1] for kp in keypoints], dtype=np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

    mean_r, max_r, std_r = _safe_stats(responses)
    density = len(keypoints) / (h * w) if (h * w) > 0 else 0.0

    # -- Spatial grid histogram -----------------------------------------------
    cell_h = h / grid_size[0]
    cell_w = w / grid_size[1]
    spatial_hist = np.zeros(n_grid, dtype=np.float32)
    for x, y in zip(xs, ys):
        row = min(int(y / cell_h), grid_size[0] - 1)
        col = min(int(x / cell_w), grid_size[1] - 1)
        spatial_hist[row * grid_size[1] + col] += 1
    if spatial_hist.sum() > 0:
        spatial_hist /= spatial_hist.sum()

    # -- Response histogram ---------------------------------------------------
    resp_hist, _ = np.histogram(responses, bins=response_hist_bins)
    resp_hist = resp_hist.astype(np.float32)
    if resp_hist.sum() > 0:
        resp_hist /= resp_hist.sum()

    # -- Mask-based features --------------------------------------------------
    n_in_mask = 0
    if mask is not None:
        for x, y in zip(xs.astype(int), ys.astype(int)):
            xc = np.clip(x, 0, w - 1)
            yc = np.clip(y, 0, h - 1)
            if mask[yc, xc] > 0:
                n_in_mask += 1
    ratio_in_mask = n_in_mask / len(keypoints) if keypoints else 0.0

    # -- Near-contour keypoints ------------------------------------------------
    n_near_contour = 0
    if contour is not None and len(contour) > 0:
        for x, y in zip(xs.astype(int), ys.astype(int)):
            dist = abs(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
            if dist < 5.0:
                n_near_contour += 1

    # -- Assemble --------------------------------------------------------------
    feat = np.concatenate([
        np.array([len(keypoints), mean_r, max_r, std_r, density], dtype=np.float32),
        spatial_hist,
        resp_hist,
        np.array([n_in_mask, ratio_in_mask, n_near_contour], dtype=np.float32),
    ])
    return feat
