"""
feature_orb.py
--------------
ORB (Oriented FAST and Rotated BRIEF) keypoint detection and feature
extraction for blood cell classification.
Provides both statistical features and descriptors for BoVW.
"""

import cv2
import numpy as np


# -- Internal helpers ---------------------------------------------------------

def _safe_stats(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, max, std); all zeros if empty."""
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(arr)), float(np.max(arr)), float(np.std(arr))


def extract_orb_keypoints_descriptors(
    gray: np.ndarray,
    nfeatures: int = 500,
) -> tuple[list, np.ndarray]:
    """
    Extract ORB keypoints and descriptors from a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    nfeatures : int
        Maximum number of features to retain.

    Returns
    -------
    tuple of (keypoints, descriptors)
        keypoints  – list of cv2.KeyPoint (may be empty)
        descriptors – np.ndarray of shape (N, 32) uint8, or empty array
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if keypoints is None:
        keypoints = []
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
    return keypoints, descriptors


def draw_orb_keypoints(
    image: np.ndarray,
    keypoints: list,
    color: tuple = (255, 165, 0),
) -> np.ndarray:
    """
    Draw ORB keypoints on an image (with orientation).

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image.
    keypoints : list of cv2.KeyPoint
        Keypoints to draw.
    color : tuple
        BGR color for keypoints.

    Returns
    -------
    np.ndarray
        BGR image with ORB keypoints drawn.
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    return cv2.drawKeypoints(
        vis, keypoints, None, color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def extract_orb_statistical_features(
    gray: np.ndarray,
    mask: np.ndarray = None,
    contour=None,
    nfeatures: int = 500,
    grid_size: tuple = (4, 4),
    response_hist_bins: int = 10,
) -> np.ndarray:
    """
    Extract a fixed-length ORB statistical feature vector.

    Feature layout:
      [0]   number_of_keypoints
      [1]   mean_response
      [2]   max_response
      [3]   std_response
      [4]   keypoint_density
      [5]   mean_angle
      [6]   std_angle
      [7]   mean_size
      [8]   std_size
      [9 .. 9+G*G-1]   spatial histogram
      [9+G*G .. 9+G*G+B-1]  response histogram
      [-3]  n_keypoints_in_mask
      [-2]  ratio_keypoints_in_mask
      [-1]  n_keypoints_near_contour

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    mask : np.ndarray or None
        Binary cell mask.
    contour : array or None
        Cell contour.
    nfeatures : int
        Max ORB features.
    grid_size : tuple
        (rows, cols) for spatial histogram.
    response_hist_bins : int
        Bins for response histogram.

    Returns
    -------
    np.ndarray
        1-D float32 feature vector.
    """
    h, w = gray.shape[:2]
    n_grid = grid_size[0] * grid_size[1]
    feat_size = 9 + n_grid + response_hist_bins + 3
    zero_feat = np.zeros(feat_size, dtype=np.float32)

    keypoints, _ = extract_orb_keypoints_descriptors(gray, nfeatures)
    if not keypoints:
        return zero_feat

    xs = np.array([kp.pt[0] for kp in keypoints], dtype=np.float32)
    ys = np.array([kp.pt[1] for kp in keypoints], dtype=np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    angles = np.array([kp.angle for kp in keypoints], dtype=np.float32)
    sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)

    mean_r, max_r, std_r = _safe_stats(responses)
    mean_a, _, std_a = _safe_stats(angles)
    mean_s, _, std_s = _safe_stats(sizes)
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

    feat = np.concatenate([
        np.array([len(keypoints), mean_r, max_r, std_r, density,
                  mean_a, std_a, mean_s, std_s], dtype=np.float32),
        spatial_hist,
        resp_hist,
        np.array([n_in_mask, ratio_in_mask, n_near_contour], dtype=np.float32),
    ])
    return feat
