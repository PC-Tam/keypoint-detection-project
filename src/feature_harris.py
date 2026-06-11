"""
feature_harris.py
-----------------
Harris Corner Detection feature extraction for blood cell classification.
Provides both visualization utilities and a fixed-length feature vector.
"""

import cv2
import numpy as np


# -- Internal helpers ---------------------------------------------------------

def _safe_stats(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, max, std) for an array; returns (0,0,0) if empty."""
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(arr)), float(np.max(arr)), float(np.std(arr))


def detect_harris_corners(
    gray: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
) -> list[tuple[int, int, float]]:
    """
    Detect Harris corners in a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    block_size : int
        Neighbourhood size for corner detection.
    ksize : int
        Aperture parameter for the Sobel operator.
    k : float
        Harris detector free parameter.
    threshold_ratio : float
        Fraction of maximum response to use as threshold.

    Returns
    -------
    list of (x, y, response)
        Detected corner coordinates and their response values.
    """
    gray_f = np.float32(gray)
    response = cv2.cornerHarris(gray_f, block_size, ksize, k)

    thresh = threshold_ratio * response.max()
    corners_mask = response > thresh
    ys, xs = np.where(corners_mask)

    return [(int(x), int(y), float(response[y, x])) for x, y in zip(xs, ys)]


def draw_harris_corners(
    image: np.ndarray,
    corners: list,
    color: tuple = (0, 0, 255),
    radius: int = 3,
) -> np.ndarray:
    """
    Draw Harris corners on an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image.
    corners : list of (x, y, response)
        Corners to draw.
    color : tuple
        BGR color for circles.
    radius : int
        Circle radius.

    Returns
    -------
    np.ndarray
        BGR image with corners drawn.
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for (x, y, _) in corners:
        cv2.circle(vis, (x, y), radius, color, -1)
    return vis


def extract_harris_features(
    gray: np.ndarray,
    mask: np.ndarray = None,
    contour=None,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
    grid_size: tuple = (4, 4),
    response_hist_bins: int = 10,
) -> np.ndarray:
    """
    Extract a fixed-length Harris feature vector from a grayscale image.

    Feature layout (total = 5 + grid_rows*grid_cols + response_hist_bins + 3):
      [0]   number_of_keypoints
      [1]   mean_response
      [2]   max_response
      [3]   std_response
      [4]   keypoint_density  (keypoints / image_area)
      [5 .. 5+G*G-1]   spatial histogram (grid_size[0] x grid_size[1] cells)
      [5+G*G .. 5+G*G+B-1]  response histogram (response_hist_bins bins)
      [-3]  n_keypoints_in_mask
      [-2]  ratio_keypoints_in_mask
      [-1]  n_keypoints_near_contour

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    mask : np.ndarray or None
        Binary mask (0/255) for the cell region.
    contour : array or None
        Cell contour for proximity analysis.
    block_size, ksize, k, threshold_ratio : Harris parameters.
    grid_size : tuple of (rows, cols) for spatial histogram.
    response_hist_bins : int
        Number of bins for response histogram.

    Returns
    -------
    np.ndarray
        1-D feature vector (float32).
    """
    h, w = gray.shape[:2]
    n_grid = grid_size[0] * grid_size[1]
    feat_size = 5 + n_grid + response_hist_bins + 3
    zero_feat = np.zeros(feat_size, dtype=np.float32)

    corners = detect_harris_corners(gray, block_size, ksize, k, threshold_ratio)

    if not corners:
        return zero_feat

    xs = np.array([c[0] for c in corners], dtype=np.float32)
    ys = np.array([c[1] for c in corners], dtype=np.float32)
    responses = np.array([c[2] for c in corners], dtype=np.float32)

    mean_r, max_r, std_r = _safe_stats(responses)
    density = len(corners) / (h * w) if (h * w) > 0 else 0.0

    # -- Spatial grid histogram -----------------------------------------------
    cell_h = h / grid_size[0]
    cell_w = w / grid_size[1]
    spatial_hist = np.zeros(n_grid, dtype=np.float32)
    for x, y in zip(xs, ys):
        row = min(int(y / cell_h), grid_size[0] - 1)
        col = min(int(x / cell_w), grid_size[1] - 1)
        spatial_hist[row * grid_size[1] + col] += 1
    # Normalize
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
    ratio_in_mask = n_in_mask / len(corners) if corners else 0.0

    # -- Near-contour keypoints ------------------------------------------------
    n_near_contour = 0
    if contour is not None and len(contour) > 0:
        for x, y in zip(xs.astype(int), ys.astype(int)):
            dist = abs(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
            if dist < 5.0:
                n_near_contour += 1

    # -- Assemble feature vector -----------------------------------------------
    feat = np.concatenate([
        np.array([len(corners), mean_r, max_r, std_r, density], dtype=np.float32),
        spatial_hist,
        resp_hist,
        np.array([n_in_mask, ratio_in_mask, n_near_contour], dtype=np.float32),
    ])
    return feat
