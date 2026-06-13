"""
feature_morphology.py
---------------------
Classical morphological feature extraction from cell segmentation masks.
Does NOT use deep learning. Features are derived from shape, moments, and
contour geometry.
"""

import cv2
import numpy as np
import math


def extract_morphology_features(
    mask: np.ndarray,
    image: np.ndarray = None,
) -> np.ndarray:
    """
    Extract morphological features from a binary cell mask.

    Feature layout (fixed length = 22):
      [0]  area
      [1]  perimeter
      [2]  circularity
      [3]  eccentricity
      [4]  aspect_ratio (bounding box w/h)
      [5]  extent (area / bbox_area)
      [6]  solidity (area / convex_hull_area)
      [7]  equivalent_diameter
      [8]  bbox_x (normalized)
      [9]  bbox_y (normalized)
      [10] bbox_w (normalized)
      [11] bbox_h (normalized)
      [12..18] Hu moments (7 values, log-transformed)
      [19] centroid_x (normalized)
      [20] centroid_y (normalized)
      [21] mean_dist_contour_to_centroid
      [22] std_dist_contour_to_centroid  <- total = 23

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8, 0/255).
    image : np.ndarray or None
        Not used directly, kept for API consistency.

    Returns
    -------
    np.ndarray
        1-D float32 feature vector of length 23.
    """
    feat_size = 23
    zero_feat = np.zeros(feat_size, dtype=np.float32)
    h, w = mask.shape[:2]

    try:
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return zero_feat

        # Largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 10:
            return zero_feat

        perimeter = cv2.arcLength(contour, True)

        # -- Basic shape features ---------------------------------------------
        circularity = (4 * math.pi * area / (perimeter ** 2)
                       if perimeter > 0 else 0.0)

        # Eccentricity via fitted ellipse
        eccentricity = 0.0
        if len(contour) >= 5:
            try:
                (cx, cy), (ma, mb), angle = cv2.fitEllipse(contour)
                # ma = major axis length, mb = minor axis length
                a = max(ma, mb) / 2.0
                b = min(ma, mb) / 2.0
                if a > 0:
                    eccentricity = math.sqrt(1 - (b / a) ** 2)
            except Exception:
                eccentricity = 0.0

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = bw / bh if bh > 0 else 0.0
        bbox_area = bw * bh
        extent = area / bbox_area if bbox_area > 0 else 0.0

        # Convex hull / solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        equiv_diam = math.sqrt(4 * area / math.pi) if area > 0 else 0.0

        # -- Moments & Hu moments ---------------------------------------------
        M = cv2.moments(contour)
        centroid_x = M["m10"] / M["m00"] if M["m00"] != 0 else w / 2.0
        centroid_y = M["m01"] / M["m00"] if M["m00"] != 0 else h / 2.0

        hu = cv2.HuMoments(M).flatten()
        # Log-transform Hu moments (standard practice for stability)
        hu_log = np.array([
            -np.sign(v) * np.log10(abs(v) + 1e-10) for v in hu
        ], dtype=np.float32)

        # -- Distance from contour points to centroid -------------------------
        contour_pts = contour.reshape(-1, 2).astype(np.float32)
        dists = np.sqrt(
            (contour_pts[:, 0] - centroid_x) ** 2 +
            (contour_pts[:, 1] - centroid_y) ** 2
        )
        mean_dist = float(np.mean(dists))
        std_dist = float(np.std(dists))

        feat = np.array([
            area,
            perimeter,
            circularity,
            eccentricity,
            aspect_ratio,
            extent,
            solidity,
            equiv_diam,
            x / w, y / h, bw / w, bh / h,   # normalized bbox
        ] + list(hu_log) + [
            centroid_x / w,
            centroid_y / h,
            mean_dist,
            std_dist,
        ], dtype=np.float32)

        return feat

    except Exception:
        return zero_feat
