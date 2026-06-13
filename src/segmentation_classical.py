"""
segmentation_classical.py
--------------------------
Classical (non-deep-learning) segmentation of blood cells from background.
Uses Otsu thresholding, HSV masking, and morphological operations.
"""

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Clean a binary mask using morphological open then close.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8, values 0 or 255).
    kernel_size : int
        Structuring element size.

    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def get_largest_contour(mask: np.ndarray):
    """
    Find the largest contour in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8).

    Returns
    -------
    contour or None
        The largest contour, or None if no contours found.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def segment_cell_otsu(image: np.ndarray) -> np.ndarray:
    """
    Segment the cell region using Otsu thresholding on grayscale.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale image.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0/255), white = cell region.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu threshold on inverted image (cells usually darker on bright background,
    # or brighter, so we try both and keep larger mask)
    _, mask_normal = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, mask_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    mask_normal_clean = clean_mask(mask_normal)
    mask_inv_clean = clean_mask(mask_inv)

    # To decide which mask is the cell (cells are often darker or brighter),
    # we look at the center of the image, because the cell is usually centered,
    # while the background dominates the edges.
    h, w = gray.shape
    cy1, cy2 = int(h * 0.25), int(h * 0.75)
    cx1, cx2 = int(w * 0.25), int(w * 0.75)
    
    center_normal = np.sum(mask_normal_clean[cy1:cy2, cx1:cx2])
    center_inv = np.sum(mask_inv_clean[cy1:cy2, cx1:cx2])

    if center_normal >= center_inv:
        return mask_normal_clean
    else:
        return mask_inv_clean


def segment_cell_hsv(image: np.ndarray) -> np.ndarray:
    """
    Segment the cell region using HSV color thresholding.
    Targets the purplish/pink hues typical of Giemsa-stained blood cells.

    Parameters
    ----------
    image : np.ndarray
        BGR image.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0/255), white = cell region.
    """
    if image.ndim != 3:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Pinkish/purplish Giemsa stain range
    lower1 = np.array([120, 30, 50])
    upper1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Reddish range (some cells)
    lower2 = np.array([0, 30, 50])
    upper2 = np.array([20, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # Light blue/cyan (some staining artifacts)
    lower3 = np.array([85, 20, 50])
    upper3 = np.array([130, 255, 255])
    mask3 = cv2.inRange(hsv, lower3, upper3)

    combined = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
    return clean_mask(combined)


def extract_cell_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to extract the cell region from an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    mask : np.ndarray
        Binary mask (uint8, 0/255).

    Returns
    -------
    np.ndarray
        Image with background set to 0 (masked out).
    """
    if image.ndim == 3:
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask
        return cv2.bitwise_and(image, mask3)
    else:
        return cv2.bitwise_and(image, mask)


def create_cell_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Create a binary cell mask using combined Otsu + HSV segmentation.
    Falls back to a full-image mask if segmentation fails.

    Parameters
    ----------
    image : np.ndarray
        BGR image (ideally after resizing/preprocessing).

    Returns
    -------
    tuple of (mask, bbox, contour)
        mask    – binary mask (uint8, 0/255)
        bbox    – bounding box (x, y, w, h) or None
        contour – largest contour or None
    """
    h, w = image.shape[:2]
    fallback_mask = np.ones((h, w), dtype=np.uint8) * 255

    try:
        mask_otsu = segment_cell_otsu(image)
        
        # The HSV mask can sometimes be too greedy and capture the background.
        # Since Otsu is robust for blood cells, we'll use it as the primary mask.
        mask_combined = mask_otsu

        # Use the contour-based mask to refine
        contour = get_largest_contour(mask_combined)
        if contour is None or cv2.contourArea(contour) < 100:
            # Fall back to Otsu only
            contour = get_largest_contour(mask_otsu)

        if contour is None or cv2.contourArea(contour) < 100:
            return fallback_mask, None, None

        # Draw mask from largest contour
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(final_mask, [contour], -1, 255, cv2.FILLED)
        final_mask = clean_mask(final_mask, kernel_size=5)

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = (x, y, bw, bh)

        return final_mask, bbox, contour

    except Exception as e:
        # Graceful fallback
        return fallback_mask, None, None
