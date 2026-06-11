"""
preprocessing.py
----------------
Image preprocessing utilities for the Blood Cell classification pipeline.
Supports loading, resizing, grayscale conversion, CLAHE, and denoising.
"""

import cv2
import numpy as np
from pathlib import Path


def load_cell_image(path: str) -> np.ndarray:
    """
    Load an image from disk in BGR format.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        BGR image (H x W x 3).

    Raises
    ------
    ValueError
        If the image cannot be loaded.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def resize_image(image: np.ndarray, size: tuple = (256, 256)) -> np.ndarray:
    """
    Resize an image to (width, height).

    Parameters
    ----------
    image : np.ndarray
        Input image (any number of channels).
    size : tuple of (width, height)
        Target size.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale uint8.

    Handles: BGR (3-channel), BGRA (4-channel), or already grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Grayscale image (H x W), dtype uint8.
    """
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = normalize_uint8(gray)
    return gray


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to uint8 (0–255).

    Parameters
    ----------
    image : np.ndarray
        Input image (any dtype).

    Returns
    -------
    np.ndarray
        uint8 image.
    """
    if image.dtype == np.uint8:
        return image
    mn, mx = image.min(), image.max()
    if mx == mn:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image.astype(np.float32) - mn) / (mx - mn) * 255.0
    return normalized.clip(0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid : tuple
        Size of grid for histogram equalization.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def denoise_median(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply median blur for salt-and-pepper noise removal.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    ksize : int
        Kernel size (must be odd).

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    return cv2.medianBlur(gray, ksize)


def denoise_gaussian(gray: np.ndarray, ksize: int = 3,
                     sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur for noise removal.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image.
    ksize : int
        Kernel size (must be odd).
    sigma : float
        Gaussian sigma; 0 = auto.

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


def preprocess_cell_image(
    path: str,
    size: tuple = (256, 256),
    use_clahe: bool = True,
    denoise: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single blood-cell image.

    Steps:
      1. Load image (BGR).
      2. Resize to `size`.
      3. Convert to grayscale.
      4. Optionally apply CLAHE.
      5. Optionally denoise.

    Parameters
    ----------
    path : str
        Path to the image file.
    size : tuple of (width, height)
        Target size.
    use_clahe : bool
        Apply CLAHE enhancement.
    denoise : str
        Denoising method: "median", "gaussian", or None.

    Returns
    -------
    tuple of (color_bgr, gray_processed)
        color_bgr  – resized BGR image for color-texture features.
        gray_processed – preprocessed grayscale image for keypoint detection.
    """
    # Load
    color_bgr = load_cell_image(str(path))

    # Resize
    color_bgr = resize_image(color_bgr, size)

    # Grayscale
    gray = to_grayscale(color_bgr)

    # CLAHE
    if use_clahe:
        gray = apply_clahe(gray)

    # Denoise
    if denoise == "median":
        gray = denoise_median(gray)
    elif denoise == "gaussian":
        gray = denoise_gaussian(gray)

    return color_bgr, gray
