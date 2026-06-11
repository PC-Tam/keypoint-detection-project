"""
transformations.py
------------------
Image transformation cases for evaluating keypoint stability.
Used by benchmark_keypoints.py and visualize_keypoints.py.
"""

import cv2
import numpy as np
from pathlib import Path


def rotate_image(image: np.ndarray, angle: float = 15) -> np.ndarray:
    """
    Rotate an image by a given angle around its center.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or BGR).
    angle : float
        Rotation angle in degrees (counter-clockwise).

    Returns
    -------
    np.ndarray
        Rotated image with black borders.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def get_rotation_matrix(image: np.ndarray, angle: float = 15) -> np.ndarray:
    """
    Return the 2x3 rotation matrix for a given image size and angle.

    Parameters
    ----------
    image : np.ndarray
        Reference image to infer dimensions.
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        2x3 affine transformation matrix.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def add_gaussian_noise(image: np.ndarray,
                       mean: float = 0,
                       sigma: float = 10) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    mean : float
        Noise mean.
    sigma : float
        Noise standard deviation.

    Returns
    -------
    np.ndarray
        Noisy image (uint8, clipped to [0, 255]).
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return noisy.clip(0, 255).astype(np.uint8)


def change_brightness_contrast(image: np.ndarray,
                                alpha: float = 1.2,
                                beta: float = 20) -> np.ndarray:
    """
    Adjust brightness and contrast of an image.
    output = alpha * input + beta

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    alpha : float
        Contrast factor (1.0 = no change).
    beta : float
        Brightness additive offset.

    Returns
    -------
    np.ndarray
        Adjusted image (uint8, clipped to [0, 255]).
    """
    adjusted = image.astype(np.float32) * alpha + beta
    return adjusted.clip(0, 255).astype(np.uint8)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    kernel_size : int
        Kernel size (must be odd and positive).

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def create_transformation_cases(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Create all 5 transformation cases for a given image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale uint8 recommended for keypoint analysis).

    Returns
    -------
    dict
        Mapping from case name to transformed image:
          "case1_original"
          "case2_rotated"
          "case3_noisy"
          "case4_brightness"
          "case5_blurred"
    """
    np.random.seed(0)  # reproducibility for noise
    return {
        "case1_original": image.copy(),
        "case2_rotated": rotate_image(image, angle=15),
        "case3_noisy": add_gaussian_noise(image, mean=0, sigma=10),
        "case4_brightness": change_brightness_contrast(image, alpha=1.2, beta=20),
        "case5_blurred": apply_gaussian_blur(image, kernel_size=5),
    }


def save_transformation_cases(image: np.ndarray,
                               save_dir: str,
                               prefix: str = "sample") -> None:
    """
    Save all transformation case images to a directory.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    save_dir : str
        Output directory path.
    prefix : str
        Filename prefix (e.g., class name or image stem).
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = create_transformation_cases(image)
    for case_name, case_img in cases.items():
        out_path = output_dir / f"{prefix}_{case_name}.png"
        cv2.imwrite(str(out_path), case_img)

    print(f"[transformations] Saved {len(cases)} cases to: {output_dir}")
