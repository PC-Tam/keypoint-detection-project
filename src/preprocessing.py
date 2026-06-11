import cv2
import numpy as np

def load_image(path):
    """Load an image from the given path."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image at {path}")
    return image

def resize_image(image, size=(400, 400)):
    """Resize the image to the specified size."""
    return cv2.resize(image, size)

def extract_green_channel(image):
    """
    Extract the green channel from an BGR image.
    In fundus images, the green channel usually provides the best contrast 
    for detecting vessels and the optic disc.
    """
    # OpenCV loads images in BGR format
    b, g, r = cv2.split(image)
    return g

def apply_clahe(gray_image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances the local contrast of the image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def preprocess_fundus_image(path, size=(400, 400), use_green=True, use_clahe=True):
    """
    Full preprocessing pipeline for a single fundus image.
    Returns the preprocessed grayscale image.
    """
    # Load
    image = load_image(path)
    
    # Resize
    image = resize_image(image, size)
    
    # Convert to grayscale or extract green channel
    if use_green:
        gray = extract_green_channel(image)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Apply CLAHE
    if use_clahe:
        processed_image = apply_clahe(gray)
    else:
        processed_image = gray
        
    return processed_image
