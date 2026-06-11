import cv2
import numpy as np

def extract_orb_descriptors(image, nfeatures=500):
    """
    Extract ORB keypoints and descriptors from an image.
    Args:
        image: Preprocessed grayscale image.
        nfeatures: Maximum number of features to retain.
    Returns:
        kp: List of keypoints.
        des: Numpy array of shape (num_keypoints, 32) containing descriptors.
             Returns empty array if no descriptors found.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(image, None)
    
    if des is None:
        return kp, np.array([])
        
    return kp, des

def draw_orb_keypoints(image, nfeatures=500):
    """
    Detect and draw ORB keypoints on the image for visualization.
    Returns a BGR image with drawn keypoints.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_bgr = image.copy()
    else:
        gray = image
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    kp, _ = orb.detectAndCompute(gray, None)
    img_bgr = cv2.drawKeypoints(img_bgr, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Green circles
    
    return img_bgr
