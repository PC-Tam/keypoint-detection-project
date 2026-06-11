import cv2
import numpy as np

def extract_fast_features(image, threshold=20, nonmax_suppression=True):
    """
    Extract a fixed-size feature vector using FAST keypoint detection.
    Args:
        image: Preprocessed grayscale image.
        threshold: Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.
        nonmax_suppression: If true, non-maximum suppression is applied to detected corners.
    Returns:
        feature_vector: A 1D numpy array of length 21 (5 global stats + 16 spatial hist).
    """
    # Initialize FAST object
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
    
    # Detect keypoints
    kp = fast.detect(image, None)
    
    num_kp = len(kp)
    if num_kp == 0:
        return np.zeros(21)
        
    # Extract responses
    responses = np.array([p.response for p in kp])
    
    # 1. Global Statistics (5 features)
    mean_resp = np.mean(responses)
    max_resp = np.max(responses)
    std_resp = np.std(responses)
    density = num_kp / (image.shape[0] * image.shape[1])
    
    global_stats = np.array([num_kp, mean_resp, max_resp, std_resp, density])
    
    # 2. Spatial Distribution Histogram (4x4 grid = 16 features)
    h, w = image.shape
    grid_h = h // 4
    grid_w = w // 4
    
    spatial_hist = np.zeros(16)
    for p in kp:
        x, y = int(p.pt[0]), int(p.pt[1])
        grid_y = min(y // grid_h, 3)
        grid_x = min(x // grid_w, 3)
        idx = grid_y * 4 + grid_x
        spatial_hist[idx] += 1
        
    # Normalize the spatial histogram
    if num_kp > 0:
        spatial_hist = spatial_hist / num_kp
        
    feature_vector = np.concatenate([global_stats, spatial_hist])
    return feature_vector

def draw_fast_keypoints(image, threshold=20, nonmax_suppression=True):
    """
    Detect and draw FAST keypoints on the image for visualization.
    Returns a BGR image with drawn keypoints.
    """
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_bgr = image.copy()
    else:
        gray = image
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    kp = fast.detect(gray, None)
    img_bgr = cv2.drawKeypoints(img_bgr, kp, None, color=(255, 0, 0)) # Blue circles
    
    return img_bgr
