import cv2
import numpy as np

def extract_harris_features(image, block_size=2, ksize=3, k=0.04):
    """
    Extract a fixed-size feature vector using Harris Corner Detection.
    Args:
        image: Preprocessed grayscale image.
        block_size: Neighborhood size.
        ksize: Aperture parameter for the Sobel operator.
        k: Harris detector free parameter.
    Returns:
        feature_vector: A 1D numpy array of length 21 (5 global stats + 16 spatial hist).
    """
    # Compute Harris response
    # cv2.cornerHarris requires float32 input
    dst = cv2.cornerHarris(np.float32(image), block_size, ksize, k)
    
    # Threshold for an optimal value, it may vary depending on the image.
    # We use a threshold relative to the max response
    threshold = 0.01 * dst.max()
    corners = np.argwhere(dst > threshold)
    
    # 1. Global Statistics (5 features)
    num_corners = len(corners)
    
    if num_corners == 0:
        return np.zeros(21)
    
    # Get the response values of the detected corners
    corner_responses = dst[dst > threshold]
    
    mean_resp = np.mean(corner_responses)
    max_resp = np.max(corner_responses)
    std_resp = np.std(corner_responses)
    
    # Density: corners per pixel
    density = num_corners / (image.shape[0] * image.shape[1])
    
    global_stats = np.array([num_corners, mean_resp, max_resp, std_resp, density])
    
    # 2. Spatial Distribution Histogram (4x4 grid = 16 features)
    h, w = image.shape
    grid_h = h // 4
    grid_w = w // 4
    
    spatial_hist = np.zeros(16)
    for y, x in corners:
        grid_y = min(y // grid_h, 3)
        grid_x = min(x // grid_w, 3)
        idx = grid_y * 4 + grid_x
        spatial_hist[idx] += 1
        
    # Normalize the spatial histogram
    if num_corners > 0:
        spatial_hist = spatial_hist / num_corners
        
    feature_vector = np.concatenate([global_stats, spatial_hist])
    return feature_vector

def draw_harris_corners(image, block_size=2, ksize=3, k=0.04):
    """
    Detect and draw Harris corners on the image for visualization.
    Returns a BGR image with drawn corners.
    """
    if len(image.shape) == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image
    else:
        img_bgr = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    dst = cv2.cornerHarris(np.float32(gray), block_size, ksize, k)
    dst = cv2.dilate(dst, None) # Dilate to make corners more visible
    
    img_bgr[dst > 0.01 * dst.max()] = [0, 0, 255] # Red dots
    
    return img_bgr
