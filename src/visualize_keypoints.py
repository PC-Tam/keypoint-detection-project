import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from preprocessing import load_image, extract_green_channel, apply_clahe
from feature_harris import draw_harris_corners
from feature_fast import draw_fast_keypoints
from feature_orb import draw_orb_keypoints

def visualize_preprocessing_and_features(image_path, save_name="visualization.png"):
    """
    Generate a plot showing the original image, preprocessing steps, and extracted keypoints.
    """
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    
    # Load original
    orig_bgr = load_image(image_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    green = extract_green_channel(orig_bgr)
    clahe = apply_clahe(green)
    
    # Keypoints
    harris_img = draw_harris_corners(clahe)
    harris_rgb = cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB)
    
    fast_img = draw_fast_keypoints(clahe)
    fast_rgb = cv2.cvtColor(fast_img, cv2.COLOR_BGR2RGB)
    
    orb_img = draw_orb_keypoints(clahe)
    orb_rgb = cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB)
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    titles = ["Original", "Green Channel", "CLAHE", "Harris Corners", "FAST Keypoints", "ORB Keypoints"]
    images = [orig_rgb, green, clahe, harris_rgb, fast_rgb, orb_rgb]
    cmaps = [None, 'gray', 'gray', None, None, None]
    
    for i in range(6):
        if cmaps[i]:
            axes[i].imshow(images[i], cmap=cmaps[i])
        else:
            axes[i].imshow(images[i])
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        
    plt.tight_layout()
    save_path = f"outputs/figures/{save_name}"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # Example usage if run directly (requires an image path)
    import sys
    if len(sys.argv) > 1:
        visualize_preprocessing_and_features(sys.argv[1])
    else:
        print("Please provide an image path. e.g., python visualize_keypoints.py data/raw/ACRIMA/images/sample.jpg")
