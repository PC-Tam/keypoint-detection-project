"""
visualize_keypoints.py
-----------------------
Generate visual comparisons of Harris, FAST, and ORB keypoints
on sample blood cell images from each class.
Saves full grid visualizations and individual panel images.
"""

import sys
import yaml
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import preprocess_cell_image
from segmentation_classical import create_cell_mask
from transformations import create_transformation_cases
from feature_harris import detect_harris_corners, draw_harris_corners
from feature_fast import detect_fast_keypoints, draw_fast_keypoints
from feature_orb import extract_orb_keypoints_descriptors, draw_orb_keypoints


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for matplotlib display."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_single_image(
    image_path: str,
    class_name: str,
    config: dict,
    save_dir: Path,
) -> None:
    """
    Create a full visualization panel for one image:
      Row 1: Original | Preprocessed Gray | Cell Mask
      Row 2: Harris   | FAST              | ORB
      Row 3: Case2(Rotate) | Case3(Noise) | Case4(Brightness) | Case5(Blur)

    Parameters
    ----------
    image_path : str
        Path to the blood cell image.
    class_name : str
        Cell class label (used in figure title and filename).
    config : dict
        Project configuration.
    save_dir : Path
        Directory to save the visualization.
    """
    img_size = tuple(config.get("image_size", [256, 256]))
    orb_n = config.get("orb_nfeatures", 500)
    harris_bs = config.get("harris_block_size", 2)
    harris_ks = config.get("harris_ksize", 3)
    harris_k = config.get("harris_k", 0.04)
    harris_tr = config.get("harris_threshold_ratio", 0.01)
    fast_thresh = config.get("fast_threshold", 20)

    try:
        color_bgr, gray = preprocess_cell_image(
            image_path, size=img_size, use_clahe=True, denoise="median"
        )
        mask, bbox, contour = create_cell_mask(color_bgr)

        # -- Keypoint detection --------------------------------------------
        h_corners = detect_harris_corners(gray, harris_bs, harris_ks, harris_k, harris_tr)
        h_vis = draw_harris_corners(gray, h_corners)

        f_kps = detect_fast_keypoints(gray, threshold=fast_thresh)
        f_vis = draw_fast_keypoints(gray, f_kps)

        o_kps, o_descs = extract_orb_keypoints_descriptors(gray, nfeatures=orb_n)
        o_vis = draw_orb_keypoints(gray, o_kps)

        # -- Transformation cases ------------------------------------------
        cases = create_transformation_cases(gray)

        # -- Build figure --------------------------------------------------
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"Blood Cell Keypoint Visualization — Class: {class_name}",
            fontsize=13, fontweight="bold"
        )

        panel_images = [
            (bgr_to_rgb(color_bgr), "Original (Color)", None),
            (gray, "Preprocessed Gray", "gray"),
            (mask, "Cell Mask (Otsu+HSV)", "gray"),
            (bgr_to_rgb(h_vis), f"Harris ({len(h_corners)} corners)", None),
            (bgr_to_rgb(f_vis), f"FAST ({len(f_kps)} kps)", None),
            (bgr_to_rgb(o_vis), f"ORB ({len(o_kps)} kps)", None),
            (cases["case2_rotated"], "Rotate 15°", "gray"),
            (cases["case3_noisy"], "Gaussian Noise", "gray"),
            (cases["case4_brightness"], "Brightness/Contrast", "gray"),
            (cases["case5_blurred"], "Gaussian Blur", "gray"),
        ]

        for idx, (img, title, cmap) in enumerate(panel_images):
            ax = fig.add_subplot(2, 5, idx + 1)
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        stem = Path(image_path).stem[:20]
        save_path = save_dir / f"{class_name}__{stem}.png"
        plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  [visualize] Skipped {image_path}: {e}")


def run_visualization(
    config_path: str = "config.yaml",
    n_per_class: int = 3,
) -> None:
    """
    Main function: select sample images from each class and generate visualizations.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    n_per_class : int
        Number of images to visualize per class.
    """
    config = load_config(config_path)
    processed_dir = Path(config.get("processed_data_dir", "data/processed"))
    figures_dir = (
        Path(config.get("outputs_dir", "outputs")) /
        "figures" / "sample_visualizations"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        print("[visualize] metadata.csv not found. Run prepare_dataset.py first.")
        return

    df = pd.read_csv(meta_path)
    classes = sorted(df["class_name"].unique())
    random_state = config.get("random_state", 42)

    print(f"[visualize] Generating {n_per_class} visualization(s) per class "
          f"for {len(classes)} classes...")

    for cls in classes:
        cls_df = df[df["class_name"] == cls]
        sample = cls_df.sample(
            n=min(n_per_class, len(cls_df)),
            random_state=random_state,
        )
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=cls):
            visualize_single_image(
                row["image_path"], cls, config, figures_dir
            )

    print(f"\n[visualize] All visualizations saved to: {figures_dir}")


if __name__ == "__main__":
    run_visualization()
