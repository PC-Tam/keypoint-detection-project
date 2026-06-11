"""
feature_engineering.py
-----------------------
Assembles and saves all feature sets for the Blood Cell classification pipeline.
Computes Harris, FAST, ORB (statistical + BoVW), morphology, and color/texture
features, then creates multiple named feature bundles for ML comparison.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ensure src/ is on the path when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import preprocess_cell_image
from segmentation_classical import create_cell_mask
from feature_harris import extract_harris_features
from feature_fast import extract_fast_features
from feature_orb import extract_orb_keypoints_descriptors, extract_orb_statistical_features
from feature_morphology import extract_morphology_features
from feature_color_texture import extract_color_texture_features
from bovw import compute_bovw_histogram, load_bovw_model


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_all_for_row(
    row: pd.Series,
    config: dict,
    kmeans=None,
) -> dict:
    """
    Extract all features for a single image row.

    Returns a dict with keys:
      harris, fast, orb_stats, orb_desc, morph, color_tex
    """
    path = row["image_path"]
    img_size = tuple(config.get("image_size", [256, 256]))
    orb_n = config.get("orb_nfeatures", 500)
    grid_size = tuple(config.get("grid_size", [4, 4]))
    resp_bins = config.get("response_hist_bins", 10)
    harris_bs = config.get("harris_block_size", 2)
    harris_ks = config.get("harris_ksize", 3)
    harris_k = config.get("harris_k", 0.04)
    harris_tr = config.get("harris_threshold_ratio", 0.01)
    fast_thresh = config.get("fast_threshold", 20)

    try:
        color_bgr, gray = preprocess_cell_image(
            path, size=img_size, use_clahe=True, denoise="median"
        )
        mask, bbox, contour = create_cell_mask(color_bgr)

        h_feat = extract_harris_features(
            gray, mask=mask, contour=contour,
            block_size=harris_bs, ksize=harris_ks, k=harris_k,
            threshold_ratio=harris_tr,
            grid_size=grid_size, response_hist_bins=resp_bins,
        )
        f_feat = extract_fast_features(
            gray, mask=mask, contour=contour,
            threshold=fast_thresh, grid_size=grid_size,
            response_hist_bins=resp_bins,
        )
        o_stat = extract_orb_statistical_features(
            gray, mask=mask, contour=contour,
            nfeatures=orb_n, grid_size=grid_size,
            response_hist_bins=resp_bins,
        )
        _, o_desc = extract_orb_keypoints_descriptors(gray, nfeatures=orb_n)
        m_feat = extract_morphology_features(mask)
        ct_feat = extract_color_texture_features(color_bgr, gray, mask=mask)

        # BoVW for ORB
        o_bovw = np.zeros(config.get("bovw_clusters", 100), dtype=np.float32)
        if kmeans is not None:
            o_bovw = compute_bovw_histogram(o_desc, kmeans)

    except Exception as e:
        n_h = 5 + grid_size[0] * grid_size[1] + resp_bins + 3
        n_f = n_h
        n_o = 9 + grid_size[0] * grid_size[1] + resp_bins + 3
        n_bovw = config.get("bovw_clusters", 100)
        h_feat = np.zeros(n_h, dtype=np.float32)
        f_feat = np.zeros(n_f, dtype=np.float32)
        o_stat = np.zeros(n_o, dtype=np.float32)
        o_desc = np.empty((0, 32), dtype=np.uint8)
        o_bovw = np.zeros(n_bovw, dtype=np.float32)
        m_feat = np.zeros(23, dtype=np.float32)
        ct_feat = np.zeros(88, dtype=np.float32)

    return {
        "harris": h_feat,
        "fast": f_feat,
        "orb_stats": o_stat,
        "orb_desc": o_desc,
        "orb_bovw": o_bovw,
        "morph": m_feat,
        "color_tex": ct_feat,
    }


def extract_split_features(
    df: pd.DataFrame,
    config: dict,
    split_name: str,
    kmeans=None,
    limit: int = None,
) -> dict:
    """
    Extract features for all images in a dataframe split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'image_path' and 'label' columns.
    config : dict
        Project configuration.
    split_name : str
        Name of the split (train/val/test) for logging.
    kmeans : MiniBatchKMeans or None
        Fitted BoVW vocabulary (None during train descriptor collection).
    limit : int or None
        Limit number of images for fast testing.

    Returns
    -------
    dict
        Keys: "harris", "fast", "orb_stats", "orb_descs", "orb_bovw",
              "morph", "color_tex", "labels".
    """
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)

    all_harris, all_fast, all_orb_stats = [], [], []
    all_orb_descs, all_orb_bovw = [], []
    all_morph, all_color_tex = [], []
    labels = []

    print(f"\n[feature_engineering] Extracting features for '{split_name}' split "
          f"({len(df)} images)...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        feats = _extract_all_for_row(row, config, kmeans)
        all_harris.append(feats["harris"])
        all_fast.append(feats["fast"])
        all_orb_stats.append(feats["orb_stats"])
        all_orb_descs.append(feats["orb_desc"])
        all_orb_bovw.append(feats["orb_bovw"])
        all_morph.append(feats["morph"])
        all_color_tex.append(feats["color_tex"])
        labels.append(row["label"])

    return {
        "harris": np.array(all_harris, dtype=np.float32),
        "fast": np.array(all_fast, dtype=np.float32),
        "orb_stats": np.array(all_orb_stats, dtype=np.float32),
        "orb_descs": all_orb_descs,  # list of arrays (for BoVW fitting)
        "orb_bovw": np.array(all_orb_bovw, dtype=np.float32),
        "morph": np.array(all_morph, dtype=np.float32),
        "color_tex": np.array(all_color_tex, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
    }


def build_feature_bundles(split_data: dict) -> dict:
    """
    Assemble named feature bundles from split feature data.

    Parameters
    ----------
    split_data : dict
        Output of extract_split_features.

    Returns
    -------
    dict
        Mapping from bundle name to feature matrix (np.ndarray).
    """
    H = split_data["harris"]
    F = split_data["fast"]
    O = split_data["orb_stats"]
    B = split_data["orb_bovw"]
    M = split_data["morph"]
    C = split_data["color_tex"]

    return {
        "harris_only":            H,
        "fast_only":              F,
        "orb_stats_only":         O,
        "orb_bovw":               B,
        "harris_morphology":      np.hstack([H, M]),
        "fast_morphology":        np.hstack([F, M]),
        "orb_bovw_morphology":    np.hstack([B, M]),
        "combined_harris_fast_orb": np.hstack([H, F, O]),
        "combined_all_traditional": np.hstack([H, F, O, B, M, C]),
    }


def save_features(features_dir: Path, split: str, bundle_name: str,
                  X: np.ndarray, y: np.ndarray) -> None:
    """Save feature matrix and labels to .npy files."""
    out_dir = features_dir / bundle_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / f"X_{split}.npy"), X)
    np.save(str(out_dir / f"y_{split}.npy"), y)


def run_feature_engineering(config_path: str = "config.yaml") -> None:
    """
    Main function: extract all features and save them for all splits.

    Workflow:
      1. Load train/val/test metadata CSVs.
      2. Extract all raw features for train set (without BoVW).
      3. Fit BoVW vocabulary on train ORB descriptors.
      4. Recompute BoVW histograms for train, then do val and test.
      5. Build named feature bundles.
      6. Save .npy files.
    """
    config = load_config(config_path)
    features_dir = Path(config.get("features_dir", "data/features"))
    processed_dir = Path(config.get("processed_data_dir", "data/processed"))
    models_dir = Path(config.get("outputs_dir", "outputs")) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    limit = config.get("test_mode_limit")

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    # -- Step 1: Extract train features (BoVW = None for descriptor collection) -
    train_data = extract_split_features(train_df, config, "train", kmeans=None,
                                        limit=limit)

    # -- Step 2: Fit BoVW vocabulary from train descriptors --------------------
    from bovw import train_bovw_vocabulary
    bovw_path = str(models_dir / "orb_bovw_kmeans.joblib")
    n_clusters = config.get("bovw_clusters", 100)
    kmeans = train_bovw_vocabulary(
        train_data["orb_descs"],
        n_clusters=n_clusters,
        random_state=config.get("random_state", 42),
        save_path=bovw_path,
    )

    # -- Step 3: Recompute BoVW histograms for train ---------------------------
    print("[feature_engineering] Computing BoVW histograms for train...")
    train_data["orb_bovw"] = np.array([
        compute_bovw_histogram(d, kmeans) for d in train_data["orb_descs"]
    ], dtype=np.float32)

    # -- Step 4: Extract val and test features ---------------------------------
    val_data = extract_split_features(val_df, config, "val", kmeans=kmeans,
                                      limit=limit)
    test_data = extract_split_features(test_df, config, "test", kmeans=kmeans,
                                       limit=limit)

    # -- Step 5: Build and save feature bundles --------------------------------
    splits = {"train": train_data, "val": val_data, "test": test_data}
    for split_name, split_data in splits.items():
        bundles = build_feature_bundles(split_data)
        y = split_data["labels"]
        for bundle_name, X in bundles.items():
            save_features(features_dir, split_name, bundle_name, X, y)
            print(f"  Saved: {bundle_name}/{split_name}  shape={X.shape}")

    # -- Also save BoVW features separately -----------------------------------
    bovw_dir = features_dir / "orb_bovw_raw"
    bovw_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in splits.items():
        np.save(str(bovw_dir / f"orb_bovw_{split_name}.npy"), split_data["orb_bovw"])

    print("\n[feature_engineering] All features extracted and saved.")


if __name__ == "__main__":
    run_feature_engineering()
