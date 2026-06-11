"""
benchmark_keypoints.py
-----------------------
Comparative benchmark of Harris, FAST, and ORB keypoint detectors across
5 transformation cases on a sample of blood cell images.

Metrics evaluated:
  A. Number of Keypoints
  B. Runtime (ms)
  C. Repeatability
  D. Matching Rate (geometric for Harris/FAST, descriptor for ORB)
  E. Registration Error
  F. Cell-region Keypoint Ratio
"""

import sys
import time
import math
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
from transformations import create_transformation_cases, get_rotation_matrix
from feature_harris import detect_harris_corners
from feature_fast import detect_fast_keypoints
from feature_orb import extract_orb_keypoints_descriptors


# -- Constants -----------------------------------------------------------------
MATCH_DIST_THRESH = 5.0   # pixels for geometric matching
MIN_MATCH_HOMOGRAPHY = 4  # minimum matches needed for homography


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -- Detector wrappers ---------------------------------------------------------

def run_harris(gray: np.ndarray, config: dict) -> tuple[list, float]:
    """Run Harris and return (corners_list, runtime_ms)."""
    t0 = time.perf_counter()
    corners = detect_harris_corners(
        gray,
        block_size=config.get("harris_block_size", 2),
        ksize=config.get("harris_ksize", 3),
        k=config.get("harris_k", 0.04),
        threshold_ratio=config.get("harris_threshold_ratio", 0.01),
    )
    rt = (time.perf_counter() - t0) * 1000
    return corners, rt


def run_fast(gray: np.ndarray, config: dict) -> tuple[list, float]:
    """Run FAST and return (keypoints, runtime_ms)."""
    t0 = time.perf_counter()
    kps = detect_fast_keypoints(gray, threshold=config.get("fast_threshold", 20))
    rt = (time.perf_counter() - t0) * 1000
    return kps, rt


def run_orb(gray: np.ndarray, config: dict) -> tuple[list, np.ndarray, float]:
    """Run ORB and return (keypoints, descriptors, runtime_ms)."""
    t0 = time.perf_counter()
    kps, descs = extract_orb_keypoints_descriptors(
        gray, nfeatures=config.get("orb_nfeatures", 500)
    )
    rt = (time.perf_counter() - t0) * 1000
    return kps, descs, rt


# -- Point helpers -------------------------------------------------------------

def corners_to_pts(corners: list) -> np.ndarray:
    """Convert Harris corners [(x,y,r)] to Nx2 float32 array."""
    if not corners:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[c[0], c[1]] for c in corners], dtype=np.float32)


def kps_to_pts(keypoints: list) -> np.ndarray:
    """Convert cv2.KeyPoint list to Nx2 float32 array."""
    if not keypoints:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def transform_pts_rotation(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Transform 2-D points using a 2x3 rotation matrix."""
    if len(pts) == 0:
        return pts
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ pts_h.T).T


# -- Repeatability -------------------------------------------------------------

def compute_repeatability(
    pts_orig: np.ndarray,
    pts_trans: np.ndarray,
    M_rot: np.ndarray = None,
    thresh: float = MATCH_DIST_THRESH,
) -> float:
    """
    Compute repeatability = fraction of original keypoints found again
    in the transformed image (within `thresh` pixels).

    For rotation, pts_orig are transformed geometrically before matching.
    For other cases, direct coordinate comparison is used.

    Parameters
    ----------
    pts_orig : Nx2 array
    pts_trans : Mx2 array
    M_rot : 2x3 rotation matrix or None
    thresh : float
        Maximum pixel distance to count as a match.
    """
    if len(pts_orig) == 0 or len(pts_trans) == 0:
        return 0.0

    if M_rot is not None:
        proj_orig = transform_pts_rotation(pts_orig, M_rot)
    else:
        proj_orig = pts_orig

    matched = 0
    for pt in proj_orig:
        dists = np.linalg.norm(pts_trans - pt, axis=1)
        if dists.min() < thresh:
            matched += 1

    return matched / len(pts_orig)


# -- Matching Rate -------------------------------------------------------------

def geometric_matching_rate(
    pts_orig: np.ndarray,
    pts_trans: np.ndarray,
    M_rot: np.ndarray = None,
    thresh: float = MATCH_DIST_THRESH,
) -> float:
    """
    Geometric matching rate for Harris/FAST (no descriptors).
    Same logic as repeatability — reports fraction of matched points.
    """
    return compute_repeatability(pts_orig, pts_trans, M_rot, thresh)


def orb_descriptor_matching_rate(
    kps_orig: list, descs_orig: np.ndarray,
    kps_trans: list, descs_trans: np.ndarray,
    ratio_test: float = 0.75,
) -> float:
    """
    ORB descriptor matching rate using BFMatcher + ratio test.

    Returns
    -------
    float
        Ratio of good matches to original keypoints.
    """
    if (descs_orig is None or len(descs_orig) == 0 or
            descs_trans is None or len(descs_trans) == 0):
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(descs_orig, descs_trans, k=2)
    except Exception:
        return 0.0

    good = [m for m, n in matches if len([m, n]) == 2 and m.distance < ratio_test * n.distance]
    return len(good) / len(kps_orig) if kps_orig else 0.0


# -- Registration Error --------------------------------------------------------

def registration_error_geometric(
    pts_orig: np.ndarray,
    pts_trans: np.ndarray,
    M_rot: np.ndarray = None,
    thresh: float = MATCH_DIST_THRESH,
) -> float:
    """
    Registration error for Harris/FAST: mean distance between matched point pairs.
    Returns NaN if not enough matches.
    """
    if len(pts_orig) == 0 or len(pts_trans) == 0:
        return float("nan")

    if M_rot is not None:
        proj_orig = transform_pts_rotation(pts_orig, M_rot)
    else:
        proj_orig = pts_orig

    errors = []
    for pt in proj_orig:
        dists = np.linalg.norm(pts_trans - pt, axis=1)
        min_dist = dists.min()
        if min_dist < thresh:
            errors.append(min_dist)

    return float(np.mean(errors)) if errors else float("nan")


def registration_error_orb(
    kps_orig: list, descs_orig: np.ndarray,
    kps_trans: list, descs_trans: np.ndarray,
) -> float:
    """
    Registration error for ORB using Homography/Affine estimation.
    Returns NaN if not enough matches.
    """
    if (descs_orig is None or len(descs_orig) == 0 or
            descs_trans is None or len(descs_trans) == 0):
        return float("nan")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(descs_orig, descs_trans)
    except Exception:
        return float("nan")

    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < MIN_MATCH_HOMOGRAPHY:
        return float("nan")

    pts1 = np.float32([kps_orig[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps_trans[m.trainIdx].pt for m in matches])

    try:
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
        if M is None:
            return float("nan")
        # Compute reprojection error on inliers
        mask = mask.flatten().astype(bool)
        if mask.sum() == 0:
            return float("nan")
        pts1_h = np.hstack([pts1[mask], np.ones((mask.sum(), 1))])
        projected = (M @ pts1_h.T).T
        errors = np.linalg.norm(pts2[mask] - projected, axis=1)
        return float(np.mean(errors))
    except Exception:
        return float("nan")


# -- Cell-region Keypoint Ratio ------------------------------------------------

def cell_region_ratio(pts: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute fraction of keypoints that fall inside the cell mask.
    """
    if len(pts) == 0 or mask is None:
        return 0.0
    h, w = mask.shape[:2]
    count = 0
    for x, y in pts:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        if mask[yi, xi] > 0:
            count += 1
    return count / len(pts)


# -- Per-image benchmark -------------------------------------------------------

def benchmark_image(gray: np.ndarray, mask: np.ndarray, config: dict) -> list[dict]:
    """
    Run the full benchmark for a single image across all 5 transformation cases.

    Returns
    -------
    list of dict
        One dict per (method, case) combination with all metrics.
    """
    cases = create_transformation_cases(gray)
    case_names = list(cases.keys())
    orig_gray = cases["case1_original"]

    # Detect on original
    h_orig, h_rt = run_harris(orig_gray, config)
    f_orig, f_rt = run_fast(orig_gray, config)
    o_orig_kps, o_orig_descs, o_rt = run_orb(orig_gray, config)

    h_pts_orig = corners_to_pts(h_orig)
    f_pts_orig = kps_to_pts(f_orig)
    o_pts_orig = kps_to_pts(o_orig_kps)

    rot_M = get_rotation_matrix(orig_gray, angle=15)
    h, w = orig_gray.shape[:2]
    area = h * w if (h * w) > 0 else 1

    rows = []
    for case_name, trans_img in cases.items():
        # Detect on transformed
        h_trans, _ = run_harris(trans_img, config)
        f_trans, _ = run_fast(trans_img, config)
        o_trans_kps, o_trans_descs, _ = run_orb(trans_img, config)

        h_pts_trans = corners_to_pts(h_trans)
        f_pts_trans = kps_to_pts(f_trans)
        o_pts_trans = kps_to_pts(o_trans_kps)

        # Use rotation matrix only for case2
        M_rot = rot_M if case_name == "case2_rotated" else None

        # -- Harris ---------------------------------------------------------
        h_rep = compute_repeatability(h_pts_orig, h_pts_trans, M_rot)
        h_match = geometric_matching_rate(h_pts_orig, h_pts_trans, M_rot)
        h_reg = registration_error_geometric(h_pts_orig, h_pts_trans, M_rot)
        h_ratio = cell_region_ratio(
            corners_to_pts(h_trans) if case_name != "case1_original" else h_pts_orig,
            mask
        )
        rows.append({
            "method": "Harris",
            "case": case_name,
            "n_keypoints": len(h_trans),
            "runtime_ms": round(h_rt, 3),
            "repeatability": round(h_rep, 4),
            "matching_rate": round(h_match, 4),
            "registration_error": h_reg,
            "cell_region_ratio": round(h_ratio, 4),
            "matching_type": "geometric",
        })

        # -- FAST ------------------------------------------------------------
        f_rep = compute_repeatability(f_pts_orig, f_pts_trans, M_rot)
        f_match = geometric_matching_rate(f_pts_orig, f_pts_trans, M_rot)
        f_reg = registration_error_geometric(f_pts_orig, f_pts_trans, M_rot)
        f_ratio = cell_region_ratio(
            kps_to_pts(f_trans) if case_name != "case1_original" else f_pts_orig,
            mask
        )
        rows.append({
            "method": "FAST",
            "case": case_name,
            "n_keypoints": len(f_trans),
            "runtime_ms": round(f_rt, 3),
            "repeatability": round(f_rep, 4),
            "matching_rate": round(f_match, 4),
            "registration_error": f_reg,
            "cell_region_ratio": round(f_ratio, 4),
            "matching_type": "geometric",
        })

        # -- ORB -------------------------------------------------------------
        o_rep = compute_repeatability(o_pts_orig, o_pts_trans, M_rot)
        o_match = orb_descriptor_matching_rate(
            o_orig_kps, o_orig_descs, o_trans_kps, o_trans_descs
        )
        o_reg = registration_error_orb(
            o_orig_kps, o_orig_descs, o_trans_kps, o_trans_descs
        )
        o_ratio = cell_region_ratio(
            kps_to_pts(o_trans_kps) if case_name != "case1_original" else o_pts_orig,
            mask
        )
        rows.append({
            "method": "ORB",
            "case": case_name,
            "n_keypoints": len(o_trans_kps),
            "runtime_ms": round(o_rt, 3),
            "repeatability": round(o_rep, 4),
            "matching_rate": round(o_match, 4),
            "registration_error": o_reg,
            "cell_region_ratio": round(o_ratio, 4),
            "matching_type": "descriptor",
        })

    return rows


# -- Plotting ------------------------------------------------------------------

def plot_benchmark_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    save_path: str,
) -> None:
    """
    Plot a grouped bar chart comparing methods across transformation cases.
    """
    cases = df["case"].unique()
    methods = ["Harris", "FAST", "ORB"]
    x = np.arange(len(cases))
    width = 0.25
    colors = {"Harris": "#e74c3c", "FAST": "#2ecc71", "ORB": "#3498db"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = []
        for case in cases:
            sub = df[(df["method"] == method) & (df["case"] == case)][metric]
            vals.append(sub.mean() if len(sub) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=method, color=colors[method], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", "\n") for c in cases], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_benchmark(config_path: str = "config.yaml", n_sample: int = 200) -> pd.DataFrame:
    """
    Main benchmark function: samples images, computes metrics, saves results.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    n_sample : int
        Number of images to sample for the benchmark.

    Returns
    -------
    pd.DataFrame
        Full benchmark results table.
    """
    config = load_config(config_path)
    processed_dir = Path(config.get("processed_data_dir", "data/processed"))
    reports_dir = Path(config.get("outputs_dir", "outputs")) / "reports"
    figures_dir = Path(config.get("outputs_dir", "outputs")) / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    img_size = tuple(config.get("image_size", [256, 256]))
    random_state = config.get("random_state", 42)

    # Load metadata
    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        print("[benchmark] metadata.csv not found. Run prepare_dataset.py first.")
        return pd.DataFrame()

    df_meta = pd.read_csv(meta_path)
    # Sample stratified by class
    n_per_class = max(1, n_sample // df_meta["class_name"].nunique())
    sampled = (
        df_meta.groupby("class_name", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=random_state))
        .reset_index(drop=True)
    )
    print(f"[benchmark] Benchmarking on {len(sampled)} images ({len(df_meta)} total)...")

    all_rows = []
    n_errors = 0
    first_error_msg = None

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Benchmark"):
        try:
            color_bgr, gray = preprocess_cell_image(
                row["image_path"], size=img_size, use_clahe=True, denoise="median"
            )
            mask, _, _ = create_cell_mask(color_bgr)
            rows = benchmark_image(gray, mask, config)
            for r in rows:
                r["class_name"] = row["class_name"]
            all_rows.extend(rows)
        except Exception as e:
            n_errors += 1
            if first_error_msg is None:
                first_error_msg = f"{type(e).__name__}: {e}"

    if n_errors > 0:
        print(f"\n[benchmark] {n_errors}/{len(sampled)} images failed. "
              f"First error -> {first_error_msg}")


    if not all_rows:
        print("[benchmark] No results generated.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # -- Save CSV --------------------------------------------------------------
    csv_path = str(reports_dir / "keypoint_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[benchmark] Results saved to: {csv_path}")

    # -- Note on matching type -------------------------------------------------
    print("\n[benchmark] NOTE:")
    print("  Harris/FAST matching_rate = geometric matching rate (position-based).")
    print("  ORB matching_rate = descriptor matching rate (Hamming BFMatcher).")

    # -- Plot metrics ----------------------------------------------------------
    plot_specs = [
        ("n_keypoints",        "Number of Keypoints per Method & Case",
         "# Keypoints",        "keypoints_count_comparison.png"),
        ("runtime_ms",         "Runtime per Method & Case (ms)",
         "Runtime (ms)",       "runtime_comparison.png"),
        ("repeatability",      "Repeatability per Method & Case",
         "Repeatability",      "repeatability_comparison.png"),
        ("matching_rate",      "Matching Rate per Method & Case",
         "Matching Rate",      "matching_rate_comparison.png"),
        ("registration_error", "Registration Error per Method & Case (px)",
         "Error (pixels)",     "registration_error_comparison.png"),
        ("cell_region_ratio",  "Cell-Region Keypoint Ratio per Method & Case",
         "Ratio",              "cell_region_keypoint_ratio.png"),
    ]
    print("\n[benchmark] Generating plots...")
    for metric, title, ylabel, fname in plot_specs:
        if metric in df.columns:
            plot_df = df.copy()
            if metric == "registration_error":
                plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_benchmark_metric(
                plot_df, metric, title, ylabel,
                str(figures_dir / fname)
            )

    # -- Summary table ---------------------------------------------------------
    summary = df.groupby(["method", "case"]).agg({
        "n_keypoints": "mean",
        "runtime_ms": "mean",
        "repeatability": "mean",
        "matching_rate": "mean",
        "registration_error": "mean",
        "cell_region_ratio": "mean",
    }).round(3)
    print("\n[benchmark] Summary:\n")
    print(summary.to_string())

    return df


if __name__ == "__main__":
    run_benchmark()
