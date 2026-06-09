"""
Evaluation Module — src/evaluation.py
======================================
Đánh giá hiệu năng ORB Matching trên dataset.
Hỗ trợ:
  - Evaluate single pair (template + test)
  - Evaluate dataset (batch)
  - Tính accuracy, precision, recall, F1-score
  - Xuất CSV với đầy đủ metrics
"""

import os
import time
import cv2
import numpy as np
import pandas as pd

try:
    from src.matcher import match_orb
except ImportError:
    from matcher import match_orb

try:
    from methods.preprocessing import DEFAULT_PREPROCESS_OPTIONS
except ImportError:
    DEFAULT_PREPROCESS_OPTIONS = None


# ══════════════════════════════════════════════════════════════════════════════
#  1. Evaluate single pair
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_single(
    template_path: str,
    test_path: str,
    ground_truth: str = "Unknown",
    preprocessing_options: dict = None,
    orb_params: dict = None,
    matching_params: dict = None,
    logo_group: str = "",
) -> dict:
    """
    Đánh giá matching giữa 1 cặp ảnh template-test.

    Parameters
    ----------
    template_path        : str  — đường dẫn ảnh template
    test_path            : str  — đường dẫn ảnh test
    ground_truth         : str  — "Positive" hoặc "Negative"
    preprocessing_options: dict — tùy chọn preprocessing
    orb_params           : dict — {"nfeatures": ..., "scaleFactor": ..., ...}
    matching_params      : dict — {"ratio_threshold": ..., "min_good_matches": ..., ...}
    logo_group           : str  — tên nhóm logo (thư mục)

    Returns
    -------
    dict — kết quả đánh giá
    """
    # Đọc ảnh
    tpl = cv2.imread(template_path)
    tst = cv2.imread(test_path)

    if tpl is None:
        return _error_result(template_path, test_path, ground_truth, logo_group,
                             f"Cannot read template: {template_path}")
    if tst is None:
        return _error_result(template_path, test_path, ground_truth, logo_group,
                             f"Cannot read test: {test_path}")

    # Chuyển BGR → RGB (giả lập Streamlit input)
    tpl_rgb = cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB)
    tst_rgb = cv2.cvtColor(tst, cv2.COLOR_BGR2RGB)

    # Build kwargs
    kwargs = {}
    if preprocessing_options:
        kwargs["preprocessing_options"] = preprocessing_options
    if orb_params:
        for key in ["nfeatures", "scaleFactor", "nlevels",
                     "edgeThreshold", "patchSize", "fastThreshold"]:
            if key in orb_params:
                kwargs[key] = orb_params[key]
    if matching_params:
        for key in ["ratio_threshold", "min_good_matches",
                     "ransac_reproj_threshold", "min_inlier_ratio"]:
            if key in matching_params:
                kwargs[key] = matching_params[key]

    # Chạy matching
    result = match_orb(tpl_rgb, tst_rgb, **kwargs)

    # Xác định prediction
    prediction = "Positive" if result["detected"] else "Negative"

    # Build preprocessing config string
    pp_config = "default"
    if preprocessing_options:
        active = []
        if preprocessing_options.get("clahe", True):
            active.append("CLAHE")
        if preprocessing_options.get("gaussian_blur", False):
            active.append("Gaussian")
        if preprocessing_options.get("median_blur", False):
            active.append("Median")
        if preprocessing_options.get("contrast_stretching", False):
            active.append("ContrastStretch")
        if preprocessing_options.get("sharpen", False):
            active.append("Sharpen")
        pp_config = "+".join(active) if active else "none"

    return {
        "Logo_Group": logo_group,
        "Template": os.path.basename(template_path),
        "Image_Name": os.path.basename(test_path),
        "Algorithm": "ORB",
        "method": "ORB",
        "preprocessing_config": pp_config,
        "keypoints_template": result["keypoints_template"],
        "keypoints_query": result["keypoints_query"],
        "total_matches": result["total_matches"],
        "Good_Matches": result["good_matches"],
        "inlier_ratio": result["inlier_ratio"],
        "confidence_score": result["confidence_score"],
        "Prediction": prediction,
        "Ground_Truth": ground_truth,
        "Processing_Time": result["runtime_ms"] / 1000.0,  # giây
        "runtime_ms": result["runtime_ms"],
    }


def _error_result(template_path, test_path, ground_truth, logo_group, error_msg):
    """Tạo kết quả khi có lỗi đọc ảnh."""
    return {
        "Logo_Group": logo_group,
        "Template": os.path.basename(template_path),
        "Image_Name": os.path.basename(test_path),
        "Algorithm": "ORB",
        "method": "ORB",
        "preprocessing_config": "error",
        "keypoints_template": 0,
        "keypoints_query": 0,
        "total_matches": 0,
        "Good_Matches": 0,
        "inlier_ratio": 0.0,
        "confidence_score": 0.0,
        "Prediction": "Error",
        "Ground_Truth": ground_truth,
        "Processing_Time": 0.0,
        "runtime_ms": 0.0,
        "error": error_msg,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. Evaluate dataset (batch)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(
    templates_dir: str,
    test_images_dir: str,
    preprocessing_options: dict = None,
    orb_params: dict = None,
    matching_params: dict = None,
) -> pd.DataFrame:
    """
    Đánh giá batch trên toàn bộ dataset.

    Cấu trúc dataset kỳ vọng:
        templates_dir/
            Logo_01.jpg
            Logo_02.jpg
            ...
        test_images_dir/
            Logo_01/
                pos_01.jpg   (positive — chứa logo)
                pos_02.jpg
                neg_01.jpg   (negative — không chứa logo)
            Logo_02/
                ...

    Quy ước: file bắt đầu bằng "pos" → Positive, "neg" → Negative.

    Returns
    -------
    pd.DataFrame — bảng kết quả đầy đủ
    """
    results = []

    if not os.path.isdir(templates_dir):
        print(f"[WARN] Templates dir not found: {templates_dir}")
        return pd.DataFrame()

    if not os.path.isdir(test_images_dir):
        print(f"[WARN] Test images dir not found: {test_images_dir}")
        return pd.DataFrame()

    # Tìm tất cả template
    template_files = {}
    for f in os.listdir(templates_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(f)[0]
            template_files[name] = os.path.join(templates_dir, f)

    # Duyệt qua từng thư mục test
    for group_name in sorted(os.listdir(test_images_dir)):
        group_path = os.path.join(test_images_dir, group_name)
        if not os.path.isdir(group_path):
            continue

        # Tìm template tương ứng
        template_path = template_files.get(group_name)
        if template_path is None:
            print(f"[WARN] No template found for group: {group_name}")
            continue

        # Duyệt qua ảnh test trong group
        for img_file in sorted(os.listdir(group_path)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            test_path = os.path.join(group_path, img_file)

            # Xác định ground truth từ tên file
            name_lower = img_file.lower()
            if name_lower.startswith("pos"):
                gt = "Positive"
            elif name_lower.startswith("neg"):
                gt = "Negative"
            else:
                gt = "Unknown"

            result = evaluate_single(
                template_path=template_path,
                test_path=test_path,
                ground_truth=gt,
                preprocessing_options=preprocessing_options,
                orb_params=orb_params,
                matching_params=matching_params,
                logo_group=group_name,
            )
            results.append(result)
            print(f"  [{result['Prediction']:8s}] {group_name}/{img_file} "
                  f"— matches={result['Good_Matches']}, "
                  f"conf={result['confidence_score']:.3f}, "
                  f"gt={gt}")

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Tính metrics tổng hợp
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Tính accuracy, precision, recall, F1-score và các thống kê khác.

    Parameters
    ----------
    df : pd.DataFrame — bảng kết quả từ evaluate_dataset

    Returns
    -------
    dict — metrics tổng hợp
    """
    if df.empty:
        return {
            "total_samples": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    # Lọc bỏ samples có ground truth Unknown hoặc Error
    df_eval = df[df["Ground_Truth"].isin(["Positive", "Negative"])].copy()

    if df_eval.empty:
        return {
            "total_samples": len(df),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "note": "No labeled samples found",
        }

    total = len(df_eval)

    # Binary: Positive = 1, Negative = 0
    y_true = (df_eval["Ground_Truth"] == "Positive").astype(int)
    y_pred = (df_eval["Prediction"] == "Positive").astype(int)

    # Accuracy
    correct = (y_true == y_pred).sum()
    accuracy = correct / total if total > 0 else 0.0

    # True Positive, False Positive, False Negative
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Thống kê khác
    avg_runtime = df_eval["runtime_ms"].mean() if "runtime_ms" in df_eval.columns else 0.0
    avg_good_matches = df_eval["Good_Matches"].mean() if "Good_Matches" in df_eval.columns else 0.0
    avg_inlier_ratio = df_eval["inlier_ratio"].mean() if "inlier_ratio" in df_eval.columns else 0.0

    return {
        "total_samples": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "avg_runtime_ms": round(avg_runtime, 2),
        "avg_good_matches": round(avg_good_matches, 2),
        "avg_inlier_ratio": round(avg_inlier_ratio, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. Lưu kết quả
# ══════════════════════════════════════════════════════════════════════════════

def save_results(df: pd.DataFrame, path: str = "results.csv"):
    """Lưu DataFrame kết quả thành CSV."""
    # Chọn và sắp xếp columns
    desired_columns = [
        "Logo_Group", "Template", "Image_Name", "Algorithm",
        "preprocessing_config",
        "keypoints_template", "keypoints_query",
        "total_matches", "Good_Matches",
        "inlier_ratio", "confidence_score",
        "Prediction", "Ground_Truth",
        "Processing_Time", "runtime_ms",
    ]

    # Chỉ giữ columns có trong df
    columns = [c for c in desired_columns if c in df.columns]
    df_save = df[columns]

    df_save.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Demo / test standalone
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Sử dụng: python evaluation.py [templates_dir] [test_images_dir] [output_csv]
    templates_dir = sys.argv[1] if len(sys.argv) > 1 else "data/templates"
    test_dir = sys.argv[2] if len(sys.argv) > 2 else "data/test_images"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "results.csv"

    print(f"Templates: {templates_dir}")
    print(f"Test images: {test_dir}")
    print(f"Output: {output_csv}")
    print("=" * 60)

    df = evaluate_dataset(templates_dir, test_dir)

    if df.empty:
        print("\nNo results (check dataset paths and file structure).")
        sys.exit(0)

    save_results(df, output_csv)

    metrics = compute_metrics(df)
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)
