"""
run_pipeline.py
---------------
Master pipeline script for the Blood Cell Keypoint ML Classification project.

Steps:
  1. Check / download dataset
  2. Prepare metadata and splits
  3. Benchmark Harris / FAST / ORB (on a sample)
  4. Visualize keypoints
  5. Extract all features (Harris, FAST, ORB, morphology, color/texture)
  6. Fit ORB BoVW vocabulary
  7. Feature selection
  8. Train ML models
  9. Evaluate models
 10. Print best result and save reports
"""

import os
import sys
import yaml
import time
import traceback
from pathlib import Path

# -- Ensure src/ is importable from project root --------------------------------
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from download_dataset import download_dataset
from prepare_dataset import prepare_dataset


def load_config(path: str = "config.yaml") -> dict:
    """Load project configuration from YAML."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        # Try relative to project root
        cfg_path = SRC_DIR.parent / path
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def step(num: int, desc: str) -> None:
    """Print a formatted step header."""
    print(f"\n{'='*65}")
    print(f"  STEP {num}: {desc}")
    print(f"{'='*65}")


def ensure_dirs(config: dict) -> None:
    """Create all required output directories."""
    for d in [
        config.get("raw_data_dir", "data/raw"),
        config.get("processed_data_dir", "data/processed"),
        config.get("features_dir", "data/features"),
        "outputs/figures",
        "outputs/models",
        "outputs/reports",
        "outputs/transformed_cases",
        "outputs/figures/sample_visualizations",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


def run():
    t_start = time.time()
    print("\n" + "=" * 65)
    print("  BLOOD CELL KEYPOINT ML CLASSIFICATION PIPELINE")
    print("=" * 65)

    # -- Load config ------------------------------------------------------------
    config = load_config()
    ensure_dirs(config)

    test_mode_limit = config.get("test_mode_limit")
    benchmark_sample = 200
    viz_per_class = 2

    if test_mode_limit:
        print(f"\n[pipeline] TEST MODE: limiting to {test_mode_limit} images per split.")
        benchmark_sample = min(50, test_mode_limit)
        viz_per_class = 1

    # -- Step 1: Dataset --------------------------------------------------------
    step(1, "Check / Download Dataset")
    ok = download_dataset()
    if not ok:
        print("\n[pipeline] Dataset not available. Cannot continue.")
        print("  Please download the dataset manually and re-run:")
        print("  python src/run_pipeline.py")
        sys.exit(1)

    # -- Step 2: Metadata & Splits ----------------------------------------------
    step(2, "Prepare Metadata and Train/Val/Test Splits")
    train_path, val_path, test_path = prepare_dataset()

    # -- Step 3: Benchmark -----------------------------------------------------
    step(3, f"Benchmark Harris / FAST / ORB (sample ~ {benchmark_sample} images)")
    try:
        from benchmark_keypoints import run_benchmark
        run_benchmark(n_sample=benchmark_sample)
    except Exception as e:
        print(f"  [WARNING] Benchmark failed: {e}")
        traceback.print_exc()

    # -- Step 4: Visualize Keypoints --------------------------------------------
    step(4, "Visualize Keypoints (sample images per class)")
    try:
        from visualize_keypoints import run_visualization
        run_visualization(n_per_class=viz_per_class)
    except Exception as e:
        print(f"  [WARNING] Visualization failed: {e}")

    # -- Step 5–8: Feature Engineering (includes BoVW) --------------------------
    step(5, "Feature Extraction + ORB BoVW Vocabulary + Feature Engineering")
    try:
        from feature_engineering import run_feature_engineering
        run_feature_engineering()
    except Exception as e:
        print(f"  [ERROR] Feature engineering failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -- Step 6: Save transformation case examples ------------------------------
    step(6, "Save Transformation Case Examples")
    try:
        import pandas as pd
        from preprocessing import preprocess_cell_image
        from transformations import save_transformation_cases

        df_meta = pd.read_csv(Path(config.get("processed_data_dir", "data/processed")) / "metadata.csv")
        img_size = tuple(config.get("image_size", [256, 256]))
        trans_dir = "outputs/transformed_cases"
        classes = sorted(df_meta["class_name"].unique())
        for cls in classes[:4]:  # Save a few classes as examples
            sub = df_meta[df_meta["class_name"] == cls]
            sample_row = sub.sample(1, random_state=42).iloc[0]
            try:
                _, gray = preprocess_cell_image(
                    sample_row["image_path"], size=img_size
                )
                save_transformation_cases(gray, trans_dir, prefix=cls)
            except Exception:
                pass
        print(f"  Saved examples to: {trans_dir}")
    except Exception as e:
        print(f"  [WARNING] Transformation examples failed: {e}")

    # -- Step 7: Train ML Models ------------------------------------------------
    step(7, "Train Machine Learning Models")
    try:
        from train_ml import run_training
        run_training()
    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -- Step 8: Evaluate -------------------------------------------------------
    step(8, "Evaluate Models on Test Set")
    try:
        from evaluate import run_evaluation
        df_results = run_evaluation()
        if df_results is not None and len(df_results) > 0:
            print("\n[pipeline] TOP 10 RESULTS BY MACRO F1:")
            print(df_results.head(10).to_string(index=False))
    except Exception as e:
        print(f"  [ERROR] Evaluation failed: {e}")
        traceback.print_exc()

    # -- Done -------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} minutes)")
    print(f"{'='*65}")
    print("\n  Outputs:")
    print("  |---- outputs/reports/keypoint_benchmark.csv")
    print("  |---- outputs/reports/results.csv")
    print("  |---- outputs/reports/best_result.txt")
    print("  |---- outputs/figures/  (confusion matrices, benchmark plots)")
    print("  |---- outputs/figures/sample_visualizations/")
    print("  |---- outputs/models/best_model.joblib")
    print("  \---- outputs/transformed_cases/")
    print("\n  To launch the demo app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    run()
