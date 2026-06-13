"""
download_dataset.py
-------------------
Script to download the Blood Cells Image Dataset from Kaggle.
Supports both automatic (Kaggle API) and manual download workflows.
"""

import os
import sys
import subprocess
from pathlib import Path


DATASET_SLUG = "unclesamulus/blood-cells-image-dataset"
RAW_DATA_DIR = Path("data/raw")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images_in_dir(directory: Path) -> int:
    """Recursively count image files in a directory."""
    if not directory.exists():
        return 0
    count = 0
    for ext in SUPPORTED_EXTENSIONS:
        count += len(list(directory.rglob(f"*{ext}")))
        count += len(list(directory.rglob(f"*{ext.upper()}")))
    return count


def print_manual_instructions():
    """Print instructions for manually downloading the dataset."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print(f"1. Go to: https://www.kaggle.com/datasets/{DATASET_SLUG}")
    print("2. Click the 'Download' button (requires Kaggle account).")
    print(f"3. Unzip the downloaded file into: {RAW_DATA_DIR.resolve()}")
    print("   The final structure should look like:")
    print("   data/raw/")
    print("   \---- blood-cells-image-dataset/  (or similar folder)")
    print("       |---- basophil/")
    print("       |---- eosinophil/")
    print("       |---- ... (other cell classes)")
    print("4. Re-run: python src/run_pipeline.py")
    print("=" * 60 + "\n")


def check_kaggle_api() -> bool:
    """Check if Kaggle API is configured (kaggle.json exists)."""
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path("data/kaggle.json"),
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"
        if os.environ.get("KAGGLE_CONFIG_DIR") else None,
    ]
    for p in kaggle_json_paths:
        if p and p.exists():
            return True
    # Also check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def download_dataset() -> bool:
    """
    Download the Blood Cells dataset using Kaggle API.

    Returns
    -------
    bool
        True if dataset is available (already downloaded or successfully downloaded),
        False if dataset could not be found/downloaded.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if images already exist
    n_images = count_images_in_dir(RAW_DATA_DIR)
    if n_images > 0:
        print(f"[download_dataset] Dataset already present: {n_images} images found in '{RAW_DATA_DIR}'.")
        return True

    print(f"[download_dataset] No images found in '{RAW_DATA_DIR}'. Attempting to download...")

    # Check Kaggle API availability
    if not check_kaggle_api():
        print("[download_dataset] Kaggle API credentials not found.")
        print_manual_instructions()
        return False

    # Try downloading via Kaggle CLI
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(RAW_DATA_DIR),
        "--unzip"
    ]
    print(f"[download_dataset] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            n_after = count_images_in_dir(RAW_DATA_DIR)
            if n_after > 0:
                print(f"[download_dataset] Download successful! {n_after} images found.")
                return True
            else:
                print("[download_dataset] Download command succeeded but no images found.")
                print(result.stdout)
                print_manual_instructions()
                return False
        else:
            print(f"[download_dataset] Kaggle download failed:\n{result.stderr}")
            print_manual_instructions()
            return False
    except FileNotFoundError:
        print("[download_dataset] 'kaggle' module not found. Install with: pip install kaggle")
        print_manual_instructions()
        return False
    except subprocess.TimeoutExpired:
        print("[download_dataset] Download timed out (600s). Please download manually.")
        print_manual_instructions()
        return False
    except Exception as e:
        print(f"[download_dataset] Unexpected error: {e}")
        print_manual_instructions()
        return False


if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
