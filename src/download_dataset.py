import os
import sys
import yaml
import subprocess
from pathlib import Path

def load_config():
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("Error: config.yaml not found.")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_dataset():
    config = load_config()
    dataset_id = config['dataset']['kaggle_id']
    raw_dir = Path(config['dataset']['raw_dir'])
    
    # Check if dataset already exists (if directory has images)
    if raw_dir.exists() and len(list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))) > 0:
        print(f"Dataset already exists in {raw_dir}. Skipping download.")
        return True

    print(f"Dataset not found in {raw_dir}. Attempting to download using Kaggle API...")
    
    try:
        # Create directory if it doesn't exist
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to run kaggle command
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_id, "-p", str(raw_dir), "--unzip"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("Dataset downloaded and unzipped successfully.")
            return True
        else:
            print("Kaggle API download failed. Reason:")
            print(result.stderr)
    except FileNotFoundError:
        print("Kaggle CLI not found. Please ensure it is installed and configured.")
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")

    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print(f"Please download the dataset manually from: https://www.kaggle.com/datasets/{dataset_id}")
    print(f"Extract the images and place them inside the following directory:")
    print(f"{raw_dir.absolute()}")
    print("Then, re-run the pipeline.")
    print("="*60 + "\n")
    return False

if __name__ == "__main__":
    download_dataset()
