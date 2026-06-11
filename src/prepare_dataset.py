import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def prepare_dataset():
    config = load_config()
    raw_dir = Path(config['dataset']['raw_dir'])
    metadata_path = Path(config['dataset']['processed_metadata'])
    train_path = Path(config['dataset']['train_csv'])
    test_path = Path(config['dataset']['test_csv'])
    
    if not raw_dir.exists():
        print(f"Error: Raw directory {raw_dir} does not exist.")
        print("Please run download_dataset.py first or manually place the dataset.")
        sys.exit(1)

    image_paths = []
    filenames = []
    labels = []
    label_names = []

    # Supported image extensions
    valid_exts = {".jpg", ".jpeg", ".png"}

    for file_path in raw_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
            filename = file_path.name
            
            # Label assignment based on ACRIMA naming convention
            # Files with '_g_' or 'glaucoma' are glaucoma (1), otherwise normal (0)
            if "_g_" in filename.lower() or "glaucoma" in filename.lower():
                label = 1
                label_name = "Glaucoma"
            elif "normal" in filename.lower() or "_g_" not in filename.lower():
                label = 0
                label_name = "Normal"
            else:
                continue # Skip unrecognized files if any

            image_paths.append(str(file_path))
            filenames.append(filename)
            labels.append(label)
            label_names.append(label_name)

    if not image_paths:
        print(f"Error: No images found in {raw_dir}.")
        sys.exit(1)

    df = pd.DataFrame({
        'image_path': image_paths,
        'filename': filenames,
        'label': labels,
        'label_name': label_names
    })

    # Ensure processed dir exists
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(metadata_path, index=False)
    print(f"Saved metadata for {len(df)} images to {metadata_path}")

    # Stratified split
    train_df, test_df = train_test_split(
        df, 
        test_size=config['dataset']['test_size'], 
        stratify=df['label'], 
        random_state=config['dataset']['random_state']
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train split ({len(train_df)} samples) to {train_path}")
    print(f"Saved test split ({len(test_df)} samples) to {test_path}")

    return train_path, test_path

if __name__ == "__main__":
    prepare_dataset()
