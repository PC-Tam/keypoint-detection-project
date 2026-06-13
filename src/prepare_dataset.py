"""
prepare_dataset.py
------------------
Scans the raw data directory, normalizes class names, creates metadata CSV,
and performs stratified train/val/test splits.
"""

import os
import re
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# -- Canonical blood-cell class whitelist ------------------------------------
# Only these normalized names are accepted; any other folder is silently skipped.
BLOOD_CELL_CLASSES = {
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature_granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
}

# -- Class name normalization -------------------------------------------------

CLASS_ALIASES = {
    "basophil": "basophil",
    "basophils": "basophil",
    "eosinophil": "eosinophil",
    "eosinophils": "eosinophil",
    "erythroblast": "erythroblast",
    "erythroblasts": "erythroblast",
    "immaturegranulocyte": "immature_granulocyte",
    "immaturegranulocytes": "immature_granulocyte",
    "immature_granulocyte": "immature_granulocyte",
    "immature_granulocytes": "immature_granulocyte",
    "ig": "immature_granulocyte",
    "lymphocyte": "lymphocyte",
    "lymphocytes": "lymphocyte",
    "monocyte": "monocyte",
    "monocytes": "monocyte",
    "neutrophil": "neutrophil",
    "neutrophils": "neutrophil",
    "platelet": "platelet",
    "platelets": "platelet",
    "thrombocyte": "platelet",
    "thrombocytes": "platelet",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalize_folder_name(name: str) -> str:
    """
    Normalize a folder name to a canonical class key.
    Lowercases, strips spaces/dashes/underscores, then looks up alias table.
    """
    key = re.sub(r"[\s\-_]", "", name.lower())
    return CLASS_ALIASES.get(key, key)


def scan_images(raw_dir: Path) -> list[dict]:
    """
    Recursively scan raw_dir for image files.
    Uses the immediate parent folder name as the class label.
    Only images whose normalized class name appears in BLOOD_CELL_CLASSES
    are included — other folders (e.g. glaucoma, nonglaucoma) are skipped.

    Parameters
    ----------
    raw_dir : Path
        Root directory containing raw images.

    Returns
    -------
    list of dict
        Each dict has keys: image_path, filename, raw_class, class_name.
    """
    records = []
    skipped_classes = set()

    for ext in SUPPORTED_EXTENSIONS:
        for p in raw_dir.rglob(f"*{ext}"):
            raw_class = p.parent.name
            normalized = normalize_folder_name(raw_class)
            if normalized not in BLOOD_CELL_CLASSES:
                skipped_classes.add(raw_class)
                continue
            records.append({
                "image_path": str(p),
                "filename": p.name,
                "raw_class": raw_class,
                "class_name": normalized,
            })
        # Also uppercase extensions (e.g. .JPG)
        for p in raw_dir.rglob(f"*{ext.upper()}"):
            raw_class = p.parent.name
            normalized = normalize_folder_name(raw_class)
            if normalized not in BLOOD_CELL_CLASSES:
                skipped_classes.add(raw_class)
                continue
            records.append({
                "image_path": str(p),
                "filename": p.name,
                "raw_class": raw_class,
                "class_name": normalized,
            })

    if skipped_classes:
        print(f"  [prepare_dataset] Skipped non-blood-cell folder(s): {sorted(skipped_classes)}")
    # Deduplicate
    seen = set()
    unique_records = []
    for r in records:
        key = r["image_path"]
        if key not in seen:
            seen.add(key)
            unique_records.append(r)
    return unique_records


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(config_path: str = "config.yaml") -> tuple[str, str, str]:
    """
    Main function to prepare the dataset metadata and splits.

    Parameters
    ----------
    config_path : str
        Path to the config YAML file.

    Returns
    -------
    tuple of (train_csv_path, val_csv_path, test_csv_path)
    """
    config = load_config(config_path)
    raw_dir = Path(config.get("raw_data_dir", "data/raw"))
    processed_dir = Path(config.get("processed_data_dir", "data/processed"))
    test_size = float(config.get("test_size", 0.2))
    val_size = float(config.get("val_size", 0.1))
    random_state = int(config.get("random_state", 42))

    processed_dir.mkdir(parents=True, exist_ok=True)

    # -- Scan images ----------------------------------------------------------
    print(f"[prepare_dataset] Scanning images in: {raw_dir}")
    records = scan_images(raw_dir)

    if not records:
        print(f"[prepare_dataset] ERROR: No images found in '{raw_dir}'.")
        print("  Please download the dataset first (see download_dataset.py).")
        sys.exit(1)

    df = pd.DataFrame(records)

    # -- Assign integer labels ------------------------------------------------
    classes = sorted(df["class_name"].unique())
    class_to_label = {c: i for i, c in enumerate(classes)}
    df["label"] = df["class_name"].map(class_to_label)

    # -- Print class statistics -----------------------------------------------
    print(f"\n[prepare_dataset] Found {len(df)} images across {len(classes)} classes:")
    print(f"  {'Class':<30} {'Count':>8}  {'%':>6}")
    print("  " + "-" * 48)
    counts = df["class_name"].value_counts()
    total = len(df)
    for cls in sorted(counts.index):
        cnt = counts[cls]
        pct = cnt / total * 100
        print(f"  {cls:<30} {cnt:>8}  {pct:>5.1f}%")
    print()

    # -- Check for class imbalance --------------------------------------------
    min_count = counts.min()
    max_count = counts.max()
    imbalance_ratio = max_count / max(min_count, 1)
    if imbalance_ratio > 5:
        print(f"  [WARNING] Class imbalance detected (ratio {imbalance_ratio:.1f}x).")
        print("  Recommendation: use class_weight='balanced' in SVM/LogReg,")
        print("  and class_weight='balanced_subsample' in RandomForest.\n")

    # -- Always use stratified split (ignore pre-existing split dirs) ---------
    # The original dataset may have train/test folders whose test split
    # belonged to a different dataset (e.g. ACRIMA/Glaucoma), resulting in
    # 0 test images for blood cells after filtering. We re-split ourselves.
    print("[prepare_dataset] Performing stratified split (train/val/test)...")
    val_actual = val_size / (1.0 - test_size)
    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_actual,
        stratify=df.loc[train_val_idx, "label"],
        random_state=random_state,
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # -- Print split statistics -----------------------------------------------
    print("\n[prepare_dataset] Split statistics:")
    for split_name in ["train", "val", "test"]:
        n = len(df[df["split"] == split_name])
        print(f"  {split_name:<8}: {n} images")

    # -- Save CSVs ------------------------------------------------------------
    meta_path = processed_dir / "metadata.csv"
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    df.to_csv(meta_path, index=False)
    df[df["split"] == "train"].to_csv(train_path, index=False)
    df[df["split"] == "val"].to_csv(val_path, index=False)
    df[df["split"] == "test"].to_csv(test_path, index=False)

    # Save label mapping
    label_map = pd.DataFrame({
        "class_name": list(class_to_label.keys()),
        "label": list(class_to_label.values()),
    })
    label_map.to_csv(processed_dir / "label_map.csv", index=False)

    print(f"\n[prepare_dataset] Saved:")
    print(f"  {meta_path}")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {processed_dir / 'label_map.csv'}")

    return str(train_path), str(val_path), str(test_path)


if __name__ == "__main__":
    prepare_dataset()
