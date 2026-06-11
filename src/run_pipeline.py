import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from download_dataset import download_dataset
from prepare_dataset import prepare_dataset
from preprocessing import preprocess_fundus_image
from feature_harris import extract_harris_features
from feature_fast import extract_fast_features
from feature_orb import extract_orb_descriptors
from bovw import train_bovw_model, compute_bovw_histogram
from train_ml import train_models
from evaluate import evaluate_model, save_all_results

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def extract_features_for_split(df, config, kmeans_model=None):
    """
    Extract all features for a given dataframe of images.
    Returns dictionaries of feature arrays and the labels.
    """
    h_features = []
    f_features = []
    o_descriptors_list = []
    labels = []
    
    print("Extracting features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        path = row['image_path']
        label = row['label']
        
        try:
            img = preprocess_fundus_image(
                path, 
                size=tuple(config['preprocessing']['image_size']),
                use_green=config['preprocessing']['use_green_channel'],
                use_clahe=config['preprocessing']['use_clahe']
            )
            
            # Harris
            if config['features']['harris']['enabled']:
                hf = extract_harris_features(img, **{k:v for k,v in config['features']['harris'].items() if k != 'enabled'})
                h_features.append(hf)
                
            # FAST
            if config['features']['fast']['enabled']:
                ff = extract_fast_features(img, **{k:v for k,v in config['features']['fast'].items() if k != 'enabled'})
                f_features.append(ff)
                
            # ORB
            if config['features']['orb']['enabled']:
                _, od = extract_orb_descriptors(img, nfeatures=config['features']['orb']['nfeatures'])
                o_descriptors_list.append(od)
                
            labels.append(label)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    # BoVW for ORB
    o_features = []
    if config['features']['orb']['enabled'] and kmeans_model is not None:
        for desc in o_descriptors_list:
            bovw_hist = compute_bovw_histogram(desc, kmeans_model)
            o_features.append(bovw_hist)
            
    return np.array(h_features), np.array(f_features), o_descriptors_list, np.array(o_features), np.array(labels)


def run():
    config = load_config()
    
    # 1. Download & Check
    if not download_dataset():
        sys.exit(1)
        
    # 2. Prepare metadata and split
    train_path, test_path = prepare_dataset()
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Clear previous classification reports
    report_file = Path("outputs/reports/classification_report.txt")
    if report_file.exists():
        report_file.unlink()
        
    # 3. Process Train Set
    print("\n--- Processing Train Set ---")
    X_train_h, X_train_f, train_orb_desc, _, y_train = extract_features_for_split(train_df, config)
    
    # Build BoVW model from train ORB descriptors
    print("Building BoVW KMeans Model...")
    all_train_desc = np.vstack([d for d in train_orb_desc if len(d) > 0])
    kmeans_model = train_bovw_model(all_train_desc, n_clusters=config['features']['bovw']['n_clusters'])
    
    # Compute Train BoVW features
    X_train_o = np.array([compute_bovw_histogram(desc, kmeans_model) for desc in train_orb_desc])
    
    # 4. Process Test Set
    print("\n--- Processing Test Set ---")
    X_test_h, X_test_f, test_orb_desc, X_test_o, y_test = extract_features_for_split(test_df, config, kmeans_model)
    
    # 5. Train & Evaluate Models
    all_results = []
    
    feature_sets = [
        ("Harris", X_train_h, X_test_h),
        ("FAST", X_train_f, X_test_f),
        ("ORB_BoVW", X_train_o, X_test_o)
    ]
    
    for feat_name, X_tr, X_te in feature_sets:
        if X_tr.shape[0] == 0:
            continue
            
        print(f"\n--- Running ML on {feat_name} features ---")
        models = train_models(X_tr, y_train, feat_name)
        
        for model_name, model in models.items():
            res = evaluate_model(model, X_te, y_test, model_name, feat_name)
            all_results.append(res)
            
    # 6. Save & Summarize Results
    df_results = save_all_results(all_results)
    print("\n" + "="*50)
    print("PIPELINE COMPLETE. RESULTS:")
    print("="*50)
    print(df_results.to_string(index=False))
    
    # Find best model
    best_idx = df_results['F1-Score'].idxmax()
    best_row = df_results.iloc[best_idx]
    print(f"\nBest Model by F1-Score: {best_row['Model']} using {best_row['Feature']} (F1: {best_row['F1-Score']:.4f})")

if __name__ == "__main__":
    run()
