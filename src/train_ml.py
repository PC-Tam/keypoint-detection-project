"""
train_ml.py
-----------
Train traditional Machine Learning models on extracted feature sets.
Supports SVM (RBF + Linear), Random Forest, KNN, Gaussian NB, Logistic Regression,
and optionally XGBoost (auto-detected).
"""

import os
import sys
import yaml
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from feature_selection import build_preprocessing_pipeline, adapt_k


# -- Optional XGBoost ---------------------------------------------------------
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_classifiers(random_state: int = 42) -> dict:
    """
    Return a dict of {model_name: classifier_instance}.

    Notes
    -----
    - SVM and Logistic Regression use class_weight='balanced' for imbalanced datasets.
    - LinearSVC is wrapped in CalibratedClassifierCV to expose predict_proba.
    - XGBoost is included only if xgboost is installed.
    """
    classifiers = {
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=random_state,
        ),
        "SVM_Linear": CalibratedClassifierCV(
            LinearSVC(
                C=1.0, class_weight="balanced",
                max_iter=5000, random_state=random_state,
            )
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=random_state,
        ),
    }
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        print("[train_ml] XGBoost not installed — skipping.")
    return classifiers


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_set_name: str,
    config: dict,
    models_dir: Path,
) -> dict:
    """
    Train all classifiers on a given feature set and save to disk.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    feature_set_name : str
        Name of the feature bundle (used in saved model filenames).
    config : dict
        Project config (for random_state, feature_selection_k).
    models_dir : Path
        Directory to save trained models.

    Returns
    -------
    dict
        {model_name: fitted_pipeline}
    """
    random_state = config.get("random_state", 42)
    k = config.get("feature_selection_k", 120)
    classifiers = get_classifiers(random_state)

    trained = {}
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name, clf in classifiers.items():
        print(f"  Training {model_name} on '{feature_set_name}'...")
        try:
            pipeline = build_preprocessing_pipeline(clf, k=k)
            pipeline = adapt_k(pipeline, X_train)
            pipeline.fit(X_train, y_train)

            # Save model
            save_path = models_dir / f"{feature_set_name}__{model_name}.joblib"
            joblib.dump(pipeline, str(save_path))

            trained[model_name] = pipeline
            print(f"    OK Saved: {save_path.name}")
        except Exception as e:
            print(f"    ✗ Failed to train {model_name}: {e}")

    return trained


def load_all_feature_sets(features_dir: Path, split: str) -> dict:
    """
    Load all saved feature bundles for a given split from .npy files.

    Parameters
    ----------
    features_dir : Path
        Root features directory containing bundle subdirectories.
    split : str
        One of: "train", "val", "test".

    Returns
    -------
    dict
        {bundle_name: (X, y)}
    """
    bundles = {}
    if not features_dir.exists():
        return bundles
    for bundle_dir in sorted(features_dir.iterdir()):
        if not bundle_dir.is_dir():
            continue
        x_path = bundle_dir / f"X_{split}.npy"
        y_path = bundle_dir / f"y_{split}.npy"
        if x_path.exists() and y_path.exists():
            X = np.load(str(x_path))
            y = np.load(str(y_path))
            bundles[bundle_dir.name] = (X, y)
    return bundles


def run_training(config_path: str = "config.yaml") -> dict:
    """
    Main function: load all feature bundles and train all ML models.

    Returns
    -------
    dict
        {feature_set_name: {model_name: pipeline}}
    """
    config = load_config(config_path)
    features_dir = Path(config.get("features_dir", "data/features"))
    models_dir = Path(config.get("outputs_dir", "outputs")) / "models"

    train_bundles = load_all_feature_sets(features_dir, "train")

    if not train_bundles:
        print("[train_ml] No feature sets found. Run feature_engineering.py first.")
        return {}

    all_trained = {}
    for feat_name, (X_train, y_train) in train_bundles.items():
        print(f"\n{'='*60}")
        print(f"[train_ml] Feature set: {feat_name} | shape: {X_train.shape}")
        print(f"{'='*60}")
        trained = train_models(X_train, y_train, feat_name, config, models_dir)
        all_trained[feat_name] = trained

    print(f"\n[train_ml] Training complete. Models saved to: {models_dir}")
    return all_trained


if __name__ == "__main__":
    run_training()
