"""
evaluate.py
-----------
Evaluation utilities for Blood Cell classification models.
Computes accuracy, macro/weighted F1, confusion matrix, per-class metrics,
top-2 accuracy, and ROC-AUC. Saves reports and figures.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    top_k_accuracy_score,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_unique_labels_and_names(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_to_class: dict,
) -> tuple:
    """
    Compute the union of labels appearing in y_test and y_pred,
    and build corresponding human-readable class name list.

    This prevents the "Number of classes X does not match size of
    target_names Y" error when y_test only contains a subset of all
    trained classes (e.g., on small test splits).

    Returns
    -------
    tuple of (unique_labels: list, class_names: list)
    """
    unique_labels = sorted(
        set(np.unique(y_test).tolist()) | set(np.unique(y_pred).tolist())
    )
    class_names = [label_to_class.get(lbl, str(lbl)) for lbl in unique_labels]
    return unique_labels, class_names


def evaluate_model(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    feature_set: str,
    label_to_class: dict = None,
) -> dict:
    """
    Evaluate a trained sklearn pipeline on test data.

    Parameters
    ----------
    pipeline :
        Fitted sklearn Pipeline with a predict method.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test labels.
    model_name : str
        Name of the model (for reporting).
    feature_set : str
        Name of the feature bundle (for reporting).
    label_to_class : dict or None
        Mapping from integer label -> class name string.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    if label_to_class is None:
        label_to_class = {}

    y_pred = pipeline.predict(X_test)

    # -- Core metrics ---------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    n_classes = len(set(np.unique(y_test).tolist()) | set(np.unique(y_pred).tolist()))

    # -- Top-2 accuracy -------------------------------------------------------
    top2_acc = None
    try:
        if hasattr(pipeline, "predict_proba") and n_classes >= 2:
            y_proba = pipeline.predict_proba(X_test)
            top2_acc = top_k_accuracy_score(y_test, y_proba, k=min(2, y_proba.shape[1]))
    except Exception:
        pass

    # -- ROC-AUC --------------------------------------------------------------
    roc_auc = None
    try:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
    except Exception:
        pass

    result = {
        "feature_set": feature_set,
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "top2_accuracy": round(top2_acc, 4) if top2_acc is not None else None,
        "roc_auc_macro": round(roc_auc, 4) if roc_auc is not None else None,
    }
    return result


def save_confusion_matrix(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    feature_set: str,
    label_to_class: dict,
    figures_dir: Path,
) -> None:
    """
    Plot and save a confusion matrix as a PNG.
    Uses only the labels actually present in y_test + y_pred to avoid
    dimension mismatches on small test sets.
    """
    y_pred = pipeline.predict(X_test)
    unique_labels, class_names = _get_unique_labels_and_names(
        y_test, y_pred, label_to_class
    )

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    n = len(class_names)
    fig_size = max(8, n * 1.2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(f"Confusion Matrix\n{feature_set} | {model_name}", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    fname = f"confusion_matrix__{feature_set}__{model_name}.png"
    save_path = figures_dir / fname
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_classification_report(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    feature_set: str,
    label_to_class: dict,
    reports_dir: Path,
) -> None:
    """
    Generate and save a classification report as a .txt file.
    Passes explicit labels to avoid target_names size mismatch.
    """
    y_pred = pipeline.predict(X_test)
    unique_labels, class_names = _get_unique_labels_and_names(
        y_test, y_pred, label_to_class
    )

    report = classification_report(
        y_test, y_pred,
        labels=unique_labels,
        target_names=class_names,
        zero_division=0,
    )
    fname = f"classification_report__{feature_set}__{model_name}.txt"
    save_path = reports_dir / fname
    with open(str(save_path), "w") as f:
        f.write(f"Feature Set: {feature_set}\n")
        f.write(f"Model: {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)


def run_evaluation(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Main function: load all trained models and evaluate on the test set.

    Returns
    -------
    pd.DataFrame
        Summary of all model evaluation results, sorted by macro F1.
    """
    config = load_config(config_path)
    features_dir = Path(config.get("features_dir", "data/features"))
    models_dir = Path(config.get("outputs_dir", "outputs")) / "models"
    reports_dir = Path(config.get("outputs_dir", "outputs")) / "reports"
    figures_dir = Path(config.get("outputs_dir", "outputs")) / "figures"
    processed_dir = Path(config.get("processed_data_dir", "data/processed"))

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    label_to_class = {}
    label_map_path = processed_dir / "label_map.csv"
    if label_map_path.exists():
        label_map_df = pd.read_csv(label_map_path)
        label_to_class = dict(zip(
            label_map_df["label"].astype(int),
            label_map_df["class_name"]
        ))

    all_results = []
    best_f1 = -1.0
    best_model_meta = None
    best_pipeline = None

    # Iterate over all model files
    model_files = sorted(models_dir.glob("*.joblib"))
    if not model_files:
        print("[evaluate] No trained models found in:", models_dir)
        return pd.DataFrame()

    for model_path in model_files:
        if model_path.name in ("orb_bovw_kmeans.joblib", "best_model.joblib"):
            continue

        # Parse filename: feature_set__model_name.joblib
        parts = model_path.stem.split("__")
        if len(parts) < 2:
            continue
        feature_set = parts[0]
        model_name = "__".join(parts[1:])

        # Load test features
        x_path = features_dir / feature_set / "X_test.npy"
        y_path = features_dir / feature_set / "y_test.npy"
        if not (x_path.exists() and y_path.exists()):
            continue

        X_test = np.load(str(x_path))
        y_test = np.load(str(y_path))

        try:
            pipeline = joblib.load(str(model_path))
            res = evaluate_model(
                pipeline, X_test, y_test, model_name, feature_set, label_to_class
            )
            all_results.append(res)

            # Save artefacts using label_to_class (not pre-built class_names)
            save_confusion_matrix(
                pipeline, X_test, y_test, model_name, feature_set,
                label_to_class, figures_dir
            )
            save_classification_report(
                pipeline, X_test, y_test, model_name, feature_set,
                label_to_class, reports_dir
            )

            print(f"  [{feature_set}] {model_name}: "
                  f"acc={res['accuracy']:.3f}  macro_f1={res['f1_macro']:.3f}")

            if res["f1_macro"] > best_f1:
                best_f1 = res["f1_macro"]
                best_model_meta = res
                best_pipeline = pipeline

        except Exception as e:
            print(f"  [evaluate] Error on {model_path.name}: {e}")

    if not all_results:
        return pd.DataFrame()

    # -- Save results CSV ------------------------------------------------------
    df = pd.DataFrame(all_results)
    df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    df.to_csv(str(reports_dir / "results.csv"), index=False)

    # -- Save best model -------------------------------------------------------
    if best_pipeline is not None:
        joblib.dump(best_pipeline, str(models_dir / "best_model.joblib"))

    # -- Save best result summary ----------------------------------------------
    if best_model_meta:
        with open(str(reports_dir / "best_result.txt"), "w") as f:
            f.write("BEST MODEL BY MACRO F1-SCORE\n")
            f.write("=" * 60 + "\n")
            for k, v in best_model_meta.items():
                f.write(f"  {k}: {v}\n")

    print(f"\n[evaluate] Best model: {best_model_meta['feature_set']} | "
          f"{best_model_meta['model']} -> macro F1 = {best_f1:.4f}")
    print(f"[evaluate] Results saved to: {reports_dir / 'results.csv'}")

    return df


if __name__ == "__main__":
    run_evaluation()
