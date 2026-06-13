"""
feature_selection.py
--------------------
Feature selection utilities using SelectKBest and StandardScaler.
Builds sklearn Pipelines that prevent data leakage by fitting only on train data.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif


def build_preprocessing_pipeline(
    classifier,
    k: int = 120,
    score_func=mutual_info_classif,
) -> Pipeline:
    """
    Build a sklearn Pipeline with StandardScaler + SelectKBest + classifier.

    The scaler and selector are fitted ONLY on training data (via pipeline.fit).
    They are never fitted on validation or test data.

    Parameters
    ----------
    classifier :
        Any sklearn-compatible classifier object.
    k : int
        Number of top features to select. If the feature matrix has fewer than
        k features, all features are used (k is clamped).
    score_func : callable
        Scoring function for SelectKBest (default: mutual_info_classif).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with steps: scaler -> selector -> classifier.
    """
    steps = [
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=score_func, k=k)),
        ("classifier", classifier),
    ]
    return Pipeline(steps)


def adapt_k(pipeline: Pipeline, X_train: np.ndarray) -> Pipeline:
    """
    Adjust the selector's k parameter if the number of features < k.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline with a 'selector' step.
    X_train : np.ndarray
        Training feature matrix.

    Returns
    -------
    Pipeline
        Pipeline with adjusted k.
    """
    n_features = X_train.shape[1]
    current_k = pipeline.named_steps["selector"].k
    if isinstance(current_k, int) and current_k > n_features:
        pipeline.named_steps["selector"].k = n_features
    return pipeline


def apply_feature_selection(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """
    Fit the pipeline on training data and transform val/test.
    Only applies scaler + selector (not classifier) for the transform step.

    Parameters
    ----------
    pipeline : Pipeline
    X_train, y_train : training data
    X_val, X_test : validation and test data

    Returns
    -------
    tuple of (X_train_sel, X_val_sel, X_test_sel)
        Feature-selected matrices ready for classifier training.
    """
    # Extract preprocessing steps only (not the classifier)
    preprocess = Pipeline(pipeline.steps[:-1])
    preprocess.fit(X_train, y_train)

    X_train_sel = preprocess.transform(X_train)
    X_val_sel = preprocess.transform(X_val)
    X_test_sel = preprocess.transform(X_test)

    return X_train_sel, X_val_sel, X_test_sel
