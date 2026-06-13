"""
bovw.py
-------
Bag of Visual Words (BoVW) implementation using ORB descriptors.
Uses MiniBatchKMeans to build a visual vocabulary exclusively on training data.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans


def train_bovw_vocabulary(
    descriptors_list: list,
    n_clusters: int = 100,
    random_state: int = 42,
    save_path: str = None,
) -> MiniBatchKMeans:
    """
    Fit a visual vocabulary (KMeans) on ORB descriptors from the training set.

    IMPORTANT: Must only be called on TRAINING descriptors to prevent data leakage.

    Parameters
    ----------
    descriptors_list : list of np.ndarray
        List of descriptor arrays, each of shape (N_i, 32) uint8.
        Arrays with 0 rows are skipped automatically.
    n_clusters : int
        Number of visual words (vocabulary size).
    random_state : int
        Random seed for reproducibility.
    save_path : str or None
        If given, saves the fitted KMeans model to this path.

    Returns
    -------
    MiniBatchKMeans
        The fitted vocabulary model.
    """
    # Collect all non-empty descriptors
    valid_descs = [d for d in descriptors_list if d is not None and len(d) > 0]
    if not valid_descs:
        raise ValueError(
            "[bovw] No valid ORB descriptors found in the training set. "
            "Check that ORB extraction succeeded on at least some images."
        )

    all_descs = np.vstack(valid_descs).astype(np.float32)
    print(f"[bovw] Fitting KMeans with {n_clusters} clusters "
          f"on {len(all_descs)} descriptors from {len(valid_descs)} images...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(1024, len(all_descs)),
        n_init=3,
    )
    kmeans.fit(all_descs)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(kmeans, save_path)
        print(f"[bovw] KMeans model saved to: {save_path}")

    return kmeans


def compute_bovw_histogram(
    descriptors: np.ndarray,
    kmeans: MiniBatchKMeans,
    normalize: str = "l1",
) -> np.ndarray:
    """
    Compute a BoVW histogram for a single image's ORB descriptors.

    Parameters
    ----------
    descriptors : np.ndarray
        ORB descriptors of shape (N, 32) uint8, or empty array.
    kmeans : MiniBatchKMeans
        Fitted vocabulary model.
    normalize : str
        Normalization method: "l1", "l2", or None.

    Returns
    -------
    np.ndarray
        Histogram of shape (n_clusters,), float32.
    """
    n_clusters = kmeans.n_clusters
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(n_clusters, dtype=np.float32)

    descs = descriptors.astype(np.float32)
    word_ids = kmeans.predict(descs)
    hist = np.bincount(word_ids, minlength=n_clusters).astype(np.float32)

    if normalize == "l1":
        total = hist.sum()
        if total > 0:
            hist /= total
    elif normalize == "l2":
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm

    return hist


def load_bovw_model(model_path: str) -> MiniBatchKMeans:
    """
    Load a saved BoVW KMeans vocabulary from disk.

    Parameters
    ----------
    model_path : str
        Path to the saved .joblib file.

    Returns
    -------
    MiniBatchKMeans
        Loaded vocabulary model.
    """
    return joblib.load(str(model_path))
