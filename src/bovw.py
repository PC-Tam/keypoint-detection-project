import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

def train_bovw_model(all_descriptors, n_clusters=50, random_state=42):
    """
    Train a MiniBatchKMeans model on all collected ORB descriptors.
    Args:
        all_descriptors: A single numpy array of shape (N, 32) containing all descriptors from the training set.
        n_clusters: Number of visual words (vocabulary size).
    Returns:
        kmeans: The trained KMeans model.
    """
    # ORB descriptors are uint8 (0-255), we can convert to float32 for clustering
    all_descriptors = all_descriptors.astype(np.float32)
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1000)
    kmeans.fit(all_descriptors)
    
    # Save the model
    save_dir = Path("outputs/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, save_dir / "orb_bovw_kmeans.joblib")
    
    return kmeans

def compute_bovw_histogram(descriptors, kmeans_model):
    """
    Convert a set of descriptors from a single image into a BoVW histogram.
    Args:
        descriptors: Array of shape (num_keypoints, 32).
        kmeans_model: Trained KMeans model.
    Returns:
        histogram: Normalized 1D numpy array of length `n_clusters`.
    """
    n_clusters = kmeans_model.n_clusters
    
    if len(descriptors) == 0:
        return np.zeros(n_clusters)
        
    # Predict the visual word for each descriptor
    words = kmeans_model.predict(descriptors.astype(np.float32))
    
    # Create histogram
    histogram, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
    
    # Normalize histogram
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
        
    return histogram

def load_bovw_model(path="outputs/models/orb_bovw_kmeans.joblib"):
    return joblib.load(path)
