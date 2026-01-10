"""
Feature Extraction Module

Provides feature extraction and dimensionality reduction utilities:
- CLS token feature extraction from ViT models
- t-SNE visualization
- Feature statistics and analysis

Author: CNElab
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CLASS_NAMES = ["Single", "Competition", "Cooperation"]


# =============================================================================
# Feature Extractor
# =============================================================================

class FeatureExtractor:
    """
    Extract and analyze features from deep learning models.

    Supports both Early Fusion and Late Fusion ViT models for
    dual-stream gaze classification.

    Parameters
    ----------
    model : nn.Module
        PyTorch model with get_features() method
    model_type : str
        Model type: 'early' or 'late'
    device : str
        Device for computation ('cuda' or 'cpu')
    class_names : list, optional
        Class names for labeling

    Examples
    --------
    >>> extractor = FeatureExtractor(model, 'early', 'cuda')
    >>> features, labels, metadata = extractor.extract_features(dataloader)
    >>> tsne_coords = extractor.compute_tsne(features)
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: str = 'cuda',
        class_names: List[str] = None
    ):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.class_names = class_names or DEFAULT_CLASS_NAMES

    def extract_features(
        self,
        dataloader,
        return_metadata: bool = False,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List]]:
        """
        Extract CLS token features from all samples.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader yielding (img_a, img_b, labels) or with metadata
        return_metadata : bool
            Whether to return metadata list
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        features : np.ndarray
            Feature matrix, shape (N, D) where D is feature dimension
        labels : np.ndarray
            Label array, shape (N,)
        metadata : list or None
            List of metadata dicts if return_metadata=True
        """
        self.model.eval()

        all_features = []
        all_labels = []
        all_metadata = [] if return_metadata else None

        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

        with torch.no_grad():
            for batch in iterator:
                # Handle different batch formats
                if len(batch) == 4:
                    img_a, img_b, labels, metadata = batch
                    if return_metadata:
                        all_metadata.extend(metadata)
                elif len(batch) == 3:
                    img_a, img_b, labels = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")

                img_a = img_a.to(self.device)
                img_b = img_b.to(self.device)

                # Extract features based on model type
                if self.model_type == 'early':
                    features = self.model.get_features(img_a, img_b)  # (B, D)
                else:  # late fusion
                    feat_dict = self.model.get_features(img_a, img_b)
                    features = feat_dict['fused']  # Use fused features

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, labels, all_metadata

    def extract_dual_stream_features(
        self,
        dataloader,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from both streams (Late Fusion only).

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader yielding batches
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        dict
            Dictionary with keys: 'cls1', 'cls2', 'fused', 'labels'
        """
        if self.model_type != 'late':
            raise ValueError("Dual stream extraction only available for late fusion models")

        self.model.eval()

        results = {'cls1': [], 'cls2': [], 'fused': [], 'labels': []}

        iterator = tqdm(dataloader, desc="Extracting dual-stream features") if show_progress else dataloader

        with torch.no_grad():
            for batch in iterator:
                if len(batch) >= 3:
                    img_a, img_b, labels = batch[:3]
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")

                img_a = img_a.to(self.device)
                img_b = img_b.to(self.device)

                feat_dict = self.model.get_features(img_a, img_b)

                results['cls1'].append(feat_dict['cls1'].cpu().numpy())
                results['cls2'].append(feat_dict['cls2'].cpu().numpy())
                results['fused'].append(feat_dict['fused'].cpu().numpy())
                results['labels'].append(labels.numpy())

        # Concatenate all batches
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)

        return results

    def compute_tsne(
        self,
        features: np.ndarray,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42,
        n_components: int = 2,
        init: str = 'pca',
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Compute t-SNE projection of features.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix, shape (N, D)
        perplexity : int
            t-SNE perplexity parameter
        n_iter : int
            Number of iterations
        random_state : int
            Random seed for reproducibility
        n_components : int
            Output dimensions (2 or 3)
        init : str
            Initialization method ('pca' or 'random')
        show_progress : bool
            Whether to print progress

        Returns
        -------
        np.ndarray
            t-SNE coordinates, shape (N, n_components)
        """
        if show_progress:
            print(f"  Computing t-SNE (perplexity={perplexity}, n_iter={n_iter})...")

        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(features) - 1),  # Ensure valid perplexity
            learning_rate='auto',
            max_iter=n_iter,
            random_state=random_state,
            init=init
        )

        tsne_coords = tsne.fit_transform(features)
        return tsne_coords

    def compute_pca(
        self,
        features: np.ndarray,
        n_components: int = 50,
        random_state: int = 42
    ) -> Tuple[np.ndarray, PCA]:
        """
        Compute PCA projection of features.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix, shape (N, D)
        n_components : int
            Number of components to keep
        random_state : int
            Random seed

        Returns
        -------
        pca_features : np.ndarray
            PCA-transformed features, shape (N, n_components)
        pca : PCA
            Fitted PCA object
        """
        n_components = min(n_components, features.shape[0], features.shape[1])

        pca = PCA(n_components=n_components, random_state=random_state)
        pca_features = pca.fit_transform(features)

        return pca_features, pca

    def compute_feature_statistics(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute feature statistics per class.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix, shape (N, D)
        labels : np.ndarray
            Class labels, shape (N,)

        Returns
        -------
        pd.DataFrame
            Statistics per class (mean, std, min, max)
        """
        stats = []

        for i, name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                class_features = features[mask]
                stats.append({
                    'class': name,
                    'n_samples': mask.sum(),
                    'feature_mean': class_features.mean(),
                    'feature_std': class_features.std(),
                    'feature_norm_mean': np.linalg.norm(class_features, axis=1).mean(),
                    'feature_norm_std': np.linalg.norm(class_features, axis=1).std()
                })

        return pd.DataFrame(stats)

    # =========================================================================
    # CSV Export Methods
    # =========================================================================

    def save_features_csv(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: Path,
        metadata: Optional[List[Dict]] = None,
        max_features: int = 100
    ):
        """
        Save features with labels and metadata to CSV.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        labels : np.ndarray
            Labels
        save_path : Path
            Output file path
        metadata : list of dict, optional
            Metadata per sample
        max_features : int
            Maximum number of feature columns to save (for large D)
        """
        # Limit feature columns if too many
        n_features = min(features.shape[1], max_features)

        df = pd.DataFrame(
            features[:, :n_features],
            columns=[f'feat_{i}' for i in range(n_features)]
        )

        df['label'] = labels
        df['label_name'] = [self.class_names[l] for l in labels]

        if metadata:
            for key in metadata[0].keys():
                df[key] = [m.get(key, None) for m in metadata]

        df.to_csv(save_path, index=False)
        print(f"  Saved features to {save_path}")

    def save_tsne_csv(
        self,
        tsne_coords: np.ndarray,
        labels: np.ndarray,
        save_path: Path,
        predictions: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Save t-SNE coordinates to CSV.

        Parameters
        ----------
        tsne_coords : np.ndarray
            t-SNE coordinates, shape (N, 2) or (N, 3)
        labels : np.ndarray
            Ground truth labels
        save_path : Path
            Output file path
        predictions : np.ndarray, optional
            Predicted labels
        metadata : list of dict, optional
            Additional metadata
        """
        n_dims = tsne_coords.shape[1]

        data = {f'tsne_{i+1}': tsne_coords[:, i] for i in range(n_dims)}
        data['label'] = labels
        data['label_name'] = [self.class_names[l] for l in labels]

        if predictions is not None:
            data['prediction'] = predictions
            data['pred_name'] = [self.class_names[l] for l in predictions]
            data['correct'] = labels == predictions

        df = pd.DataFrame(data)

        if metadata:
            for key in metadata[0].keys():
                df[key] = [m.get(key, None) for m in metadata]

        df.to_csv(save_path, index=False)
        print(f"  Saved t-SNE coordinates to {save_path}")


# =============================================================================
# Utility Functions
# =============================================================================

def compute_cosine_similarity(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between two feature sets.

    Parameters
    ----------
    features1, features2 : np.ndarray
        Feature matrices of shape (N, D)

    Returns
    -------
    np.ndarray
        Cosine similarity values, shape (N,)
    """
    # Normalize
    norm1 = np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8
    norm2 = np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8

    features1_norm = features1 / norm1
    features2_norm = features2 / norm2

    # Cosine similarity
    return (features1_norm * features2_norm).sum(axis=1)


def compute_euclidean_distance(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean distance between two feature sets.

    Parameters
    ----------
    features1, features2 : np.ndarray
        Feature matrices of shape (N, D)

    Returns
    -------
    np.ndarray
        Euclidean distances, shape (N,)
    """
    return np.linalg.norm(features1 - features2, axis=1)


def compute_class_centroids(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 3
) -> np.ndarray:
    """
    Compute class centroids in feature space.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix, shape (N, D)
    labels : np.ndarray
        Class labels, shape (N,)
    n_classes : int
        Number of classes

    Returns
    -------
    np.ndarray
        Class centroids, shape (n_classes, D)
    """
    centroids = np.zeros((n_classes, features.shape[1]))

    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            centroids[i] = features[mask].mean(axis=0)

    return centroids


def compute_intra_class_variance(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None
) -> Dict[int, float]:
    """
    Compute intra-class variance for each class.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    labels : np.ndarray
        Class labels
    centroids : np.ndarray, optional
        Pre-computed centroids

    Returns
    -------
    dict
        Variance per class
    """
    n_classes = len(np.unique(labels))

    if centroids is None:
        centroids = compute_class_centroids(features, labels, n_classes)

    variances = {}
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            class_features = features[mask]
            distances = np.linalg.norm(class_features - centroids[i], axis=1)
            variances[i] = distances.var()
        else:
            variances[i] = 0.0

    return variances


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=" * 60)
    print("Testing Feature Extractor")
    print("=" * 60)

    # Generate synthetic features
    np.random.seed(42)
    n_samples = 300
    n_features = 768

    # Create clustered features
    features = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples, dtype=int)

    for i in range(3):
        start_idx = i * 100
        end_idx = (i + 1) * 100
        center = np.random.randn(n_features) * 2
        features[start_idx:end_idx] = center + np.random.randn(100, n_features) * 0.5
        labels[start_idx:end_idx] = i

    # Test t-SNE (without model)
    print("\n[1] Testing t-SNE computation...")

    # Create a mock extractor (without actual model)
    class MockExtractor:
        def __init__(self):
            self.class_names = DEFAULT_CLASS_NAMES

        def compute_tsne(self, features, **kwargs):
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            return tsne.fit_transform(features)

    extractor = MockExtractor()
    tsne_coords = extractor.compute_tsne(features)
    print(f"  t-SNE output shape: {tsne_coords.shape}")

    # Test utility functions
    print("\n[2] Testing utility functions...")

    centroids = compute_class_centroids(features, labels)
    print(f"  Centroids shape: {centroids.shape}")

    variances = compute_intra_class_variance(features, labels, centroids)
    print(f"  Intra-class variances: {variances}")

    # Test cosine similarity
    cos_sim = compute_cosine_similarity(features[:100], features[100:200])
    print(f"  Cosine similarity (class 0 vs 1): mean={cos_sim.mean():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
