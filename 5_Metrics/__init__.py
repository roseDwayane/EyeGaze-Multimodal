# 5_Metrics: Evaluation metrics, entropy calculators, and feature extractors

from .entropy_calculators import (
    BaseEntropyCalculator,
    SpatialEntropyCalculator,
    SpectralEntropyCalculator,
    STANDARD_32_CHANNELS,
    CHANNEL_POSITIONS_2D,
)

from .classification_metrics import (
    ClassificationMetrics,
    compute_class_weights,
    compute_per_class_accuracy,
    DEFAULT_CLASS_NAMES,
)

from .feature_extractors import (
    FeatureExtractor,
    compute_cosine_similarity,
    compute_euclidean_distance,
    compute_class_centroids,
    compute_intra_class_variance,
)

__all__ = [
    # Entropy calculators
    'BaseEntropyCalculator',
    'SpatialEntropyCalculator',
    'SpectralEntropyCalculator',
    'STANDARD_32_CHANNELS',
    'CHANNEL_POSITIONS_2D',
    # Classification metrics
    'ClassificationMetrics',
    'compute_class_weights',
    'compute_per_class_accuracy',
    'DEFAULT_CLASS_NAMES',
    # Feature extractors
    'FeatureExtractor',
    'compute_cosine_similarity',
    'compute_euclidean_distance',
    'compute_class_centroids',
    'compute_intra_class_variance',
]
