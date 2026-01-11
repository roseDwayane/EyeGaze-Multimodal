"""
Backbone models for EyeGaze-Multimodal project.

Contains Vision Transformer, EEG encoders, and other architectures.
"""

from .early_fusion_vit import EarlyFusionViT, create_early_fusion_vit
from .late_fusion_vit import LateFusionViT, create_late_fusion_vit
from .hypereeg import HyperEEG_Encoder, create_hypereeg_model
from .multi_stream_feature_transformer import (
    MultiStreamFeatureTransformer,
    create_msft_model,
)
from .dual_eeg_transformer import DualEEGTransformer

__all__ = [
    # Vision Transformers
    'EarlyFusionViT',
    'create_early_fusion_vit',
    'LateFusionViT',
    'create_late_fusion_vit',
    # EEG Models
    'HyperEEG_Encoder',
    'create_hypereeg_model',
    'MultiStreamFeatureTransformer',
    'create_msft_model',
    'DualEEGTransformer',
]
