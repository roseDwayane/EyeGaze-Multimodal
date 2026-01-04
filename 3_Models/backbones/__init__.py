"""
Backbone models for EyeGaze-Multimodal project.

Contains Vision Transformer and other encoder architectures.
"""

from .early_fusion_vit import EarlyFusionViT, create_early_fusion_vit
from .late_fusion_vit import LateFusionViT, create_late_fusion_vit

__all__ = [
    'EarlyFusionViT',
    'create_early_fusion_vit',
    'LateFusionViT',
    'create_late_fusion_vit',
]
