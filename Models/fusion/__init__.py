"""
Multimodal Fusion Models for Eye Gaze Image + EEG Signal
"""

from .late_fusion import LateFusionModel
from .mid_fusion import MidFusionModel, SimplifiedMidFusion
from .early_fusion import EarlyFusionModel, ChannelWiseEarlyFusion, EEGToTimeFrequency
from .symmetric_fusion import (
    SymmetricFusionOperators,
    SymmetricFusionWithGating,
    MultiScaleFusion
)
from .cross_modal_attention import (
    CrossModalAttention,
    CoAttention,
    GatedCrossModalFusion,
    MultiModalTransformerBlock
)

__all__ = [
    # Fusion models
    'LateFusionModel',
    'MidFusionModel',
    'SimplifiedMidFusion',
    'EarlyFusionModel',
    'ChannelWiseEarlyFusion',

    # Components
    'EEGToTimeFrequency',
    'SymmetricFusionOperators',
    'SymmetricFusionWithGating',
    'MultiScaleFusion',
    'CrossModalAttention',
    'CoAttention',
    'GatedCrossModalFusion',
    'MultiModalTransformerBlock'
]
