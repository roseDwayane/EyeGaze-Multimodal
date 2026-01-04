"""
Late Fusion Vision Transformer for Dual-stream Gaze Heatmap Classification

This module implements a Late Fusion ViT architecture that processes two gaze
heatmaps through a shared encoder (Siamese), then fuses the CLS token features
at the decision level.

Supported Fusion Modes:
    - concat: [CLS_1, CLS_2] -> 2D dimensions
    - add: CLS_1 + CLS_2 -> D dimensions
    - subtract: CLS_1 - CLS_2 -> D dimensions
    - multiply: CLS_1 * CLS_2 -> D dimensions
    - full: [concat, subtract, multiply] -> 4D dimensions (refer to fig2.GazeArch.png - Section B)

Architecture (refer to fig2.GazeArch.png - Section B):
    Stream 1 (B, 3, H, W) ──┐                      ┌── CLS_1 ──┐
                           ├── Shared ViT Encoder ─┤          ├── Fusion ──► Classification
    Stream 2 (B, 3, H, W) ──┘                      └── CLS_2 ──┘

Author: Kung-Yi Chang
"""

import torch
import torch.nn as nn
import timm
from typing import Literal


# Supported fusion modes
FUSION_MODES = Literal['concat', 'add', 'subtract', 'multiply', 'full']


class LateFusionViT(nn.Module):
    """
    Late Fusion Vision Transformer for dual-stream gaze heatmap classification.

    This model implements a **Siamese Network** architecture where a single shared
    ViT encoder processes two input gaze heatmaps independently. The resulting
    CLS token features are then fused at the decision level before classification.

    Siamese Architecture Benefits:
        - Parameter efficiency: Single encoder for both streams (weight sharing)
        - Symmetric processing: Both inputs processed identically
        - Feature alignment: Shared representation space enables meaningful fusion

    The `fusion_mode` parameter dynamically determines the input dimension of the
    classification Linear layer:

        | fusion_mode | Fused Dim          | Linear Layer Input    |
        |-------------|--------------------|-----------------------|
        | concat      | 2 * embed_dim      | 1536 (for ViT-Base)   |
        | add         | embed_dim          | 768                   |
        | subtract    | embed_dim          | 768                   |
        | multiply    | embed_dim          | 768                   |
        | full        | 4 * embed_dim      | 3072                  |

    Args:
        model_name: Name of the ViT model from timm (default: 'vit_base_patch16_224')
        num_classes: Number of output classes (default: 3 for Single/Competition/Cooperation)
        pretrained: Whether to load pretrained ImageNet weights (default: True)
        fusion_mode: Fusion strategy for CLS tokens (default: 'full')
            - 'concat': [CLS_1, CLS_2] -> captures both individual features
            - 'add': CLS_1 + CLS_2 -> captures common/shared attention patterns
            - 'subtract': CLS_1 - CLS_2 -> captures directional differences
            - 'multiply': CLS_1 * CLS_2 -> captures interaction/overlap regions
            - 'full': [concat, subtract, multiply] -> comprehensive multi-view fusion
        dropout: Dropout rate before classification head (default: 0.1)

    Input:
        x1: Tensor of shape (B, 3, H, W) - Gaze heatmap from Player 1
        x2: Tensor of shape (B, 3, H, W) - Gaze heatmap from Player 2

    Output:
        logits: Tensor of shape (B, num_classes) - Classification logits

    Example:
        >>> model = LateFusionViT(fusion_mode='full', num_classes=3)
        >>> x1 = torch.randn(4, 3, 224, 224)  # Player 1 gaze heatmap
        >>> x2 = torch.randn(4, 3, 224, 224)  # Player 2 gaze heatmap
        >>> logits = model(x1, x2)  # (4, 3)
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        num_classes: int = 3,
        pretrained: bool = True,
        fusion_mode: FUSION_MODES = 'full',
        dropout: float = 0.1
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode

        # Validate fusion mode
        valid_modes = ['concat', 'add', 'subtract', 'multiply', 'full']
        if fusion_mode not in valid_modes:
            raise ValueError(f"fusion_mode must be one of {valid_modes}, got '{fusion_mode}'")

        # ================================================================
        # Shared ViT Encoder (Siamese)
        # num_classes=0 removes the classification head, outputs CLS token
        # ================================================================
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head, output features only
        )

        # Get embedding dimension from encoder (768 for ViT-Base)
        self.embed_dim = self.encoder.num_features

        # ================================================================
        # Calculate fused feature dimension based on fusion mode
        # ================================================================
        if fusion_mode == 'concat':
            # [CLS_1, CLS_2] -> 2D
            self.fused_dim = 2 * self.embed_dim
        elif fusion_mode in ['add', 'subtract', 'multiply']:
            # Element-wise operation -> D
            self.fused_dim = self.embed_dim
        elif fusion_mode == 'full':
            # [concat(2D), subtract(D), multiply(D)] -> 4D
            self.fused_dim = 4 * self.embed_dim

        # ================================================================
        # Classification Head with Dropout
        # ================================================================
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.fused_dim, num_classes)

    def _fuse_features(self, cls1: torch.Tensor, cls2: torch.Tensor) -> torch.Tensor:
        """
        Fuse two CLS token features based on the configured fusion mode.

        Args:
            cls1: CLS token from stream 1, shape (B, D)
            cls2: CLS token from stream 2, shape (B, D)

        Returns:
            fused: Fused features, shape depends on fusion_mode:
                - concat: (B, 2D)
                - add/subtract/multiply: (B, D)
                - full: (B, 4D)
        """
        if self.fusion_mode == 'concat':
            # Simple concatenation: [CLS_1, CLS_2]
            # (B, D) + (B, D) -> (B, 2D)
            fused = torch.cat([cls1, cls2], dim=1)

        elif self.fusion_mode == 'add':
            # Element-wise addition: CLS_1 + CLS_2
            # Captures common/shared features
            fused = cls1 + cls2

        elif self.fusion_mode == 'subtract':
            # Element-wise subtraction: CLS_1 - CLS_2
            # Captures directional differences
            fused = cls1 - cls2

        elif self.fusion_mode == 'multiply':
            # Element-wise multiplication: CLS_1 * CLS_2
            # Captures interaction/overlap features
            fused = cls1 * cls2

        elif self.fusion_mode == 'full':
            # Full fusion: combine concat, subtract, and multiply
            # As shown in fig2.GazeArch.png - Section B
            concat_feat = torch.cat([cls1, cls2], dim=1)  # (B, 2D)
            subtract_feat = cls1 - cls2                    # (B, D)
            multiply_feat = cls1 * cls2                    # (B, D)

            # Concatenate all: (B, 2D) + (B, D) + (B, D) -> (B, 4D)
            fused = torch.cat([concat_feat, subtract_feat, multiply_feat], dim=1)

        return fused

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Siamese encoder with late fusion.

        The two input gaze heatmaps are processed by the same shared ViT encoder
        (Siamese architecture), producing CLS token features that are then fused
        according to the configured fusion_mode before classification.

        Args:
            x1: Gaze heatmap from Player 1, shape (B, 3, H, W)
            x2: Gaze heatmap from Player 2, shape (B, 3, H, W)

        Returns:
            logits: Classification logits, shape (B, num_classes)

        Dimension Flow (Siamese Late Fusion):
            x1: (B, 3, H, W) ──┐
                              ├── Shared ViT Encoder ──> cls1: (B, D)
            x2: (B, 3, H, W) ──┘                     ──> cls2: (B, D)
                                                              |
                                                              v Fusion (mode-dependent)
                                                        fused: (B, fused_dim)
                                                              |
                                                              v Dropout + Linear
                                                        logits: (B, num_classes)

        Note:
            The fused_dim depends on fusion_mode:
            - concat: 2*D, add/subtract/multiply: D, full: 4*D
            where D is the encoder's embedding dimension (768 for ViT-Base).
        """
        # ================================================================
        # Shared Encoder: Extract CLS tokens from both streams
        # ================================================================
        cls1 = self.encoder(x1)  # (B, D) - CLS token for stream 1
        cls2 = self.encoder(x2)  # (B, D) - CLS token for stream 2

        # ================================================================
        # Fuse features based on fusion mode
        # ================================================================
        fused = self._fuse_features(cls1, cls2)

        # ================================================================
        # Classification with dropout
        # ================================================================
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        return logits

    def get_features(self, x1: torch.Tensor, x2: torch.Tensor) -> dict:
        """
        Extract features at different stages (useful for visualization/analysis).

        Args:
            x1: Gaze heatmap from Player 1, shape (B, 3, H, W)
            x2: Gaze heatmap from Player 2, shape (B, 3, H, W)

        Returns:
            Dictionary containing:
                - 'cls1': CLS token from stream 1, shape (B, D)
                - 'cls2': CLS token from stream 2, shape (B, D)
                - 'fused': Fused features, shape (B, fused_dim)
        """
        cls1 = self.encoder(x1)
        cls2 = self.encoder(x2)
        fused = self._fuse_features(cls1, cls2)

        return {
            'cls1': cls1,
            'cls2': cls2,
            'fused': fused
        }


# ============================================================================
# Factory function for convenience
# ============================================================================

def create_late_fusion_vit(
    model_name: str = 'vit_base_patch16_224',
    num_classes: int = 3,
    pretrained: bool = True,
    fusion_mode: str = 'full',
    **kwargs
) -> LateFusionViT:
    """
    Factory function to create a Late Fusion ViT model.

    Args:
        model_name: Name of ViT model from timm
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        fusion_mode: Fusion strategy ('concat', 'add', 'subtract', 'multiply', 'full')
        **kwargs: Additional arguments passed to LateFusionViT

    Returns:
        Configured LateFusionViT model
    """
    return LateFusionViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        fusion_mode=fusion_mode,
        **kwargs
    )


# ============================================================================
# Test / Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing LateFusionViT with all fusion modes")
    print("=" * 60)

    batch_size = 2
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)

    fusion_modes = ['concat', 'add', 'subtract', 'multiply', 'full']
    embed_dim = 768  # ViT-Base

    expected_dims = {
        'concat': 2 * embed_dim,     # 1536
        'add': embed_dim,            # 768
        'subtract': embed_dim,       # 768
        'multiply': embed_dim,       # 768
        'full': 4 * embed_dim        # 3072
    }

    for mode in fusion_modes:
        print(f"\n{'='*40}")
        print(f"Testing fusion_mode='{mode}'")
        print(f"{'='*40}")

        model = LateFusionViT(
            model_name='vit_base_patch16_224',
            num_classes=3,
            pretrained=True,
            fusion_mode=mode,
            dropout=0.1
        )

        model.eval()
        with torch.no_grad():
            logits = model(x1, x2)
            features = model.get_features(x1, x2)

        print(f"  Input: x1 {x1.shape}, x2 {x2.shape}")
        print(f"  CLS tokens: cls1 {features['cls1'].shape}, cls2 {features['cls2'].shape}")
        print(f"  Fused features: {features['fused'].shape} (expected dim: {expected_dims[mode]})")
        print(f"  Output logits: {logits.shape}")

        # Verify dimensions
        assert features['fused'].shape[1] == expected_dims[mode], \
            f"Expected fused_dim={expected_dims[mode]}, got {features['fused'].shape[1]}"
        assert logits.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {logits.shape}"
        print(f"  [OK] Test passed!")

    print("\n" + "=" * 60)
    print("[OK] All fusion modes tested successfully!")
    print("=" * 60)

    # Model statistics for 'full' mode
    model = LateFusionViT(fusion_mode='full', pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics (fusion_mode='full'):")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
