"""
Early Fusion Vision Transformer for Dual-stream Gaze Heatmap Classification

This module implements an Early Fusion ViT architecture that fuses two gaze
heatmaps (from two players) before feeding them into a pretrained Vision Transformer.

Supported Fusion Modes:
    - concat: Channel concatenation (6 channels) - requires modified patch_embed
    - add: Pixel-wise addition (3 channels) - captures common features
    - subtract: Pixel-wise subtraction (3 channels) - captures differences
    - subtract_abs: Absolute difference (3 channels) - captures differences (symmetric)
    - multiply: Pixel-wise multiplication (3 channels) - captures overlapping regions

Architecture (refer to fig2.GazeArch.png - Section A):
    Stream 1 (B, 3, H, W) ──┬── Fusion Strategy ──► (B, C, H, W) ──► ViT ──► Classification
    Stream 2 (B, 3, H, W) ──┘   (C=6 for concat, C=3 for others)

Author: Kung-Yi Chang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Literal


# Supported fusion modes
FUSION_MODES = Literal['concat', 'add', 'subtract', 'subtract_abs', 'multiply']


class EarlyFusionViT(nn.Module):
    """
    Early Fusion Vision Transformer for dual-stream gaze heatmap classification.

    This model performs early fusion using various strategies (concat, add, subtract,
    multiply) before processing through a pretrained ViT backbone.

    Args:
        model_name: Name of the ViT model from timm (default: 'vit_base_patch16_224')
        num_classes: Number of output classes (default: 3 for Single/Competition/Cooperation)
        pretrained: Whether to load pretrained weights (default: True)
        img_size: Input image size (default: 224)
        fusion_mode: Fusion strategy to use (default: 'concat')
            - 'concat': Channel concatenation -> 6 channels (requires patch_embed modification)
            - 'add': (img_a + img_b) / 2 -> 3 channels (common features)
            - 'subtract': (img_a - img_b) / 2 -> 3 channels (directional difference)
            - 'subtract_abs': |img_a - img_b| -> 3 channels (symmetric difference)
            - 'multiply': normalized(img_a * img_b) -> 3 channels (overlapping emphasis)
        weight_init_strategy: Strategy for initializing 6-channel weights (only for concat mode)
            - 'duplicate': Copy original weights to both channel groups (default)
            - 'average': Use averaged weights for the second channel group

    Input:
        img_a: Tensor of shape (B, 3, H, W) - Gaze heatmap from Player 1
        img_b: Tensor of shape (B, 3, H, W) - Gaze heatmap from Player 2

    Output:
        logits: Tensor of shape (B, num_classes) - Classification logits
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        num_classes: int = 3,
        pretrained: bool = True,
        img_size: int = 224,
        fusion_mode: FUSION_MODES = 'concat',
        weight_init_strategy: Literal['duplicate', 'average'] = 'duplicate'
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self.weight_init_strategy = weight_init_strategy

        # Validate fusion mode
        valid_modes = ['concat', 'add', 'subtract', 'subtract_abs', 'multiply']
        if fusion_mode not in valid_modes:
            raise ValueError(f"fusion_mode must be one of {valid_modes}, got '{fusion_mode}'")

        # ================================================================
        # Load pretrained ViT backbone from timm
        # ================================================================
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size
        )

        # ================================================================
        # Only modify patch_embed for concat mode (6 channels)
        # Other modes use original 3-channel input (full pretrained weights)
        # ================================================================
        if fusion_mode == 'concat':
            self._modify_patch_embed_for_6_channels()
            print(f"[EarlyFusionViT] fusion_mode='concat' -> 6-channel input")
        else:
            print(f"[EarlyFusionViT] fusion_mode='{fusion_mode}' -> 3-channel input (using full pretrained weights)")

    def _modify_patch_embed_for_6_channels(self):
        """
        Modify the patch embedding layer to accept 6-channel input instead of 3.

        The pretrained ViT's patch_embed.proj is a Conv2d with:
            - in_channels=3, out_channels=embed_dim, kernel_size=patch_size

        We create a new Conv2d with in_channels=6 and initialize its weights
        by leveraging the pretrained 3-channel weights to preserve feature
        extraction capabilities.
        """
        original_proj = self.backbone.patch_embed.proj

        out_channels = original_proj.out_channels
        kernel_size = original_proj.kernel_size
        stride = original_proj.stride
        padding = original_proj.padding

        original_weight = original_proj.weight.data.clone()
        original_bias = original_proj.bias.data.clone() if original_proj.bias is not None else None

        new_proj = nn.Conv2d(
            in_channels=6,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(original_bias is not None)
        )

        with torch.no_grad():
            new_weight = new_proj.weight.data

            if self.weight_init_strategy == 'duplicate':
                new_weight[:, 0:3, :, :] = original_weight
                new_weight[:, 3:6, :, :] = original_weight
            elif self.weight_init_strategy == 'average':
                new_weight[:, 0:3, :, :] = original_weight
                avg_weight = original_weight.mean(dim=1, keepdim=True)
                new_weight[:, 3:6, :, :] = avg_weight.expand_as(original_weight)

            if original_bias is not None:
                new_proj.bias.data = original_bias

        self.backbone.patch_embed.proj = new_proj

    def _fuse_inputs(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Fuse two input images based on the configured fusion mode.

        All pixel-wise operations handle value range to maintain compatibility
        with pretrained ViT expectations (normalized ImageNet distribution).

        Args:
            img_a: (B, 3, H, W) - normalized image from Player 1
            img_b: (B, 3, H, W) - normalized image from Player 2

        Returns:
            fused: (B, C, H, W) where C=6 for concat, C=3 for others
        """
        if self.fusion_mode == 'concat':
            # Channel concatenation: (B, 3, H, W) + (B, 3, H, W) -> (B, 6, H, W)
            fused = torch.cat([img_a, img_b], dim=1)

        elif self.fusion_mode == 'add':
            # Addition with scaling to maintain similar value range
            # (a + b) / 2 keeps values in similar range as original normalized images
            fused = (img_a + img_b) / 2.0

        elif self.fusion_mode == 'subtract':
            # Subtraction with scaling
            # (a - b) / 2 keeps values in similar range, preserves sign (directional)
            fused = (img_a - img_b) / 2.0

        elif self.fusion_mode == 'subtract_abs':
            # Absolute difference - symmetric, always positive relative to 0
            # |a - b| captures magnitude of difference without direction
            fused = torch.abs(img_a - img_b)

        elif self.fusion_mode == 'multiply':
            # Multiplication with instance normalization
            # Product can have very different distribution, so we normalize per-instance
            product = img_a * img_b  # (B, 3, H, W)

            # Instance normalization: normalize each image in batch independently
            # This brings values back to roughly zero-mean, unit-variance
            B, C, H, W = product.shape
            product_flat = product.view(B, C, -1)  # (B, 3, H*W)
            mean = product_flat.mean(dim=2, keepdim=True)  # (B, 3, 1)
            std = product_flat.std(dim=2, keepdim=True) + 1e-6  # (B, 3, 1)
            normalized = (product_flat - mean) / std
            fused = normalized.view(B, C, H, W)

        return fused

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with early fusion.

        Args:
            img_a: Gaze heatmap from Player 1, shape (B, 3, H, W)
            img_b: Gaze heatmap from Player 2, shape (B, 3, H, W)

        Returns:
            logits: Classification logits, shape (B, num_classes)

        Dimension Flow:
            img_a:  (B, 3, H, W)
            img_b:  (B, 3, H, W)
                    | Fusion (mode-dependent)
                    v
            fused:  (B, 6, H, W) for concat
                    (B, 3, H, W) for add/subtract/subtract_abs/multiply
                    |
                    v ViT Backbone
            logits: (B, num_classes)
        """
        # Fuse inputs based on configured mode
        fused = self._fuse_inputs(img_a, img_b)

        # Forward through ViT backbone
        logits = self.backbone(fused)

        return logits

    def get_features(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classification head (useful for visualization/analysis).

        Args:
            img_a: Gaze heatmap from Player 1, shape (B, 3, H, W)
            img_b: Gaze heatmap from Player 2, shape (B, 3, H, W)

        Returns:
            features: CLS token features, shape (B, embed_dim)
        """
        fused = self._fuse_inputs(img_a, img_b)
        features = self.backbone.forward_features(fused)
        cls_features = features[:, 0]
        return cls_features


# ============================================================================
# Factory function for convenience
# ============================================================================

def create_early_fusion_vit(
    model_name: str = 'vit_base_patch16_224',
    num_classes: int = 3,
    pretrained: bool = True,
    fusion_mode: str = 'concat',
    **kwargs
) -> EarlyFusionViT:
    """
    Factory function to create an Early Fusion ViT model.

    Args:
        model_name: Name of ViT model from timm
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        fusion_mode: Fusion strategy ('concat', 'add', 'subtract', 'subtract_abs', 'multiply')
        **kwargs: Additional arguments passed to EarlyFusionViT

    Returns:
        Configured EarlyFusionViT model
    """
    return EarlyFusionViT(
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
    print("Testing EarlyFusionViT with all fusion modes")
    print("=" * 60)

    batch_size = 2
    img_a = torch.randn(batch_size, 3, 224, 224)
    img_b = torch.randn(batch_size, 3, 224, 224)

    fusion_modes = ['concat', 'add', 'subtract', 'subtract_abs', 'multiply']

    for mode in fusion_modes:
        print(f"\n{'='*40}")
        print(f"Testing fusion_mode='{mode}'")
        print(f"{'='*40}")

        model = EarlyFusionViT(
            model_name='vit_base_patch16_224',
            num_classes=3,
            pretrained=True,
            fusion_mode=mode
        )

        model.eval()
        with torch.no_grad():
            logits = model(img_a, img_b)
            features = model.get_features(img_a, img_b)

        expected_channels = 6 if mode == 'concat' else 3
        print(f"  Input: img_a {img_a.shape}, img_b {img_b.shape}")
        print(f"  Fused channels: {expected_channels}")
        print(f"  Output logits: {logits.shape}")
        print(f"  Output features: {features.shape}")

        assert logits.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {logits.shape}"
        print(f"  [OK] Test passed!")

    print("\n" + "=" * 60)
    print("[OK] All fusion modes tested successfully!")
    print("=" * 60)
