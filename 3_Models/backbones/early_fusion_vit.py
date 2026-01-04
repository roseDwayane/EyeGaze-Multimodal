"""
Early Fusion Vision Transformer for Dual-stream Gaze Heatmap Classification

This module implements an Early Fusion ViT architecture that concatenates
two gaze heatmaps (from two players) along the channel dimension before
feeding them into a pretrained Vision Transformer.

Architecture (refer to fig2.GazeArch.png - Section A):
    Stream 1 (B, 3, H, W) ──┬── Channel Concat ──► (B, 6, H, W) ──► ViT ──► Classification
    Stream 2 (B, 3, H, W) ──┘

Author: Kung-Yi Chang
"""

import torch
import torch.nn as nn
import timm
from typing import Literal


class EarlyFusionViT(nn.Module):
    """
    Early Fusion Vision Transformer for dual-stream gaze heatmap classification.

    This model performs early fusion by concatenating two RGB heatmaps along
    the channel dimension, then processes the fused 6-channel input through
    a pretrained ViT backbone with a modified patch embedding layer.

    Args:
        model_name: Name of the ViT model from timm (default: 'vit_base_patch16_224')
        num_classes: Number of output classes (default: 3 for Single/Competition/Cooperation)
        pretrained: Whether to load pretrained weights (default: True)
        img_size: Input image size (default: 224)
        weight_init_strategy: Strategy for initializing 6-channel weights
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
        weight_init_strategy: Literal['duplicate', 'average'] = 'duplicate'
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.weight_init_strategy = weight_init_strategy

        # ================================================================
        # Step 1: Load pretrained ViT backbone from timm
        # ================================================================
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size
        )

        # ================================================================
        # Step 2: Modify patch_embed to accept 6-channel input
        # ================================================================
        self._modify_patch_embed_for_6_channels()

    def _modify_patch_embed_for_6_channels(self):
        """
        Modify the patch embedding layer to accept 6-channel input instead of 3.

        The pretrained ViT's patch_embed.proj is a Conv2d with:
            - in_channels=3, out_channels=embed_dim, kernel_size=patch_size

        We create a new Conv2d with in_channels=6 and initialize its weights
        by leveraging the pretrained 3-channel weights to preserve feature
        extraction capabilities.
        """
        # Get the original patch embedding projection layer
        original_proj = self.backbone.patch_embed.proj

        # Extract original layer parameters
        out_channels = original_proj.out_channels      # embed_dim (e.g., 768 for ViT-Base)
        kernel_size = original_proj.kernel_size        # patch_size (e.g., 16)
        stride = original_proj.stride                  # typically same as kernel_size
        padding = original_proj.padding                # typically 0

        # Get original weights: shape (out_channels, 3, kernel_h, kernel_w)
        original_weight = original_proj.weight.data.clone()
        original_bias = original_proj.bias.data.clone() if original_proj.bias is not None else None

        # ================================================================
        # Create new Conv2d layer for 6-channel input
        # ================================================================
        new_proj = nn.Conv2d(
            in_channels=6,                             # Modified: 6 channels (concatenated)
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(original_bias is not None)
        )

        # ================================================================
        # Initialize new weights using pretrained weights
        # ================================================================
        with torch.no_grad():
            # New weight shape: (out_channels, 6, kernel_h, kernel_w)
            new_weight = new_proj.weight.data

            if self.weight_init_strategy == 'duplicate':
                # Strategy: Duplicate original weights for both channel groups
                # - Channels 0-2 (img_a): Use original pretrained weights
                # - Channels 3-5 (img_b): Use same pretrained weights (duplicated)
                new_weight[:, 0:3, :, :] = original_weight  # For img_a channels
                new_weight[:, 3:6, :, :] = original_weight  # For img_b channels

            elif self.weight_init_strategy == 'average':
                # Strategy: Original weights for first 3, averaged for last 3
                # - Channels 0-2: Use original pretrained weights
                # - Channels 3-5: Use channel-averaged weights (promotes learning differences)
                new_weight[:, 0:3, :, :] = original_weight
                # Average across input channels and broadcast back
                avg_weight = original_weight.mean(dim=1, keepdim=True)
                new_weight[:, 3:6, :, :] = avg_weight.expand_as(original_weight)

            # Copy bias (unchanged, as it's per output channel)
            if original_bias is not None:
                new_proj.bias.data = original_bias

        # Replace the original projection layer
        self.backbone.patch_embed.proj = new_proj

        # Update the num_features attribute if it exists
        if hasattr(self.backbone.patch_embed, 'num_features'):
            # num_features remains the same (embed_dim), only input channels changed
            pass

        print(f"[EarlyFusionViT] Modified patch_embed.proj: "
              f"in_channels 3 -> 6, weight_init='{self.weight_init_strategy}'")

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with early fusion (channel concatenation).

        Args:
            img_a: Gaze heatmap from Player 1, shape (B, 3, H, W)
            img_b: Gaze heatmap from Player 2, shape (B, 3, H, W)

        Returns:
            logits: Classification logits, shape (B, num_classes)

        Dimension Flow:
            img_a:  (B, 3, H, W)
            img_b:  (B, 3, H, W)
                    ↓ Channel Concatenation
            fused:  (B, 6, H, W)
                    ↓ ViT Backbone (modified patch_embed)
            logits: (B, num_classes)
        """
        # ================================================================
        # Early Fusion: Concatenate along channel dimension
        # ================================================================
        # img_a: (B, 3, H, W) + img_b: (B, 3, H, W) → fused: (B, 6, H, W)
        fused = torch.cat([img_a, img_b], dim=1)

        # ================================================================
        # Forward through ViT backbone
        # ================================================================
        # The backbone's forward_features handles:
        #   1. Patch embedding (now accepts 6 channels)
        #   2. Add CLS token
        #   3. Add positional embedding
        #   4. Transformer encoder blocks
        #   5. Classification head
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
        fused = torch.cat([img_a, img_b], dim=1)  # (B, 6, H, W)
        features = self.backbone.forward_features(fused)  # (B, num_patches+1, embed_dim)

        # Extract CLS token (first token)
        cls_features = features[:, 0]  # (B, embed_dim)
        return cls_features


# ============================================================================
# Factory function for convenience
# ============================================================================

def create_early_fusion_vit(
    model_name: str = 'vit_base_patch16_224',
    num_classes: int = 3,
    pretrained: bool = True,
    **kwargs
) -> EarlyFusionViT:
    """
    Factory function to create an Early Fusion ViT model.

    Args:
        model_name: Name of ViT model from timm. Supported models include:
            - 'vit_base_patch16_224' (default)
            - 'vit_small_patch16_224'
            - 'vit_large_patch16_224'
            - 'vit_base_patch32_224'
            - 'deit_base_patch16_224' (DeiT variant)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments passed to EarlyFusionViT

    Returns:
        Configured EarlyFusionViT model
    """
    return EarlyFusionViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


# ============================================================================
# Test / Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing EarlyFusionViT")
    print("=" * 60)

    # Create model
    model = EarlyFusionViT(
        model_name='vit_base_patch16_224',
        num_classes=3,
        pretrained=True,
        weight_init_strategy='duplicate'
    )

    # Create dummy input
    batch_size = 4
    img_a = torch.randn(batch_size, 3, 224, 224)  # Player 1 gaze heatmap
    img_b = torch.randn(batch_size, 3, 224, 224)  # Player 2 gaze heatmap

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(img_a, img_b)
        features = model.get_features(img_a, img_b)

    print(f"\nInput shapes:")
    print(f"  img_a: {img_a.shape}")
    print(f"  img_b: {img_b.shape}")
    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  features (CLS token): {features.shape}")

    # Verify output
    assert logits.shape == (batch_size, 3), f"Expected (4, 3), got {logits.shape}"
    print("\n[OK] All tests passed!")

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
