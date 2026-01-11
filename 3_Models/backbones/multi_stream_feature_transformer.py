"""
Multi-Stream Feature Transformer for Hyperscanning EEG Classification

This module implements a Transformer-based architecture that processes pre-extracted
EEG features (from extract_eeg_features.py) for social interaction classification.

Architecture:
    - 3 Specialized Encoders:
        - SpectralEncoder (MLP): Process band energy features
        - IntraConnEncoder (CNN): Process intra-brain connectivity matrices
        - InterConnEncoder (CNN): Process inter-brain connectivity matrices

    - 5 Tokens:
        - Spec_P1, Spec_P2: Spectral features for each player
        - Intra_P1, Intra_P2: Intra-brain connectivity for each player
        - Inter_12: Inter-brain connectivity between players

    - TransformerFusion: 2-layer Transformer for cross-modality attention
    - MultiTokenUncertaintyFusion: Inverse-variance weighted fusion

Input Features (from extract_eeg_features.py):
    - bands_energy: (B, 2, 32, 5) - Players x Channels x Bands
    - intra_con: (B, 2, 7, 5, 32, 32) - Players x Metrics x Bands x Ch x Ch
    - inter_con: (B, 7, 5, 32, 32) - Metrics x Bands x Ch_A x Ch_B

Output:
    - logits: (B, 3) for {Single, Competition, Cooperation}

Supports ablation studies via boolean switches.

Author: Generated for Gaze-EEG Multimodal Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, List


# =============================================================================
# Encoder 1: Spectral/Band Energy Encoder (MLP)
# =============================================================================

class SpectralEncoder(nn.Module):
    """
    Spectral/Band Energy Encoder using MLP.

    Processes per-player band energy features.

    Args:
        num_channels: Number of EEG channels (default: 32)
        num_bands: Number of frequency bands (default: 5)
        embed_dim: Output embedding dimension (default: 256)
        dropout: Dropout rate (default: 0.1)

    Shape:
        Input: (B, num_channels, num_bands) = (B, 32, 5)
        Output: (B, embed_dim)
    """

    def __init__(
        self,
        num_channels: int = 32,
        num_bands: int = 5,
        embed_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_bands = num_bands
        self.input_dim = num_channels * num_bands  # 32 * 5 = 160

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),                    # (B, 160)
            nn.Linear(self.input_dim, embed_dim * 2),   # (B, 512)
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),        # (B, 256)
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 32, 5) band energy per player

        Returns:
            (B, embed_dim) embedding
        """
        return self.encoder(x)


# =============================================================================
# Encoder 2: Intra-brain Connectivity Encoder (CNN)
# =============================================================================

class IntraConnEncoder(nn.Module):
    """
    Intra-brain Connectivity Encoder using Simple ConvNet.

    Processes per-player connectivity matrices (32x32 images with 35 channels).
    Input is reshaped from (7, 5, 32, 32) to (35, 32, 32).

    Args:
        num_metrics: Number of connectivity metrics (default: 7)
        num_bands: Number of frequency bands (default: 5)
        embed_dim: Output embedding dimension (default: 256)
        dropout: Dropout rate (default: 0.1)

    Shape:
        Input: (B, 7, 5, 32, 32) per player
        Output: (B, embed_dim)
    """

    def __init__(
        self,
        num_metrics: int = 7,
        num_bands: int = 5,
        embed_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_metrics = num_metrics
        self.num_bands = num_bands
        self.in_channels = num_metrics * num_bands  # 7 * 5 = 35

        self.conv_encoder = nn.Sequential(
            # Conv1: (35, 32, 32) -> (64, 16, 16)
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Conv2: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Conv3: (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Conv4: (256, 4, 4) -> (embed_dim, 2, 2)
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            # Global Average Pooling: (embed_dim, 2, 2) -> (embed_dim, 1, 1)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, 5, 32, 32) connectivity matrices per player

        Returns:
            (B, embed_dim) embedding
        """
        B = x.size(0)
        # Reshape: (B, 7, 5, 32, 32) -> (B, 35, 32, 32)
        x = x.view(B, self.in_channels, 32, 32)
        return self.conv_encoder(x)


# =============================================================================
# Encoder 3: Inter-brain Connectivity Encoder (CNN)
# =============================================================================

class InterConnEncoder(nn.Module):
    """
    Inter-brain Connectivity Encoder using Simple ConvNet.

    Processes inter-brain connectivity matrices (32x32 images with 35 channels).
    Same architecture as IntraConnEncoder but separate weights.

    Args:
        num_metrics: Number of connectivity metrics (default: 7)
        num_bands: Number of frequency bands (default: 5)
        embed_dim: Output embedding dimension (default: 256)
        dropout: Dropout rate (default: 0.1)

    Shape:
        Input: (B, 7, 5, 32, 32)
        Output: (B, embed_dim)
    """

    def __init__(
        self,
        num_metrics: int = 7,
        num_bands: int = 5,
        embed_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_metrics = num_metrics
        self.num_bands = num_bands
        self.in_channels = num_metrics * num_bands  # 7 * 5 = 35

        self.conv_encoder = nn.Sequential(
            # Conv1: (35, 32, 32) -> (64, 16, 16)
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Conv2: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Conv3: (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Conv4: (256, 4, 4) -> (embed_dim, 2, 2)
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, 5, 32, 32) inter-brain connectivity matrices

        Returns:
            (B, embed_dim) embedding
        """
        B = x.size(0)
        # Reshape: (B, 7, 5, 32, 32) -> (B, 35, 32, 32)
        x = x.view(B, self.in_channels, 32, 32)
        return self.conv_encoder(x)


# =============================================================================
# Transformer Fusion Layer
# =============================================================================

class TransformerFusion(nn.Module):
    """
    Transformer-based fusion for cross-modality attention.

    Takes N tokens and applies self-attention to learn cross-modality relationships.
    Uses learnable token type embeddings to distinguish different feature sources.

    Args:
        embed_dim: Token embedding dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of Transformer layers (default: 2)
        num_tokens: Number of input tokens (default: 5)
        dropout: Dropout rate (default: 0.1)

    Shape:
        Input: (B, num_tokens, embed_dim)
        Output: (B, num_tokens, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_tokens: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Learnable token type embeddings
        # Token types: 0=Spec_P1, 1=Spec_P2, 2=Intra_P1, 3=Intra_P2, 4=Inter_12
        self.token_type_embedding = nn.Embedding(num_tokens, embed_dim)

        # Transformer Encoder with Pre-LayerNorm for stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for training stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            tokens: (B, N, D) - N tokens of dimension D
            return_attn: If True, return attention weights (not fully supported)

        Returns:
            (B, N, D) - Tokens with cross-modality attention
        """
        B, N, D = tokens.shape

        # Add token type embeddings
        token_types = torch.arange(N, device=tokens.device)
        type_emb = self.token_type_embedding(token_types)  # (N, D)
        tokens = tokens + type_emb.unsqueeze(0)  # (B, N, D)

        # Apply Transformer
        out = self.transformer(tokens)
        out = self.final_norm(out)

        return out


# =============================================================================
# Multi-Token Uncertainty Fusion
# =============================================================================

class MultiTokenUncertaintyFusion(nn.Module):
    """
    Uncertainty-Aware Fusion for multiple tokens.

    Estimates mean and variance for each token, then performs
    inverse-variance weighted fusion. Extends HyperEEG's UncertaintyFusion
    to handle N tokens instead of just 2.

    Mathematical formulation:
        For each token i:
            μ_i, σ²_i = Estimate(token_i)

        Precision weighting:
            w_i = (1/σ²_i) / Σ(1/σ²_j)

        Weighted fusion:
            Z = Σ(w_i * μ_i)

    Args:
        embed_dim: Input token dimension (default: 256)
        output_dim: Output fused dimension (default: 256)
        dropout: Dropout rate (default: 0.1)

    Shape:
        Input: (B, N, embed_dim)
        Output: (B, output_dim)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Shared mean estimation head
        self.mean_head = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        # Shared variance estimation head (outputs positive values via Softplus)
        self.var_head = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Softplus()  # Ensure positive variance
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            tokens: (B, N, D) - N tokens of dimension D
            return_uncertainty: If True, return uncertainty estimates

        Returns:
            fused: (B, output_dim) - Fused representation
            uncertainties: Optional dict with per-token estimates
        """
        B, N, D = tokens.shape

        # Compute mean and variance for each token
        means = []
        vars = []
        for i in range(N):
            token = tokens[:, i, :]  # (B, D)
            means.append(self.mean_head(token))
            vars.append(self.var_head(token) + 1e-6)  # Add small value for stability

        means = torch.stack(means, dim=1)  # (B, N, output_dim)
        vars = torch.stack(vars, dim=1)    # (B, N, output_dim)

        # Inverse variance weighting
        # w_i = (1/σ²_i) / Σ(1/σ²_j)
        precision = 1.0 / vars  # (B, N, output_dim)
        total_precision = precision.sum(dim=1, keepdim=True)  # (B, 1, output_dim)
        weights = precision / total_precision  # (B, N, output_dim)

        # Weighted fusion
        fused = (weights * means).sum(dim=1)  # (B, output_dim)

        if return_uncertainty:
            uncertainties = {
                'means': means,
                'vars': vars,
                'weights': weights
            }
            return fused, uncertainties

        return fused


class SimpleFusion(nn.Module):
    """
    Simple fusion baseline (fallback when use_uncertainty=False).

    Uses mean pooling over tokens with optional projection.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, None]]:
        """
        Simple mean pooling fusion.

        Args:
            tokens: (B, N, D)

        Returns:
            (B, output_dim)
        """
        # Mean pooling over tokens
        pooled = tokens.mean(dim=1)  # (B, D)

        # Project
        fused = self.projection(pooled)  # (B, output_dim)

        if return_uncertainty:
            return fused, None
        return fused


# =============================================================================
# Main Model: MultiStreamFeatureTransformer
# =============================================================================

class MultiStreamFeatureTransformer(nn.Module):
    """
    Multi-Stream Feature Transformer for Hyperscanning EEG Classification.

    Processes pre-extracted EEG features through specialized encoders,
    fuses them via Transformer attention, and applies uncertainty-aware fusion.

    Args:
        embed_dim: Embedding dimension for all tokens (default: 256)
        num_heads: Number of attention heads in Transformer (default: 8)
        num_layers: Number of Transformer encoder layers (default: 2)
        num_classes: Number of output classes (default: 3)
        dropout: Dropout rate (default: 0.1)
        num_channels: Number of EEG channels (default: 32)
        num_bands: Number of frequency bands (default: 5)
        num_metrics: Number of connectivity metrics (default: 7)

        # Ablation switches
        use_spectral: Enable Spectral Encoder (default: True)
        use_intra: Enable Intra-Conn Encoder (default: True)
        use_inter: Enable Inter-Conn Encoder (default: True)
        use_transformer_fusion: Use Transformer fusion (default: True)
        use_uncertainty: Use Uncertainty Fusion (default: True)

    Input:
        features: Dict containing:
            - 'bands_energy': (B, 2, 32, 5)
            - 'intra_con': (B, 2, 7, 5, 32, 32)
            - 'inter_con': (B, 7, 5, 32, 32)

    Output:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
        num_channels: int = 32,
        num_bands: int = 5,
        num_metrics: int = 7,
        # Ablation switches
        use_spectral: bool = True,
        use_intra: bool = True,
        use_inter: bool = True,
        use_transformer_fusion: bool = True,
        use_uncertainty: bool = True
    ):
        super().__init__()

        # Save configuration
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_spectral = use_spectral
        self.use_intra = use_intra
        self.use_inter = use_inter
        self.use_transformer_fusion = use_transformer_fusion
        self.use_uncertainty = use_uncertainty

        # Count number of active tokens
        self.num_tokens = 0
        if use_spectral:
            self.num_tokens += 2  # Spec_P1, Spec_P2
        if use_intra:
            self.num_tokens += 2  # Intra_P1, Intra_P2
        if use_inter:
            self.num_tokens += 1  # Inter_12

        assert self.num_tokens > 0, "At least one encoder must be enabled"

        # ============================================================
        # Encoders (Shared weights for P1 and P2 within each encoder)
        # ============================================================

        if use_spectral:
            self.spectral_encoder = SpectralEncoder(
                num_channels=num_channels,
                num_bands=num_bands,
                embed_dim=embed_dim,
                dropout=dropout
            )

        if use_intra:
            self.intra_encoder = IntraConnEncoder(
                num_metrics=num_metrics,
                num_bands=num_bands,
                embed_dim=embed_dim,
                dropout=dropout
            )

        if use_inter:
            self.inter_encoder = InterConnEncoder(
                num_metrics=num_metrics,
                num_bands=num_bands,
                embed_dim=embed_dim,
                dropout=dropout
            )

        # ============================================================
        # Transformer Fusion (optional)
        # ============================================================

        if use_transformer_fusion:
            self.transformer_fusion = TransformerFusion(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_tokens=self.num_tokens,
                dropout=dropout
            )

        # ============================================================
        # Uncertainty Fusion or Simple Fusion
        # ============================================================

        if use_uncertainty:
            self.fusion = MultiTokenUncertaintyFusion(
                embed_dim=embed_dim,
                output_dim=embed_dim,
                dropout=dropout
            )
        else:
            self.fusion = SimpleFusion(
                embed_dim=embed_dim,
                output_dim=embed_dim,
                dropout=dropout
            )

        # ============================================================
        # Classifier Head
        # ============================================================

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of Multi-Stream Feature Transformer.

        Args:
            features: Dict containing:
                - 'bands_energy': (B, 2, 32, 5) - Players x Channels x Bands
                - 'intra_con': (B, 2, 7, 5, 32, 32) - Players x Metrics x Bands x Ch x Ch
                - 'inter_con': (B, 7, 5, 32, 32) - Metrics x Bands x Ch_A x Ch_B

        Returns:
            logits: (B, num_classes)
        """
        tokens = []

        # ============================================================
        # Encoder 1: Spectral/Band Energy (MLP)
        # ============================================================
        if self.use_spectral:
            bands_energy = features['bands_energy']  # (B, 2, 32, 5)

            # Player 1: (B, 32, 5) -> (B, D)
            spec_p1 = self.spectral_encoder(bands_energy[:, 0])
            # Player 2: (B, 32, 5) -> (B, D)
            spec_p2 = self.spectral_encoder(bands_energy[:, 1])

            tokens.extend([spec_p1, spec_p2])

        # ============================================================
        # Encoder 2: Intra-brain Connectivity (ConvNet)
        # ============================================================
        if self.use_intra:
            intra_con = features['intra_con']  # (B, 2, 7, 5, 32, 32)

            # Player 1: (B, 7, 5, 32, 32) -> (B, D)
            intra_p1 = self.intra_encoder(intra_con[:, 0])
            # Player 2: (B, 7, 5, 32, 32) -> (B, D)
            intra_p2 = self.intra_encoder(intra_con[:, 1])

            tokens.extend([intra_p1, intra_p2])

        # ============================================================
        # Encoder 3: Inter-brain Connectivity (ConvNet)
        # ============================================================
        if self.use_inter:
            inter_con = features['inter_con']  # (B, 7, 5, 32, 32)

            # (B, 7, 5, 32, 32) -> (B, D)
            inter_emb = self.inter_encoder(inter_con)

            tokens.append(inter_emb)

        # ============================================================
        # Stack tokens: List of (B, D) -> (B, N, D)
        # ============================================================
        tokens = torch.stack(tokens, dim=1)  # (B, N, D)

        # ============================================================
        # Transformer Fusion (optional)
        # ============================================================
        if self.use_transformer_fusion:
            tokens = self.transformer_fusion(tokens)  # (B, N, D)

        # ============================================================
        # Uncertainty Fusion or Simple Pooling
        # ============================================================
        fused = self.fusion(tokens)  # (B, D)

        # ============================================================
        # Classifier
        # ============================================================
        logits = self.classifier(fused)  # (B, num_classes)

        return logits

    def get_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for visualization/analysis.

        Returns:
            Dict containing:
                - 'spectral_p1', 'spectral_p2': (B, D) spectral embeddings
                - 'intra_p1', 'intra_p2': (B, D) intra-brain embeddings
                - 'inter': (B, D) inter-brain embedding
                - 'tokens_pre_transformer': (B, N, D)
                - 'tokens_post_transformer': (B, N, D)
                - 'fused': (B, D)
        """
        result = {}
        tokens = []

        if self.use_spectral:
            bands_energy = features['bands_energy']
            spec_p1 = self.spectral_encoder(bands_energy[:, 0])
            spec_p2 = self.spectral_encoder(bands_energy[:, 1])
            result['spectral_p1'] = spec_p1
            result['spectral_p2'] = spec_p2
            tokens.extend([spec_p1, spec_p2])

        if self.use_intra:
            intra_con = features['intra_con']
            intra_p1 = self.intra_encoder(intra_con[:, 0])
            intra_p2 = self.intra_encoder(intra_con[:, 1])
            result['intra_p1'] = intra_p1
            result['intra_p2'] = intra_p2
            tokens.extend([intra_p1, intra_p2])

        if self.use_inter:
            inter_con = features['inter_con']
            inter_emb = self.inter_encoder(inter_con)
            result['inter'] = inter_emb
            tokens.append(inter_emb)

        tokens = torch.stack(tokens, dim=1)
        result['tokens_pre_transformer'] = tokens.clone()

        if self.use_transformer_fusion:
            tokens = self.transformer_fusion(tokens)
        result['tokens_post_transformer'] = tokens

        fused = self.fusion(tokens)
        result['fused'] = fused

        return result

    def get_uncertainty(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get uncertainty estimates for each token.

        Returns:
            Dict with 'means', 'vars', 'weights' if use_uncertainty=True, else None
        """
        if not self.use_uncertainty:
            return None

        tokens = []

        if self.use_spectral:
            bands_energy = features['bands_energy']
            tokens.extend([
                self.spectral_encoder(bands_energy[:, 0]),
                self.spectral_encoder(bands_energy[:, 1])
            ])

        if self.use_intra:
            intra_con = features['intra_con']
            tokens.extend([
                self.intra_encoder(intra_con[:, 0]),
                self.intra_encoder(intra_con[:, 1])
            ])

        if self.use_inter:
            inter_con = features['inter_con']
            tokens.append(self.inter_encoder(inter_con))

        tokens = torch.stack(tokens, dim=1)

        if self.use_transformer_fusion:
            tokens = self.transformer_fusion(tokens)

        _, uncertainties = self.fusion(tokens, return_uncertainty=True)

        return uncertainties


# =============================================================================
# Factory Functions and Ablation Configurations
# =============================================================================

ABLATION_CONFIGS = {
    # Full model - all components enabled
    'full': {
        'use_spectral': True,
        'use_intra': True,
        'use_inter': True,
        'use_transformer_fusion': True,
        'use_uncertainty': True
    },

    # Baseline - no Transformer, no uncertainty
    'baseline': {
        'use_spectral': True,
        'use_intra': True,
        'use_inter': True,
        'use_transformer_fusion': False,
        'use_uncertainty': False
    },

    # Single encoder ablations
    'spectral_only': {
        'use_spectral': True,
        'use_intra': False,
        'use_inter': False,
        'use_transformer_fusion': False,
        'use_uncertainty': False
    },
    'intra_only': {
        'use_spectral': False,
        'use_intra': True,
        'use_inter': False,
        'use_transformer_fusion': False,
        'use_uncertainty': False
    },
    'inter_only': {
        'use_spectral': False,
        'use_intra': False,
        'use_inter': True,
        'use_transformer_fusion': False,
        'use_uncertainty': False
    },

    # Remove single component
    'no_spectral': {
        'use_spectral': False,
        'use_intra': True,
        'use_inter': True,
        'use_transformer_fusion': True,
        'use_uncertainty': True
    },
    'no_intra': {
        'use_spectral': True,
        'use_intra': False,
        'use_inter': True,
        'use_transformer_fusion': True,
        'use_uncertainty': True
    },
    'no_inter': {
        'use_spectral': True,
        'use_intra': True,
        'use_inter': False,
        'use_transformer_fusion': True,
        'use_uncertainty': True
    },
    'no_transformer': {
        'use_spectral': True,
        'use_intra': True,
        'use_inter': True,
        'use_transformer_fusion': False,
        'use_uncertainty': True
    },
    'no_uncertainty': {
        'use_spectral': True,
        'use_intra': True,
        'use_inter': True,
        'use_transformer_fusion': True,
        'use_uncertainty': False
    },

    # Dual-stream variants (for HyperEEG comparison)
    'dual_intra': {
        'use_spectral': False,
        'use_intra': True,
        'use_inter': False,
        'use_transformer_fusion': True,
        'use_uncertainty': True
    },
    'intra_inter': {
        'use_spectral': False,
        'use_intra': True,
        'use_inter': True,
        'use_transformer_fusion': True,
        'use_uncertainty': True
    },
    'spectral_intra': {
        'use_spectral': True,
        'use_intra': True,
        'use_inter': False,
        'use_transformer_fusion': True,
        'use_uncertainty': True
    }
}


def create_msft_model(
    config_name: str = 'full',
    **kwargs
) -> MultiStreamFeatureTransformer:
    """
    Create Multi-Stream Feature Transformer with predefined ablation configuration.

    Args:
        config_name: One of the keys in ABLATION_CONFIGS
        **kwargs: Override any model parameter

    Returns:
        Configured MultiStreamFeatureTransformer instance

    Example:
        # Full model
        model = create_msft_model('full')

        # Baseline for comparison
        model = create_msft_model('baseline')

        # Custom configuration
        model = create_msft_model('full', embed_dim=512, num_heads=16)
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(ABLATION_CONFIGS.keys())}")

    config = ABLATION_CONFIGS[config_name].copy()
    config.update(kwargs)

    return MultiStreamFeatureTransformer(**config)


def get_model_config(model: MultiStreamFeatureTransformer) -> Dict[str, bool]:
    """
    Get the ablation configuration of a model.

    Args:
        model: MultiStreamFeatureTransformer instance

    Returns:
        Dictionary with ablation switch states
    """
    return {
        'use_spectral': model.use_spectral,
        'use_intra': model.use_intra,
        'use_inter': model.use_inter,
        'use_transformer_fusion': model.use_transformer_fusion,
        'use_uncertainty': model.use_uncertainty
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Multi-Stream Feature Transformer")
    print("=" * 60)

    # Test configuration
    batch_size = 4
    num_channels = 32
    num_bands = 5
    num_metrics = 7
    num_classes = 3

    # Create dummy input features
    features = {
        'bands_energy': torch.randn(batch_size, 2, num_channels, num_bands),
        'intra_con': torch.randn(batch_size, 2, num_metrics, num_bands, 32, 32),
        'inter_con': torch.randn(batch_size, num_metrics, num_bands, 32, 32)
    }

    print(f"\nInput shapes:")
    print(f"  bands_energy: {features['bands_energy'].shape}")
    print(f"  intra_con: {features['intra_con'].shape}")
    print(f"  inter_con: {features['inter_con'].shape}")

    # Test all ablation configurations
    for config_name in ABLATION_CONFIGS.keys():
        print(f"\n{'─' * 40}")
        print(f"Testing config: {config_name}")
        print(f"{'─' * 40}")

        try:
            model = create_msft_model(config_name)
            config = get_model_config(model)

            print(f"  Spectral: {config['use_spectral']}")
            print(f"  Intra: {config['use_intra']}")
            print(f"  Inter: {config['use_inter']}")
            print(f"  Transformer: {config['use_transformer_fusion']}")
            print(f"  Uncertainty: {config['use_uncertainty']}")
            print(f"  Num tokens: {model.num_tokens}")

            # Forward pass
            model.eval()
            with torch.no_grad():
                logits = model(features)

            print(f"  Output shape: {logits.shape}")
            assert logits.shape == (batch_size, num_classes), \
                f"Expected ({batch_size}, {num_classes}), got {logits.shape}"

            # Test feature extraction
            feats = model.get_features(features)
            print(f"  Feature extraction keys: {list(feats.keys())}")

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
