"""
HyperEEG Encoder for Hyperscanning EEG Classification

This module implements a dual-stream architecture for social interaction classification
using hyperscanning EEG data from two participants.

Architecture based on fig3.EEGArch.png with 4 progressive stages:
- M1: SincConv1d (Learnable band-pass filtering)
- M2: IntraGraphBlock (Intra-brain topology via Self-Attention)
- M3: InterBrainCrossAttn (Inter-brain interaction via Cross-Attention)
- M4: UncertaintyFusion (Uncertainty-aware weighted fusion)

Supports ablation studies via boolean switches (use_sinc, use_graph, use_cross_attn, use_uncertainty).

Input:
    x1, x2: EEG signals from two participants, shape (B, Channels, Timepoints)
    Default: (B, 32, 1024)

Output:
    logits: Classification logits, shape (B, num_classes)
    Default: (B, 3) for {Single, Competition, Cooperation}

Author: Generated for Gaze-EEG Multimodal Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, Optional, Union


# =============================================================================
# M1: Learnable Temporal Encoding - SincConv1d
# =============================================================================

class SincConv1d(nn.Module):
    """
    Sinc-based Convolutional Layer for Learnable Band-pass Filtering.

    Based on "Speaker Recognition from Raw Waveform with SincNet" (Ravanelli & Bengio, 2018).
    The filter is parameterized by learnable low and high cutoff frequencies.

    Mathematical Formulation:
        h[n] = 2*f_high*sinc(2*pi*f_high*n) - 2*f_low*sinc(2*pi*f_low*n)
        where sinc(x) = sin(x)/x (with sinc(0) = 1)

        The filter is multiplied by a Hamming window for smooth frequency response.

    Args:
        in_channels: Number of input channels (EEG channels)
        out_channels: Number of output channels (filters per input channel)
        kernel_size: Length of the sinc filter (should be odd)
        stride: Stride for downsampling
        sample_rate: Sampling rate of the EEG signal (Hz)
        min_low_hz: Minimum low cutoff frequency (Hz)
        min_band_hz: Minimum bandwidth (Hz)
        padding_mode: Padding mode for convolution

    Shape:
        Input: (B, in_channels, T)
        Output: (B, out_channels, T') where T' = (T - kernel_size) / stride + 1
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        kernel_size: int = 251,
        stride: int = 8,
        sample_rate: int = 500,
        min_low_hz: float = 1.0,
        min_band_hz: float = 4.0,
        padding_mode: str = 'zeros'
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.padding = kernel_size // 2

        # Initialize learnable frequency parameters
        # Each output channel learns its own band-pass filter
        # low_hz will be: min_low_hz + softplus(low_hz_)
        # band_hz will be: min_band_hz + softplus(band_hz_)

        # Initialize low frequencies uniformly between min_low_hz and sample_rate/2 - min_band_hz
        low_hz_init = min_low_hz + np.abs(
            np.random.randn(out_channels) * (sample_rate / 2 - min_low_hz - min_band_hz) / 4
        )
        self.low_hz_ = nn.Parameter(torch.from_numpy(low_hz_init).float())

        # Initialize bandwidths
        band_hz_init = np.abs(np.random.randn(out_channels) * (sample_rate / 4 - min_band_hz) / 4) + min_band_hz
        self.band_hz_ = nn.Parameter(torch.from_numpy(band_hz_init).float())

        # Hamming window (fixed, not learnable)
        n = torch.arange(kernel_size).float()
        self.register_buffer(
            'hamming_window',
            0.54 - 0.46 * torch.cos(2 * math.pi * n / (kernel_size - 1))
        )

        # Time indices for sinc function
        # Centered at 0: [-kernel_size//2, ..., 0, ..., kernel_size//2]
        n_half = kernel_size // 2
        self.register_buffer(
            'n_',
            (torch.arange(kernel_size).float() - n_half) / sample_rate
        )

    def _sinc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sinc function with numerical stability.
        sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
        """
        # Avoid division by zero
        x_safe = torch.where(x == 0, torch.ones_like(x) * 1e-10, x)
        return torch.sin(math.pi * x_safe) / (math.pi * x_safe)

    def _get_sinc_filters(self) -> torch.Tensor:
        """
        Construct band-pass filters from learnable parameters.

        Returns:
            filters: Shape (out_channels, 1, kernel_size)
        """
        # Get actual frequencies using softplus to ensure positive values
        low_hz = self.min_low_hz + F.softplus(self.low_hz_)  # (out_channels,)
        high_hz = low_hz + self.min_band_hz + F.softplus(self.band_hz_)  # (out_channels,)

        # Clamp to valid range [min_low_hz, sample_rate/2]
        low_hz = torch.clamp(low_hz, self.min_low_hz, self.sample_rate / 2)
        high_hz = torch.clamp(high_hz, self.min_low_hz, self.sample_rate / 2)

        # Construct band-pass filter for each output channel
        # h[n] = 2*f_high*sinc(2*f_high*n) - 2*f_low*sinc(2*f_low*n)
        # Shape: (out_channels, kernel_size)
        low_pass_low = 2 * low_hz.unsqueeze(1) * self._sinc(2 * low_hz.unsqueeze(1) * self.n_)
        low_pass_high = 2 * high_hz.unsqueeze(1) * self._sinc(2 * high_hz.unsqueeze(1) * self.n_)

        # Band-pass = high_cutoff_lowpass - low_cutoff_lowpass
        band_pass = low_pass_high - low_pass_low  # (out_channels, kernel_size)

        # Apply Hamming window
        band_pass = band_pass * self.hamming_window  # (out_channels, kernel_size)

        # Normalize filters
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)

        # Reshape for depthwise convolution: (out_channels, 1, kernel_size)
        return band_pass.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable sinc band-pass filters.

        Args:
            x: Input tensor, shape (B, in_channels, T)

        Returns:
            Filtered tensor, shape (B, out_channels, T')
        """
        # x: (B, C, T) e.g., (B, 32, 1024)

        # Get sinc filters: (out_channels, 1, kernel_size)
        filters = self._get_sinc_filters()
        # filters: (32, 1, 251)

        # Apply depthwise convolution (each channel filtered independently)
        # For depthwise conv: groups=in_channels, each input channel gets its own filter
        # Output shape: (B, out_channels, T')
        out = F.conv1d(
            x,
            filters,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=min(self.in_channels, self.out_channels)
        )
        # out: (B, 32, T') where T' = (1024 - 251 + 2*125) / 8 + 1 ≈ 125

        return out


class TemporalEncoderBaseline(nn.Module):
    """
    Baseline temporal encoder using standard Conv1d (fallback when use_sinc=False).

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Stride for downsampling
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        kernel_size: int = 251,
        stride: int = 8
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size // 2, groups=in_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T')
        """
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    Complete temporal encoding block with projection to embed_dim.

    Combines SincConv/Conv1d with adaptive pooling and linear projection.

    Args:
        in_channels: Number of EEG channels
        embed_dim: Output embedding dimension
        use_sinc: Whether to use SincConv1d or baseline Conv1d
        kernel_size: Filter kernel size
        stride: Convolution stride
        sample_rate: EEG sampling rate
    """

    def __init__(
        self,
        in_channels: int = 32,
        embed_dim: int = 128,
        use_sinc: bool = True,
        kernel_size: int = 251,
        stride: int = 8,
        sample_rate: int = 500
    ):
        super().__init__()

        self.use_sinc = use_sinc

        # M1 check: SincConv vs Conv1d
        if use_sinc:
            self.temporal_conv = SincConv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                sample_rate=sample_rate
            )
        else:
            self.temporal_conv = TemporalEncoderBaseline(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride
            )

        # Batch normalization after temporal convolution
        self.bn = nn.BatchNorm1d(in_channels)

        # Adaptive pooling to fixed time dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(embed_dim)

        # Optional: Additional 1x1 conv for feature mixing
        self.feature_proj = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) e.g., (B, 32, 1024)

        Returns:
            (B, C, embed_dim) e.g., (B, 32, 128)
        """
        # x: (B, 32, 1024)

        # Temporal convolution with downsampling
        x = self.temporal_conv(x)
        # x: (B, 32, ~125) after stride=8

        x = self.bn(x)
        x = F.gelu(x)
        # x: (B, 32, ~125)

        # Adaptive pooling to fixed dimension
        x = self.adaptive_pool(x)
        # x: (B, 32, 128)

        # Feature projection
        x = self.feature_proj(x)
        # x: (B, 32, 128)

        return x


# =============================================================================
# M2: Intra-Brain Topology - Self-Attention Graph
# =============================================================================

class IntraGraphBlock(nn.Module):
    """
    Intra-Brain Topology Block using Self-Attention.

    Models spatial connectivity between EEG channels by treating each channel
    as a graph node and using self-attention to learn the adjacency matrix.

    The attention matrix can be interpreted as learned functional connectivity.

    Args:
        embed_dim: Feature dimension per channel
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_residual: Whether to use residual connection

    Shape:
        Input: (B, num_channels, embed_dim)
        Output: (B, num_channels, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_residual = use_residual

        # Multi-head self-attention
        # Treats channels as sequence tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, C, D) - C channels, D features
            return_attn: If True, also return attention weights

        Returns:
            out: (B, C, D)
            attn_weights: (B, C, C) if return_attn=True
        """
        # x: (B, 32, 128) - treating 32 channels as sequence, 128 as features

        # Self-attention among channels
        identity = x
        x_norm = self.norm1(x)
        # x_norm: (B, 32, 128)

        attn_out, attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=True,
            average_attn_weights=True
        )
        # attn_out: (B, 32, 128)
        # attn_weights: (B, 32, 32) - learned connectivity matrix

        x = identity + self.dropout(attn_out) if self.use_residual else attn_out
        # x: (B, 32, 128)

        # Feed-forward with residual
        identity = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = identity + x if self.use_residual else x
        # x: (B, 32, 128)

        if return_attn:
            return x, attn_weights
        return x


class IntraGraphBaseline(nn.Module):
    """
    Baseline for Intra-Brain processing (fallback when use_graph=False).

    Uses simple flatten + linear projection instead of graph attention.
    """

    def __init__(
        self,
        num_channels: int = 32,
        embed_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Flatten and project, then reshape back
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),  # (B, C*D)
            nn.Linear(num_channels * embed_dim, num_channels * embed_dim),
            nn.LayerNorm(num_channels * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, None]]:
        """
        Args:
            x: (B, C, D)

        Returns:
            out: (B, C, D)
        """
        B = x.size(0)
        # x: (B, 32, 128)

        x = self.projection(x)
        # x: (B, 32*128) = (B, 4096)

        x = x.view(B, self.num_channels, self.embed_dim)
        # x: (B, 32, 128)

        if return_attn:
            return x, None
        return x


# =============================================================================
# M3: Inter-Brain Interaction - Cross-Attention
# =============================================================================

class InterBrainCrossAttn(nn.Module):
    """
    Inter-Brain Cross-Attention Block.

    Models inter-brain synchrony by allowing each participant's features
    to attend to the other participant's features.

    Bidirectional cross-attention:
        - Person A queries Person B: Q_A attends to K_B, V_B
        - Person B queries Person A: Q_B attends to K_A, V_A

    Args:
        embed_dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_residual: Whether to use residual connections

    Shape:
        Input: feat_a (B, C, D), feat_b (B, C, D)
        Output: out_a (B, C, D), out_b (B, C, D)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_residual = use_residual

        # Cross-attention: A attends to B
        self.cross_attn_a2b = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: B attends to A
        self.cross_attn_b2a = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm_a1 = nn.LayerNorm(embed_dim)
        self.norm_b1 = nn.LayerNorm(embed_dim)
        self.norm_a2 = nn.LayerNorm(embed_dim)
        self.norm_b2 = nn.LayerNorm(embed_dim)

        # Feed-forward networks (separate for A and B)
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.ffn_b = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        return_attn: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            feat_a: Features from Person A, shape (B, C, D)
            feat_b: Features from Person B, shape (B, C, D)
            return_attn: If True, return attention weights

        Returns:
            out_a: Updated features for A, shape (B, C, D)
            out_b: Updated features for B, shape (B, C, D)
            attn_a2b: Attention weights A->B, shape (B, C, C) if return_attn
            attn_b2a: Attention weights B->A, shape (B, C, C) if return_attn
        """
        # feat_a, feat_b: (B, 32, 128)

        # Store identity for residual connection
        identity_a = feat_a
        identity_b = feat_b

        # Normalize
        feat_a_norm = self.norm_a1(feat_a)
        feat_b_norm = self.norm_b1(feat_b)
        # feat_a_norm, feat_b_norm: (B, 32, 128)

        # Cross-attention: A queries B (A attends to B's information)
        cross_a, attn_a2b = self.cross_attn_a2b(
            query=feat_a_norm,
            key=feat_b_norm,
            value=feat_b_norm,
            need_weights=True,
            average_attn_weights=True
        )
        # cross_a: (B, 32, 128) - A's features enriched with B's information
        # attn_a2b: (B, 32, 32) - how A's channels attend to B's channels

        # Cross-attention: B queries A
        cross_b, attn_b2a = self.cross_attn_b2a(
            query=feat_b_norm,
            key=feat_a_norm,
            value=feat_a_norm,
            need_weights=True,
            average_attn_weights=True
        )
        # cross_b: (B, 32, 128)
        # attn_b2a: (B, 32, 32)

        # Residual connection
        if self.use_residual:
            out_a = identity_a + self.dropout(cross_a)
            out_b = identity_b + self.dropout(cross_b)
        else:
            out_a = cross_a
            out_b = cross_b
        # out_a, out_b: (B, 32, 128)

        # Feed-forward with residual
        identity_a = out_a
        identity_b = out_b

        out_a = self.norm_a2(out_a)
        out_a = self.ffn_a(out_a)
        out_a = identity_a + out_a if self.use_residual else out_a

        out_b = self.norm_b2(out_b)
        out_b = self.ffn_b(out_b)
        out_b = identity_b + out_b if self.use_residual else out_b
        # out_a, out_b: (B, 32, 128)

        if return_attn:
            return out_a, out_b, attn_a2b, attn_b2a

        return out_a, out_b


class InterBrainBaseline(nn.Module):
    """
    Baseline for Inter-Brain interaction (fallback when use_cross_attn=False).

    Simply passes through features without cross-attention.
    The fusion happens later in Stage 4.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        return_attn: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, None, None]]:
        """
        Pass-through baseline - no cross-interaction.
        """
        if return_attn:
            return feat_a, feat_b, None, None
        return feat_a, feat_b


# =============================================================================
# M4: Uncertainty Estimation & Fusion
# =============================================================================

class UncertaintyFusion(nn.Module):
    """
    Uncertainty-Aware Fusion Module.

    Estimates mean and variance for each participant's features,
    then performs inverse-variance weighted fusion.

    Mathematical formulation:
        μ_A, σ²_A = Estimate(pooled_A)
        μ_B, σ²_B = Estimate(pooled_B)

        w_A = σ²_B / (σ²_A + σ²_B + ε)  # Higher weight if lower uncertainty
        w_B = σ²_A / (σ²_A + σ²_B + ε)

        Z = w_A * μ_A + w_B * μ_B

    Args:
        input_dim: Input feature dimension (after pooling)
        output_dim: Output fused dimension
        dropout: Dropout probability

    Shape:
        Input: feat_a (B, C, D), feat_b (B, C, D)
        Output: fused (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Mean estimation head
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        # Variance estimation head (outputs positive values via Softplus)
        self.var_head = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Softplus()  # Ensure positive variance
        )

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Graph pooling using Mean + Max.

        Args:
            x: (B, C, D)

        Returns:
            pooled: (B, D)
        """
        mean_pool = x.mean(dim=1)       # (B, D)
        max_pool = x.max(dim=1)[0]      # (B, D)
        return mean_pool + max_pool     # (B, D)

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            feat_a: Features from Person A, shape (B, C, D)
            feat_b: Features from Person B, shape (B, C, D)
            return_uncertainty: If True, return uncertainty estimates

        Returns:
            fused: Fused features, shape (B, output_dim)
            uncertainties: Dict with variance estimates if return_uncertainty
        """
        # feat_a, feat_b: (B, 32, 128)

        # Graph pooling: Mean + Max
        pooled_a = self._pool_features(feat_a)
        pooled_b = self._pool_features(feat_b)
        # pooled_a, pooled_b: (B, 128)

        # Estimate mean and variance for A
        mean_a = self.mean_head(pooled_a)
        var_a = self.var_head(pooled_a) + 1e-6  # Add small value for stability
        # mean_a, var_a: (B, 128)

        # Estimate mean and variance for B
        mean_b = self.mean_head(pooled_b)
        var_b = self.var_head(pooled_b) + 1e-6
        # mean_b, var_b: (B, 128)

        # Inverse variance weighting
        # w_A = σ²_B / (σ²_A + σ²_B), w_B = σ²_A / (σ²_A + σ²_B)
        total_var = var_a + var_b
        weight_a = var_b / total_var  # Higher weight if B has higher uncertainty
        weight_b = var_a / total_var  # Higher weight if A has higher uncertainty
        # weight_a, weight_b: (B, 128)

        # Weighted fusion
        fused = weight_a * mean_a + weight_b * mean_b
        # fused: (B, 128)

        if return_uncertainty:
            uncertainties = {
                'var_a': var_a,
                'var_b': var_b,
                'weight_a': weight_a,
                'weight_b': weight_b,
                'mean_a': mean_a,
                'mean_b': mean_b
            }
            return fused, uncertainties

        return fused


class FusionBaseline(nn.Module):
    """
    Baseline fusion module (fallback when use_uncertainty=False).

    Simply averages pooled features from both participants.
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, None]]:
        """
        Simple average fusion baseline.
        """
        # feat_a, feat_b: (B, C, D)

        # Mean pooling
        pooled_a = feat_a.mean(dim=1)  # (B, D)
        pooled_b = feat_b.mean(dim=1)  # (B, D)

        # Simple average
        fused = (pooled_a + pooled_b) / 2  # (B, D)

        # Project
        fused = self.projection(fused)  # (B, output_dim)

        if return_uncertainty:
            return fused, None
        return fused


# =============================================================================
# Main Model: HyperEEG_Encoder
# =============================================================================

class HyperEEG_Encoder(nn.Module):
    """
    HyperEEG Encoder for Hyperscanning EEG Social Interaction Classification.

    A dual-stream architecture with 4 progressive stages:
        - Stage 1 (M1): Learnable Temporal Encoding (SincConv1d / Conv1d)
        - Stage 2 (M2): Intra-Brain Topology (Self-Attention Graph / Linear)
        - Stage 3 (M3): Inter-Brain Interaction (Cross-Attention / Pass-through)
        - Stage 4 (M4): Uncertainty Fusion (Variance-weighted / Average)

    Supports ablation studies via boolean switches.

    Args:
        in_channels: Number of EEG channels (default: 32)
        in_timepoints: Number of timepoints per segment (default: 1024)
        num_classes: Number of output classes (default: 3)
        embed_dim: Feature embedding dimension (default: 128)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
        use_sinc: M1 - Use SincConv1d (True) or Conv1d (False)
        use_graph: M2 - Use Graph Attention (True) or Linear (False)
        use_cross_attn: M3 - Use Cross-Attention (True) or Pass-through (False)
        use_uncertainty: M4 - Use Uncertainty Fusion (True) or Average (False)
        sample_rate: EEG sampling rate in Hz (default: 500)
        sinc_kernel_size: SincConv kernel size (default: 251)

    Input:
        x1: EEG from Person A, shape (B, in_channels, in_timepoints)
        x2: EEG from Person B, shape (B, in_channels, in_timepoints)

    Output:
        logits: Classification logits, shape (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 32,
        in_timepoints: int = 1024,
        num_classes: int = 3,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_sinc: bool = True,
        use_graph: bool = True,
        use_cross_attn: bool = True,
        use_uncertainty: bool = True,
        sample_rate: int = 500,
        sinc_kernel_size: int = 251
    ):
        super().__init__()

        # Save configuration
        self.in_channels = in_channels
        self.in_timepoints = in_timepoints
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_sinc = use_sinc
        self.use_graph = use_graph
        self.use_cross_attn = use_cross_attn
        self.use_uncertainty = use_uncertainty

        # ============================================================
        # Stage 1: Learnable Temporal Encoding (M1)
        # Shared weights between two streams
        # ============================================================
        self.temporal_encoder = TemporalBlock(
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_sinc=use_sinc,
            kernel_size=sinc_kernel_size,
            stride=8,
            sample_rate=sample_rate
        )

        # ============================================================
        # Stage 2: Intra-Brain Topology (M2)
        # Shared weights between two streams
        # ============================================================
        if use_graph:
            self.intra_brain = IntraGraphBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_residual=True
            )
        else:
            self.intra_brain = IntraGraphBaseline(
                num_channels=in_channels,
                embed_dim=embed_dim,
                dropout=dropout
            )

        # ============================================================
        # Stage 3: Inter-Brain Interaction (M3)
        # ============================================================
        if use_cross_attn:
            self.inter_brain = InterBrainCrossAttn(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_residual=True
            )
        else:
            self.inter_brain = InterBrainBaseline()

        # ============================================================
        # Stage 4: Uncertainty Fusion (M4)
        # ============================================================
        if use_uncertainty:
            self.fusion = UncertaintyFusion(
                input_dim=embed_dim,
                output_dim=embed_dim,
                dropout=dropout
            )
        else:
            self.fusion = FusionBaseline(
                input_dim=embed_dim,
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
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of HyperEEG Encoder.

        Args:
            x1: EEG from Person A, shape (B, 32, 1024)
            x2: EEG from Person B, shape (B, 32, 1024)

        Returns:
            logits: Classification logits, shape (B, 3)
        """
        # ============================================================
        # Stage 1: Learnable Temporal Encoding (M1)
        # ============================================================
        # x1: (B, 32, 1024) -> (B, 32, 128)
        # x2: (B, 32, 1024) -> (B, 32, 128)
        x1 = self.temporal_encoder(x1)
        x2 = self.temporal_encoder(x2)  # Shared weights
        # x1, x2: (B, 32, 128)

        # ============================================================
        # Stage 2: Intra-Brain Topology (M2)
        # ============================================================
        # x1: (B, 32, 128) -> (B, 32, 128)
        # x2: (B, 32, 128) -> (B, 32, 128)
        x1 = self.intra_brain(x1)
        x2 = self.intra_brain(x2)  # Shared weights
        # x1, x2: (B, 32, 128)

        # ============================================================
        # Stage 3: Inter-Brain Cross Attention (M3)
        # ============================================================
        # x1, x2: (B, 32, 128) × 2 -> (B, 32, 128) × 2
        x1, x2 = self.inter_brain(x1, x2)
        # x1, x2: (B, 32, 128)

        # ============================================================
        # Stage 4: Uncertainty Estimation & Fusion (M4)
        # ============================================================
        # x1, x2: (B, 32, 128) × 2 -> fused: (B, 128)
        fused = self.fusion(x1, x2)
        # fused: (B, 128)

        # ============================================================
        # Classifier Head
        # ============================================================
        # fused: (B, 128) -> logits: (B, 3)
        logits = self.classifier(fused)
        # logits: (B, 3)

        return logits

    def get_features(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for visualization/analysis.

        Args:
            x1: EEG from Person A, shape (B, 32, 1024)
            x2: EEG from Person B, shape (B, 32, 1024)

        Returns:
            Dictionary containing intermediate features:
                - 'temporal_a', 'temporal_b': After Stage 1
                - 'intra_a', 'intra_b': After Stage 2
                - 'inter_a', 'inter_b': After Stage 3
                - 'fused': After Stage 4
        """
        features = {}

        # Stage 1
        temporal_a = self.temporal_encoder(x1)
        temporal_b = self.temporal_encoder(x2)
        features['temporal_a'] = temporal_a
        features['temporal_b'] = temporal_b

        # Stage 2
        intra_a = self.intra_brain(temporal_a)
        intra_b = self.intra_brain(temporal_b)
        features['intra_a'] = intra_a
        features['intra_b'] = intra_b

        # Stage 3
        inter_a, inter_b = self.inter_brain(intra_a, intra_b)
        features['inter_a'] = inter_a
        features['inter_b'] = inter_b

        # Stage 4
        fused = self.fusion(inter_a, inter_b)
        features['fused'] = fused

        return features

    def get_attention_weights(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Extract attention weights for interpretability.

        Args:
            x1: EEG from Person A, shape (B, 32, 1024)
            x2: EEG from Person B, shape (B, 32, 1024)

        Returns:
            Dictionary containing attention weights:
                - 'intra_attn_a': Intra-brain attention for A (if use_graph=True)
                - 'intra_attn_b': Intra-brain attention for B
                - 'cross_attn_a2b': Cross-attention A->B (if use_cross_attn=True)
                - 'cross_attn_b2a': Cross-attention B->A
        """
        attention_weights = {}

        # Stage 1
        temporal_a = self.temporal_encoder(x1)
        temporal_b = self.temporal_encoder(x2)

        # Stage 2 - Get intra-brain attention
        if self.use_graph:
            intra_a, attn_a = self.intra_brain(temporal_a, return_attn=True)
            intra_b, attn_b = self.intra_brain(temporal_b, return_attn=True)
            attention_weights['intra_attn_a'] = attn_a
            attention_weights['intra_attn_b'] = attn_b
        else:
            intra_a = self.intra_brain(temporal_a)
            intra_b = self.intra_brain(temporal_b)
            attention_weights['intra_attn_a'] = None
            attention_weights['intra_attn_b'] = None

        # Stage 3 - Get cross-attention
        if self.use_cross_attn:
            inter_a, inter_b, attn_a2b, attn_b2a = self.inter_brain(
                intra_a, intra_b, return_attn=True
            )
            attention_weights['cross_attn_a2b'] = attn_a2b
            attention_weights['cross_attn_b2a'] = attn_b2a
        else:
            attention_weights['cross_attn_a2b'] = None
            attention_weights['cross_attn_b2a'] = None

        return attention_weights


# =============================================================================
# Factory Functions for Ablation Studies
# =============================================================================

ABLATION_CONFIGS = {
    # Full model
    'full': {
        'use_sinc': True,
        'use_graph': True,
        'use_cross_attn': True,
        'use_uncertainty': True
    },
    # Pure baseline
    'baseline': {
        'use_sinc': False,
        'use_graph': False,
        'use_cross_attn': False,
        'use_uncertainty': False
    },
    # Single component ablations
    'sinc_only': {
        'use_sinc': True,
        'use_graph': False,
        'use_cross_attn': False,
        'use_uncertainty': False
    },
    'graph_only': {
        'use_sinc': False,
        'use_graph': True,
        'use_cross_attn': False,
        'use_uncertainty': False
    },
    'cross_only': {
        'use_sinc': False,
        'use_graph': False,
        'use_cross_attn': True,
        'use_uncertainty': False
    },
    'uncertainty_only': {
        'use_sinc': False,
        'use_graph': False,
        'use_cross_attn': False,
        'use_uncertainty': True
    },
    # Remove single component
    'no_sinc': {
        'use_sinc': False,
        'use_graph': True,
        'use_cross_attn': True,
        'use_uncertainty': True
    },
    'no_graph': {
        'use_sinc': True,
        'use_graph': False,
        'use_cross_attn': True,
        'use_uncertainty': True
    },
    'no_cross': {
        'use_sinc': True,
        'use_graph': True,
        'use_cross_attn': False,
        'use_uncertainty': True
    },
    'no_uncertainty': {
        'use_sinc': True,
        'use_graph': True,
        'use_cross_attn': True,
        'use_uncertainty': False
    }
}


def create_hypereeg_model(
    config_name: str = 'full',
    **kwargs
) -> HyperEEG_Encoder:
    """
    Create HyperEEG model with predefined ablation configuration.

    Args:
        config_name: One of ['full', 'baseline', 'sinc_only', 'graph_only',
                           'cross_only', 'uncertainty_only', 'no_sinc',
                           'no_graph', 'no_cross', 'no_uncertainty']
        **kwargs: Override any model parameter

    Returns:
        Configured HyperEEG_Encoder instance

    Example:
        # Full model
        model = create_hypereeg_model('full')

        # Baseline for comparison
        model = create_hypereeg_model('baseline')

        # Ablation: Full model without SincConv
        model = create_hypereeg_model('no_sinc')

        # Custom configuration
        model = create_hypereeg_model('full', embed_dim=256, num_heads=8)
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(ABLATION_CONFIGS.keys())}")

    config = ABLATION_CONFIGS[config_name].copy()
    config.update(kwargs)

    return HyperEEG_Encoder(**config)


def get_model_config(model: HyperEEG_Encoder) -> Dict[str, bool]:
    """
    Get the ablation configuration of a model.

    Args:
        model: HyperEEG_Encoder instance

    Returns:
        Dictionary with use_sinc, use_graph, use_cross_attn, use_uncertainty
    """
    return {
        'use_sinc': model.use_sinc,
        'use_graph': model.use_graph,
        'use_cross_attn': model.use_cross_attn,
        'use_uncertainty': model.use_uncertainty
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing HyperEEG Encoder")
    print("=" * 60)

    # Test configuration
    batch_size = 4
    channels = 32
    timepoints = 1024
    num_classes = 3

    # Create dummy input
    x1 = torch.randn(batch_size, channels, timepoints)
    x2 = torch.randn(batch_size, channels, timepoints)

    print(f"\nInput shapes: x1={x1.shape}, x2={x2.shape}")

    # Test all ablation configurations
    for config_name in ABLATION_CONFIGS.keys():
        print(f"\n{'─' * 40}")
        print(f"Testing config: {config_name}")
        print(f"{'─' * 40}")

        model = create_hypereeg_model(config_name)
        config = get_model_config(model)
        print(f"  M1 (SincConv): {config['use_sinc']}")
        print(f"  M2 (Graph): {config['use_graph']}")
        print(f"  M3 (CrossAttn): {config['use_cross_attn']}")
        print(f"  M4 (Uncertainty): {config['use_uncertainty']}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(x1, x2)

        print(f"  Output shape: {logits.shape}")
        assert logits.shape == (batch_size, num_classes), \
            f"Expected ({batch_size}, {num_classes}), got {logits.shape}"

        # Test feature extraction
        features = model.get_features(x1, x2)
        print(f"  Feature shapes:")
        for name, feat in features.items():
            print(f"    {name}: {feat.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
