"""
Dual EEG Transformer for Inter-Brain Synchrony Classification
Fuses two players' EEG signals with IBS (Inter-Brain Synchrony) token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from .art import (
    TransformerEncoder,
    TransformerEncoderBlock,
    MultiHeadAttention,
    PositionalEmbedding
)


class TemporalConvFrontend(nn.Module):
    """
    Temporal convolution frontend to downsample and embed EEG channels
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int = 25,
        stride: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        self.convs = nn.ModuleList()

        # First conv layer
        self.convs.append(nn.Conv1d(in_channels, d_model, kernel_size, stride, padding=kernel_size//2))

        # Additional conv layers
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv1d(d_model, d_model, kernel_size, stride, padding=kernel_size//2))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - EEG signal
        Returns:
            (B, T̃, d) - Downsampled and embedded
        """
        for conv in self.convs:
            x = self.dropout(self.relu(conv(x)))

        # Permute to (B, T, d)
        x = x.permute(0, 2, 1)
        return x


class IBSTokenGenerator(nn.Module):
    """
    Generates Inter-Brain Synchrony (IBS) token from dual EEG signals
    Computes cross-brain connectivity features (PLV, power correlation, etc.)
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_freq_bands: int = 4,  # theta, alpha, beta, gamma
    ):
        super().__init__()
        self.num_freq_bands = num_freq_bands

        # Feature dimension: num_freq_bands * (PLV + power_corr + phase_diff)
        feature_dim = num_freq_bands * 3

        self.proj = nn.Linear(feature_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def compute_plv(self, phase1: torch.Tensor, phase2: torch.Tensor) -> torch.Tensor:
        """
        Compute Phase Locking Value between two signals
        Args:
            phase1, phase2: (B, C, T) - phase of signals
        Returns:
            PLV: (B,) - scalar PLV value per batch
        """
        phase_diff = phase1 - phase2
        plv = torch.abs(torch.mean(torch.exp(1j * phase_diff), dim=(1, 2)))
        return plv.real

    def compute_power_correlation(self, power1: torch.Tensor, power2: torch.Tensor) -> torch.Tensor:
        """
        Compute power correlation between two signals
        Args:
            power1, power2: (B, C, T)
        Returns:
            corr: (B,) - correlation coefficient per batch
        """
        p1_flat = power1.flatten(1)  # (B, C*T)
        p2_flat = power2.flatten(1)

        # Normalize
        p1_norm = (p1_flat - p1_flat.mean(dim=1, keepdim=True)) / (p1_flat.std(dim=1, keepdim=True) + 1e-8)
        p2_norm = (p2_flat - p2_flat.mean(dim=1, keepdim=True)) / (p2_flat.std(dim=1, keepdim=True) + 1e-8)

        # Compute correlation
        corr = (p1_norm * p2_norm).mean(dim=1)
        return corr

    def forward(
        self,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate IBS token from dual EEG signals
        Args:
            eeg1, eeg2: (B, C, T) - EEG signals
        Returns:
            ibs_token: (B, d_model) - IBS token
        """
        B = eeg1.shape[0]
        features = []

        # For simplicity, use power and phase approximations
        # In practice, use proper spectral analysis (FFT, wavelets, etc.)

        for freq_idx in range(self.num_freq_bands):
            # Simple approximation: use different frequency ranges
            # In real implementation, use proper filtering

            # Compute power (amplitude squared)
            power1 = eeg1 ** 2
            power2 = eeg2 ** 2

            # Compute phase (simplified using Hilbert-like transform)
            # In practice: use scipy.signal.hilbert or torch FFT
            phase1 = torch.angle(torch.fft.rfft(eeg1, dim=2))
            phase2 = torch.angle(torch.fft.rfft(eeg2, dim=2))

            # Match dimensions
            min_len = min(power1.shape[2], phase1.shape[2])
            power1, power2 = power1[:, :, :min_len], power2[:, :, :min_len]
            phase1, phase2 = phase1[:, :, :min_len], phase2[:, :, :min_len]

            # Compute features
            plv = self.compute_plv(phase1, phase2)
            power_corr = self.compute_power_correlation(power1, power2)
            phase_diff = torch.abs(torch.mean(phase1 - phase2, dim=(1, 2)))

            features.extend([plv, power_corr, phase_diff])

        # Stack features: (B, feature_dim)
        features = torch.stack(features, dim=1)

        # Project to d_model
        ibs_token = self.norm(self.proj(features))  # (B, d_model)

        return ibs_token


class SymmetricFusion(nn.Module):
    """
    Symmetric operator for fusing two representations
    Ensures permutation invariance: f(z1, z2) = f(z2, z1)
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Operations: element-wise sum, product, abs difference
        # Total: d_model * 3
        self.proj = nn.Linear(d_model * 3, d_model)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, d_model)
        Returns:
            fused: (B, d_model)
        """
        # Symmetric operations
        add = z1 + z2
        mul = z1 * z2
        abs_diff = torch.abs(z1 - z2)

        # Concat all symmetric operations and project
        combined = torch.cat([add, mul, abs_diff], dim=-1)
        fused = self.proj(combined)

        return fused


class CrossBrainAttention(nn.Module):
    """
    Cross-attention between two brain sequences
    Z1' = CrossAttn(Z1, Z2), Z2' = CrossAttn(Z2, Z1)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z1, z2: (B, T, d_model) - encoded sequences
        Returns:
            z1', z2': (B, T, d_model) - cross-attended sequences
        """
        # Z1 attends to Z2
        z1_cross = self.cross_attn(q=z1, k=z2, v=z2)
        z1_out = self.norm(z1 + self.dropout(z1_cross))

        # Z2 attends to Z1
        z2_cross = self.cross_attn(q=z2, k=z1, v=z1)
        z2_out = self.norm(z2 + self.dropout(z2_cross))

        return z1_out, z2_out


class DualEEGTransformer(nn.Module):
    """
    Dual-stream EEG Transformer with Inter-Brain Synchrony token

    Architecture:
    1. Temporal convolution frontend for each player
    2. IBS token generation from cross-brain features
    3. Token sequence: [CLS, IBS, H1(1), ..., H1(T̃)]
    4. Shared Transformer Encoder (Siamese)
    5. Cross-brain attention
    6. Symmetric fusion and classification
    """
    def __init__(
        self,
        in_channels: int = 62,  # Number of EEG channels
        num_classes: int = 3,    # Single, Competition, Cooperation
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
        conv_kernel_size: int = 25,
        conv_stride: int = 4,
        conv_layers: int = 2,
    ):
        super().__init__()

        self.d_model = d_model

        # Temporal convolution frontend
        self.temporal_conv = TemporalConvFrontend(
            in_channels, d_model, conv_kernel_size, conv_stride, conv_layers
        )

        # IBS token generator
        self.ibs_generator = IBSTokenGenerator(in_channels, d_model)

        # CLS tokens (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding
        self.pos_embed = PositionalEmbedding(max_len, d_model, mode='learned')

        # Shared Transformer Encoder (Siamese)
        self.encoder = TransformerEncoder(
            d_model, num_layers, num_heads, d_ff, dropout, dropout
        )

        # Cross-brain attention
        self.cross_attn = CrossBrainAttention(d_model, num_heads, dropout)

        # Symmetric fusion
        self.symmetric_fusion = SymmetricFusion(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # [f_pair, mp1', mp2']
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            eeg1, eeg2: (B, C, T) - EEG signals for player 1 and 2
            labels: (B,) - class labels (optional, for computing loss)

        Returns:
            dict with 'logits', 'loss' (if labels provided)
        """
        B = eeg1.shape[0]

        # 1. Temporal convolution frontend
        h1 = self.temporal_conv(eeg1)  # (B, T̃, d)
        h2 = self.temporal_conv(eeg2)  # (B, T̃, d)

        # 2. Generate IBS token
        ibs_token = self.ibs_generator(eeg1, eeg2)  # (B, d)
        ibs_token = ibs_token.unsqueeze(1)  # (B, 1, d)

        # 3. Build sequences: [CLS, IBS, H(1), ..., H(T̃)]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)

        seq1 = torch.cat([cls, ibs_token, h1], dim=1)  # (B, T̃+2, d)
        seq2 = torch.cat([cls, ibs_token, h2], dim=1)  # (B, T̃+2, d)

        # Add positional embedding
        seq1 = self.pos_embed(seq1)
        seq2 = self.pos_embed(seq2)

        # 4. Shared Transformer Encoder (Siamese)
        z1 = self.encoder(seq1)  # (B, T̃+2, d)
        z2 = self.encoder(seq2)  # (B, T̃+2, d)

        # 5. Extract CLS tokens
        cls1 = z1[:, 0, :]  # (B, d)
        cls2 = z2[:, 0, :]  # (B, d)

        # 6. Cross-brain attention (skip CLS and IBS for mean pooling)
        z1_cross, z2_cross = self.cross_attn(z1[:, 2:, :], z2[:, 2:, :])

        # Mean pooling (excluding CLS and IBS)
        mp1 = z1_cross.mean(dim=1)  # (B, d)
        mp2 = z2_cross.mean(dim=1)  # (B, d)

        # 7. Symmetric fusion
        f_pair = self.symmetric_fusion(cls1, cls2)  # (B, d)

        # 8. Concatenate and classify
        z_fuse = torch.cat([f_pair, mp1, mp2], dim=-1)  # (B, 3*d)
        logits = self.classifier(z_fuse)  # (B, num_classes)

        # Compute loss if labels provided
        output = {
            'logits': logits,
            'cls1': cls1,
            'cls2': cls2,
            'ibs_token': ibs_token.squeeze(1)  # (B, d) - remove the unsqueezed dimension
        }

        if labels is not None:
            loss_ce = F.cross_entropy(logits, labels)

            # Optional: Symmetry loss L_sym = ||cls1 - cls2||^2
            # Encourages similar representations when appropriate
            # Can be weighted and disabled initially

            output['loss'] = loss_ce
            output['loss_ce'] = loss_ce

        return output

    def compute_symmetry_loss(self, cls1: torch.Tensor, cls2: torch.Tensor) -> torch.Tensor:
        """
        Symmetry loss: encourages similar CLS representations
        Only use when task requires symmetry (e.g., cooperation)
        """
        return F.mse_loss(cls1, cls2)

    def compute_ibs_alignment_loss(
        self,
        ibs_token: torch.Tensor,
        cls1: torch.Tensor,
        cls2: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        IBS alignment loss using InfoNCE
        Encourages IBS token to align with CLS tokens from the same window

        Args:
            ibs_token: (B, d) - IBS tokens
            cls1, cls2: (B, d) - CLS tokens
            temperature: softmax temperature

        Returns:
            loss: scalar
        """
        B = ibs_token.shape[0]

        # Normalize
        ibs_norm = F.normalize(ibs_token, dim=-1)
        cls1_norm = F.normalize(cls1, dim=-1)
        cls2_norm = F.normalize(cls2, dim=-1)

        # Positive pairs: IBS with corresponding CLS1 and CLS2
        pos_sim1 = (ibs_norm * cls1_norm).sum(dim=-1) / temperature  # (B,)
        pos_sim2 = (ibs_norm * cls2_norm).sum(dim=-1) / temperature  # (B,)

        # Negative pairs: IBS with all other CLS tokens in the batch
        all_cls = torch.cat([cls1_norm, cls2_norm], dim=0)  # (2B, d)
        neg_sim = torch.matmul(ibs_norm, all_cls.T) / temperature  # (B, 2B)

        # InfoNCE loss
        # For each IBS, positive examples are cls1 and cls2 from same sample
        # Negative examples are all other cls tokens in batch

        # Simplified: treat cls1 as positive, all others as negative
        labels = torch.arange(B, device=ibs_token.device)
        loss = F.cross_entropy(neg_sim, labels)

        return loss
