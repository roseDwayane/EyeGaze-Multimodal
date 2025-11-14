"""
Symmetric Fusion Operators for Multimodal Fusion
Ensures permutation invariance: f(z1, z2) = f(z2, z1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SymmetricFusionOperators(nn.Module):
    """
    Symmetric fusion of two representations using multiple operators

    Four symmetric operations:
    1. Sum: z1 + z2
    2. Product (Hadamard): z1 * z2
    3. Absolute Difference: |z1 - z2|
    4. Concatenation: [z1, z2] (becomes symmetric after projection)

    These operations preserve permutation invariance when combined properly.
    """

    def __init__(self, d_model: int, output_dim: int = None, mode: str = 'all'):
        """
        Args:
            d_model: Input feature dimension
            output_dim: Output dimension (default: same as d_model)
            mode: Fusion mode
                - 'all': Use all 4 operators (sum, mul, diff, concat) → 4*d
                - 'basic': Use 3 operators (sum, mul, diff) → 3*d
                - 'simple': Use only sum and mul → 2*d
        """
        super().__init__()

        self.d_model = d_model
        self.mode = mode
        self.output_dim = output_dim or d_model

        # Determine feature dimension based on mode
        if mode == 'all':
            feature_dim = d_model * 4  # [sum, mul, diff, concat(z1,z2)]
        elif mode == 'basic':
            feature_dim = d_model * 3  # [sum, mul, diff]
        elif mode == 'simple':
            feature_dim = d_model * 2  # [sum, mul]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Projection layer
        self.proj = nn.Linear(feature_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric fusion of two representations

        Args:
            z1, z2: (B, d_model) - Two representations to fuse

        Returns:
            fused: (B, output_dim) - Fused representation
        """
        # Symmetric operations
        z_sum = z1 + z2  # (B, d)
        z_mul = z1 * z2  # (B, d) - Hadamard product
        z_diff = torch.abs(z1 - z2)  # (B, d) - Absolute difference

        if self.mode == 'all':
            # Also include concatenation (not inherently symmetric but useful)
            z_concat = torch.cat([z1, z2], dim=-1)  # (B, 2*d)
            features = torch.cat([z_sum, z_mul, z_diff, z_concat], dim=-1)  # (B, 4*d)
        elif self.mode == 'basic':
            features = torch.cat([z_sum, z_mul, z_diff], dim=-1)  # (B, 3*d)
        elif self.mode == 'simple':
            features = torch.cat([z_sum, z_mul], dim=-1)  # (B, 2*d)

        # Project to output dimension
        fused = self.proj(features)  # (B, output_dim)
        fused = self.norm(fused)

        return fused


class SymmetricFusionWithGating(nn.Module):
    """
    Symmetric fusion with learnable gating mechanism
    Allows the model to adaptively weight different fusion operations
    """

    def __init__(self, d_model: int, output_dim: int = None):
        super().__init__()

        self.d_model = d_model
        self.output_dim = output_dim or d_model

        # Four parallel projection branches
        self.proj_sum = nn.Linear(d_model, self.output_dim)
        self.proj_mul = nn.Linear(d_model, self.output_dim)
        self.proj_diff = nn.Linear(d_model, self.output_dim)
        self.proj_concat = nn.Linear(d_model * 2, self.output_dim)

        # Gating network (learns to weight each operation)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )

        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Gated symmetric fusion

        Args:
            z1, z2: (B, d_model)

        Returns:
            fused: (B, output_dim)
        """
        B = z1.shape[0]

        # Compute symmetric operations
        z_sum = z1 + z2
        z_mul = z1 * z2
        z_diff = torch.abs(z1 - z2)
        z_concat = torch.cat([z1, z2], dim=-1)

        # Project each operation
        f_sum = self.proj_sum(z_sum)  # (B, output_dim)
        f_mul = self.proj_mul(z_mul)
        f_diff = self.proj_diff(z_diff)
        f_concat = self.proj_concat(z_concat)

        # Stack features
        features = torch.stack([f_sum, f_mul, f_diff, f_concat], dim=1)  # (B, 4, output_dim)

        # Compute gates
        gate_input = torch.cat([z1, z2], dim=-1)  # (B, 2*d)
        gates = self.gate(gate_input)  # (B, 4)
        gates = gates.unsqueeze(-1)  # (B, 4, 1)

        # Weighted combination
        fused = (features * gates).sum(dim=1)  # (B, output_dim)
        fused = self.norm(fused)

        return fused


class PairwiseInteraction(nn.Module):
    """
    Pairwise interaction module for dual representations
    Computes interaction features between two inputs
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        # Bilinear interaction
        self.W_bilinear = nn.Parameter(torch.randn(d_model, d_model))

        # MLP for processing concatenated features
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise interaction

        Args:
            z1, z2: (B, d_model)

        Returns:
            interaction: (B, d_model)
        """
        # Bilinear interaction: z1^T W z2
        bilinear = torch.matmul(z1, self.W_bilinear)  # (B, d)
        bilinear = (bilinear * z2).sum(dim=-1, keepdim=True)  # (B, 1)

        # MLP interaction
        concat = torch.cat([z1, z2], dim=-1)  # (B, 2*d)
        mlp_out = self.mlp(concat)  # (B, d)

        # Combine
        interaction = mlp_out * bilinear  # Gated by bilinear score

        return interaction


class MultiScaleFusion(nn.Module):
    """
    Multi-scale symmetric fusion
    Fuses representations at multiple granularities
    """

    def __init__(self, d_model: int, num_scales: int = 3):
        super().__init__()

        self.d_model = d_model
        self.num_scales = num_scales

        # Different scale projections
        self.scale_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales)
        ])

        # Symmetric fusion at each scale
        self.scale_fusions = nn.ModuleList([
            SymmetricFusionOperators(d_model, d_model, mode='basic')
            for _ in range(num_scales)
        ])

        # Final aggregation
        self.aggregation = nn.Linear(d_model * num_scales, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale fusion

        Args:
            z1, z2: (B, d_model)

        Returns:
            fused: (B, d_model)
        """
        scale_outputs = []

        for scale_idx in range(self.num_scales):
            # Project to scale-specific space
            z1_scale = self.scale_projs[scale_idx](z1)
            z2_scale = self.scale_projs[scale_idx](z2)

            # Symmetric fusion at this scale
            fused_scale = self.scale_fusions[scale_idx](z1_scale, z2_scale)
            scale_outputs.append(fused_scale)

        # Concatenate all scales
        multi_scale = torch.cat(scale_outputs, dim=-1)  # (B, num_scales * d)

        # Aggregate
        fused = self.aggregation(multi_scale)  # (B, d)
        fused = self.norm(fused)

        return fused


# ========== Helper Functions ==========

def symmetric_cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric cosine similarity

    Args:
        z1, z2: (B, d)

    Returns:
        similarity: (B,)
    """
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    similarity = (z1_norm * z2_norm).sum(dim=-1)
    return similarity


def symmetric_l2_distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric L2 distance

    Args:
        z1, z2: (B, d)

    Returns:
        distance: (B,)
    """
    distance = torch.norm(z1 - z2, p=2, dim=-1)
    return distance
