"""
Cross-Modal Attention for Multimodal Fusion
Enables bidirectional information flow between different modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CrossModalAttention(nn.Module):
    """
    Cross-attention between two modalities

    Given modality A and modality B:
    - A' = CrossAttn(Q=A, K=B, V=B)  # A attends to B
    - B' = CrossAttn(Q=B, K=A, V=A)  # B attends to A
    """

    def __init__(
        self,
        d_model_a: int,
        d_model_b: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_projection: bool = True
    ):
        """
        Args:
            d_model_a: Dimension of modality A
            d_model_b: Dimension of modality B
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_projection: Whether to project to common space
        """
        super().__init__()

        self.d_model_a = d_model_a
        self.d_model_b = d_model_b
        self.num_heads = num_heads
        self.use_projection = use_projection

        # If dimensions differ, project to common space
        if use_projection:
            self.d_common = max(d_model_a, d_model_b)
            self.proj_a = nn.Linear(d_model_a, self.d_common)
            self.proj_b = nn.Linear(d_model_b, self.d_common)
            self.proj_out_a = nn.Linear(self.d_common, d_model_a)
            self.proj_out_b = nn.Linear(self.d_common, d_model_b)
            d_k = self.d_common // num_heads
        else:
            assert d_model_a == d_model_b, "Dimensions must match if not using projection"
            self.d_common = d_model_a
            d_k = d_model_a // num_heads

        self.d_k = d_k

        # Attention layers for A → B
        self.W_q_a = nn.Linear(self.d_common, self.d_common)
        self.W_k_b = nn.Linear(self.d_common, self.d_common)
        self.W_v_b = nn.Linear(self.d_common, self.d_common)

        # Attention layers for B → A
        self.W_q_b = nn.Linear(self.d_common, self.d_common)
        self.W_k_a = nn.Linear(self.d_common, self.d_common)
        self.W_v_a = nn.Linear(self.d_common, self.d_common)

        self.dropout = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(d_model_a)
        self.norm_b = nn.LayerNorm(d_model_b)

    def _attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention

        Args:
            Q, K, V: (B, num_heads, L, d_k)
            mask: Optional attention mask

        Returns:
            output: (B, num_heads, L, d_k)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, L, L)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (B, h, L, L)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (B, h, L, d_k)

        return output

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-modal attention

        Args:
            z_a: (B, d_model_a) or (B, L_a, d_model_a) - Modality A
            z_b: (B, d_model_b) or (B, L_b, d_model_b) - Modality B
            mask_a, mask_b: Optional masks

        Returns:
            z_a_out: (B, d_model_a) or (B, L_a, d_model_a) - Updated A
            z_b_out: (B, d_model_b) or (B, L_b, d_model_b) - Updated B
        """
        B = z_a.shape[0]

        # Handle single vector vs sequence
        is_single_a = (z_a.dim() == 2)
        is_single_b = (z_b.dim() == 2)

        if is_single_a:
            z_a = z_a.unsqueeze(1)  # (B, 1, d_model_a)
        if is_single_b:
            z_b = z_b.unsqueeze(1)  # (B, 1, d_model_b)

        L_a = z_a.shape[1]
        L_b = z_b.shape[1]

        # Project to common space if needed
        if self.use_projection:
            z_a_proj = self.proj_a(z_a)  # (B, L_a, d_common)
            z_b_proj = self.proj_b(z_b)  # (B, L_b, d_common)
        else:
            z_a_proj = z_a
            z_b_proj = z_b

        # ========== A attends to B ==========
        Q_a = self.W_q_a(z_a_proj)  # (B, L_a, d_common)
        K_b = self.W_k_b(z_b_proj)  # (B, L_b, d_common)
        V_b = self.W_v_b(z_b_proj)  # (B, L_b, d_common)

        # Reshape for multi-head attention
        Q_a = Q_a.view(B, L_a, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, L_a, d_k)
        K_b = K_b.view(B, L_b, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, L_b, d_k)
        V_b = V_b.view(B, L_b, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, L_b, d_k)

        # Compute attention
        attn_a = self._attention(Q_a, K_b, V_b, mask_b)  # (B, h, L_a, d_k)

        # Reshape back
        attn_a = attn_a.transpose(1, 2).contiguous().view(B, L_a, self.d_common)  # (B, L_a, d_common)

        # Project back to original dimension
        if self.use_projection:
            attn_a = self.proj_out_a(attn_a)  # (B, L_a, d_model_a)

        # Residual + Norm
        z_a_out = self.norm_a(z_a + self.dropout(attn_a))

        # ========== B attends to A ==========
        Q_b = self.W_q_b(z_b_proj)  # (B, L_b, d_common)
        K_a = self.W_k_a(z_a_proj)  # (B, L_a, d_common)
        V_a = self.W_v_a(z_a_proj)  # (B, L_a, d_common)

        # Reshape for multi-head attention
        Q_b = Q_b.view(B, L_b, self.num_heads, self.d_k).transpose(1, 2)
        K_a = K_a.view(B, L_a, self.num_heads, self.d_k).transpose(1, 2)
        V_a = V_a.view(B, L_a, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention
        attn_b = self._attention(Q_b, K_a, V_a, mask_a)  # (B, h, L_b, d_k)

        # Reshape back
        attn_b = attn_b.transpose(1, 2).contiguous().view(B, L_b, self.d_common)

        # Project back
        if self.use_projection:
            attn_b = self.proj_out_b(attn_b)  # (B, L_b, d_model_b)

        # Residual + Norm
        z_b_out = self.norm_b(z_b + self.dropout(attn_b))

        # Remove singleton dimension if input was single vector
        if is_single_a:
            z_a_out = z_a_out.squeeze(1)
        if is_single_b:
            z_b_out = z_b_out.squeeze(1)

        return z_a_out, z_b_out


class CoAttention(nn.Module):
    """
    Co-Attention: Joint attention over two modalities
    Computes attention map conditioned on both modalities
    """

    def __init__(
        self,
        d_model_a: int,
        d_model_b: int,
        d_hidden: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model_a = d_model_a
        self.d_model_b = d_model_b
        self.d_hidden = d_hidden
        self.num_heads = num_heads

        # Project both modalities to common hidden space
        self.proj_a = nn.Linear(d_model_a, d_hidden)
        self.proj_b = nn.Linear(d_model_b, d_hidden)

        # Co-attention mechanism
        self.W_co = nn.Parameter(torch.randn(d_hidden, d_hidden))

        # Output projections
        self.proj_out_a = nn.Linear(d_hidden, d_model_a)
        self.proj_out_b = nn.Linear(d_hidden, d_model_b)

        self.dropout = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(d_model_a)
        self.norm_b = nn.LayerNorm(d_model_b)

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Co-attention

        Args:
            z_a: (B, L_a, d_model_a) or (B, d_model_a)
            z_b: (B, L_b, d_model_b) or (B, d_model_b)

        Returns:
            z_a_out, z_b_out: Updated representations
        """
        # Handle single vectors
        is_single_a = (z_a.dim() == 2)
        is_single_b = (z_b.dim() == 2)

        if is_single_a:
            z_a = z_a.unsqueeze(1)
        if is_single_b:
            z_b = z_b.unsqueeze(1)

        B, L_a, _ = z_a.shape
        _, L_b, _ = z_b.shape

        # Project to common space
        h_a = self.proj_a(z_a)  # (B, L_a, d_hidden)
        h_b = self.proj_b(z_b)  # (B, L_b, d_hidden)

        # Compute co-attention matrix
        # C = tanh(H_a^T W H_b)
        h_a_W = torch.matmul(h_a, self.W_co)  # (B, L_a, d_hidden)
        C = torch.matmul(h_a_W, h_b.transpose(1, 2))  # (B, L_a, L_b)
        C = torch.tanh(C)

        # Attention weights
        attn_a = F.softmax(C, dim=-1)  # (B, L_a, L_b) - A attends to B
        attn_b = F.softmax(C.transpose(1, 2), dim=-1)  # (B, L_b, L_a) - B attends to A

        # Apply attention
        h_a_co = torch.matmul(attn_a, h_b)  # (B, L_a, d_hidden)
        h_b_co = torch.matmul(attn_b, h_a)  # (B, L_b, d_hidden)

        # Project back
        z_a_co = self.proj_out_a(h_a_co)  # (B, L_a, d_model_a)
        z_b_co = self.proj_out_b(h_b_co)  # (B, L_b, d_model_b)

        # Residual + Norm
        z_a_out = self.norm_a(z_a + self.dropout(z_a_co))
        z_b_out = self.norm_b(z_b + self.dropout(z_b_co))

        # Remove singleton dimension
        if is_single_a:
            z_a_out = z_a_out.squeeze(1)
        if is_single_b:
            z_b_out = z_b_out.squeeze(1)

        return z_a_out, z_b_out


class GatedCrossModalFusion(nn.Module):
    """
    Gated cross-modal fusion
    Uses one modality to gate information from the other
    """

    def __init__(self, d_model_a: int, d_model_b: int):
        super().__init__()

        self.d_model_a = d_model_a
        self.d_model_b = d_model_b

        # Gate generation (B modality gates A)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model_b, d_model_a),
            nn.Sigmoid()
        )

        # Transform A
        self.transform_a = nn.Linear(d_model_a, d_model_a)

        self.norm = nn.LayerNorm(d_model_a)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Gate A using B

        Args:
            z_a: (B, d_model_a) - Primary modality
            z_b: (B, d_model_b) - Gating modality

        Returns:
            z_out: (B, d_model_a) - Gated output
        """
        # Generate gate from B
        gate = self.gate_net(z_b)  # (B, d_model_a)

        # Transform A
        z_a_transformed = self.transform_a(z_a)  # (B, d_model_a)

        # Apply gate
        z_gated = gate * z_a_transformed  # (B, d_model_a)

        # Residual
        z_out = self.norm(z_a + z_gated)

        return z_out


class MultiModalTransformerBlock(nn.Module):
    """
    Transformer block with cross-modal attention
    Alternates self-attention and cross-modal attention
    """

    def __init__(
        self,
        d_model_a: int,
        d_model_b: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention for each modality
        self.self_attn_a = nn.MultiheadAttention(d_model_a, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(d_model_b, num_heads, dropout=dropout, batch_first=True)

        # Cross-modal attention
        self.cross_attn = CrossModalAttention(d_model_a, d_model_b, num_heads, dropout, use_projection=True)

        # Feed-forward networks
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model_a, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model_a),
            nn.Dropout(dropout)
        )

        self.ffn_b = nn.Sequential(
            nn.Linear(d_model_b, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model_b),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1_a = nn.LayerNorm(d_model_a)
        self.norm2_a = nn.LayerNorm(d_model_a)
        self.norm3_a = nn.LayerNorm(d_model_a)

        self.norm1_b = nn.LayerNorm(d_model_b)
        self.norm2_b = nn.LayerNorm(d_model_b)
        self.norm3_b = nn.LayerNorm(d_model_b)

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multimodal transformer block

        Args:
            z_a: (B, L_a, d_model_a)
            z_b: (B, L_b, d_model_b)

        Returns:
            z_a_out, z_b_out: Updated representations
        """
        # Self-attention for A
        attn_a, _ = self.self_attn_a(z_a, z_a, z_a)
        z_a = self.norm1_a(z_a + attn_a)

        # Self-attention for B
        attn_b, _ = self.self_attn_b(z_b, z_b, z_b)
        z_b = self.norm1_b(z_b + attn_b)

        # Cross-modal attention
        z_a_cross, z_b_cross = self.cross_attn(z_a, z_b)
        z_a = self.norm2_a(z_a_cross)
        z_b = self.norm2_b(z_b_cross)

        # Feed-forward
        z_a = self.norm3_a(z_a + self.ffn_a(z_a))
        z_b = self.norm3_b(z_b + self.ffn_b(z_b))

        return z_a, z_b
