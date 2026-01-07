# Implementation Plan: HyperEEG Encoder

## Architecture Overview (Based on fig3.EEGArch.png)

The HyperEEG Encoder is a dual-stream architecture for hyperscanning EEG classification with 4 progressive stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Dual-Stream Input                                  │
│                                                                             │
│    Player A Raw EEG (B, 32, 1024)      Player B Raw EEG (B, 32, 1024)       │
│              │                                    │                          │
│              ▼                                    ▼                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 1: Learnable Temporal (M1 - SincConv)                                │
│  ┌──────────────────────┐              ┌──────────────────────┐             │
│  │ SincConv1d           │   Shared     │ SincConv1d           │             │
│  │ (Learnable Band-pass)│◄──Weights───►│ (Learnable Band-pass)│             │
│  └──────────────────────┘              └──────────────────────┘             │
│       (B, 32, 1024) → (B, 32, 128)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 2: Intra-Brain Topology (M2 - Graph)                                 │
│  ┌──────────────────────┐              ┌──────────────────────┐             │
│  │ Self-Attention GNN   │   Shared     │ Self-Attention GNN   │             │
│  │ (Intra-Brain Connect)│◄──Weights───►│ (Intra-Brain Connect)│             │
│  └──────────────────────┘              └──────────────────────┘             │
│       (B, 32, 128) → (B, 32, 128)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 3: Inter-Brain Cross Attention (M3 - CrossAttn)                      │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │         Cross-Attention (Q_A @ K_B, V_B)                  │              │
│  │         Cross-Attention (Q_B @ K_A, V_A)                  │              │
│  └───────────────────────────────────────────────────────────┘              │
│       (B, 32, 128) × 2 → (B, 32, 128) × 2                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 4: Uncertainty Estimation & Fusion (M4 - Uncertainty)                │
│  ┌────────────────┐                    ┌────────────────┐                   │
│  │ Graph Pooling  │                    │ Graph Pooling  │                   │
│  │ + Var Estimate │                    │ + Var Estimate │                   │
│  └───────┬────────┘                    └───────┬────────┘                   │
│          │ Z_A, σ²_A                           │ Z_B, σ²_B                  │
│          └──────────────┬──────────────────────┘                            │
│                         ▼                                                   │
│            Weighted Fusion: (Z_A·σ²_B + Z_B·σ²_A) / (σ²_A + σ²_B)          │
│                         │                                                   │
│                         ▼ (B, 128)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Classifier Head                                                            │
│  ┌───────────────────────────────────┐                                      │
│  │ MLP + Softmax → (B, 3)            │                                      │
│  │ {Single, Competition, Cooperation}│                                      │
│  └───────────────────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. SincConv1d (M1 - Learnable Temporal Filter)

**Purpose:** Learnable band-pass filter based on sinc function (Ravanelli & Bengio, 2018)

**Mathematical Formulation:**
```
h[n] = 2f_high · sinc(2πf_high · n) - 2f_low · sinc(2πf_low · n)
sinc(x) = sin(x) / x  (with sinc(0) = 1)

Windowed: h_windowed[n] = h[n] · w[n]  (Hamming window)
```

**Parameters:**
- `in_channels`: 32 (EEG channels)
- `out_channels`: 32 (preserve channel structure)
- `kernel_size`: 251 (covers ~0.5s at 500Hz sampling)
- `sample_rate`: 500 Hz (typical EEG)
- `min_low_hz`: 1 Hz
- `min_band_hz`: 4 Hz

**Learnable Parameters:**
- `low_hz_`: Low cutoff frequencies (shape: out_channels)
- `band_hz_`: Bandwidth (shape: out_channels)

**Implementation:**
```python
class SincConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 sample_rate=500, min_low_hz=1, min_band_hz=4):
        # Initialize learnable frequency parameters
        # low_hz = min_low_hz + softplus(low_hz_)
        # high_hz = low_hz + min_band_hz + softplus(band_hz_)

    def _get_sinc_filters(self):
        # Construct sinc filters from learnable parameters
        # Apply Hamming window
        # Return: (out_channels, 1, kernel_size)

    def forward(self, x):
        # x: (B, C, T) → (B, C, T')
        # Apply depthwise conv with sinc filters
```

**Fallback (use_sinc=False):**
```python
nn.Sequential(
    nn.Conv1d(in_channels, out_channels, kernel_size=251, padding=125, groups=in_channels),
    nn.BatchNorm1d(out_channels),
    nn.GELU()
)
```

---

### 2. IntraGraphBlock (M2 - Intra-Brain Topology)

**Purpose:** Model spatial connectivity between EEG channels using self-attention as a learnable adjacency matrix.

**Concept:**
- Treat each channel as a graph node
- Use self-attention to learn connectivity weights
- Aggregate information across channels

**Architecture:**
```
Input: (B, C, D) where C=channels (nodes), D=features (embed_dim)

Q, K, V = Linear(D → D) applied to each node
Attention = softmax(Q @ K.T / sqrt(D))  # (B, C, C) - Adjacency matrix
Output = Attention @ V  # (B, C, D)
```

**Parameters:**
- `embed_dim`: 128
- `num_heads`: 4
- `dropout`: 0.1

**Implementation:**
```python
class IntraGraphBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, D)
        # Self-attention among channels (treating channels as sequence)
        # Return: (B, C, D)
```

**Fallback (use_graph=False):**
```python
# Flatten + Linear projection
x = x.flatten(1)  # (B, C*D)
x = Linear(C*D, C*D)(x)
x = x.view(B, C, D)  # (B, C, D)
```

---

### 3. InterBrainCrossAttn (M3 - Inter-Brain Interaction)

**Purpose:** Model inter-brain synchrony using cross-attention mechanism.

**Concept:**
```
# Person A attends to Person B
Q_A, K_B, V_B
cross_A = softmax(Q_A @ K_B.T / sqrt(D)) @ V_B

# Person B attends to Person A
Q_B, K_A, V_A
cross_B = softmax(Q_B @ K_A.T / sqrt(D)) @ V_A
```

**Architecture:**
```
Input: feat_A (B, C, D), feat_B (B, C, D)

# Bidirectional cross-attention
Q_A = Linear(feat_A)
K_B, V_B = Linear(feat_B), Linear(feat_B)
cross_A = Attention(Q_A, K_B, V_B)

Q_B = Linear(feat_B)
K_A, V_A = Linear(feat_A), Linear(feat_A)
cross_B = Attention(Q_B, K_A, V_A)

# Residual connection
out_A = feat_A + cross_A
out_B = feat_B + cross_B
```

**Parameters:**
- `embed_dim`: 128
- `num_heads`: 4
- `dropout`: 0.1

**Implementation:**
```python
class InterBrainCrossAttn(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        self.cross_attn_a2b = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.cross_attn_b2a = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

    def forward(self, feat_a, feat_b):
        # feat_a: (B, C, D), feat_b: (B, C, D)
        # Cross-attention: A queries B, B queries A
        # Return: (B, C, D), (B, C, D)
```

**Fallback (use_cross_attn=False):**
```python
# Early fusion - concatenate features
combined = torch.cat([feat_a, feat_b], dim=1)  # (B, 2C, D)
# Or keep separate for later fusion
return feat_a, feat_b  # No interaction
```

---

### 4. UncertaintyFusion (M4 - Uncertainty-Aware Fusion)

**Purpose:** Fuse features from two brains with uncertainty-weighted averaging.

**Mathematical Formulation:**
```
# Estimate mean and variance for each stream
μ_A, σ²_A = MLP(pooled_A)  # Mean and variance
μ_B, σ²_B = MLP(pooled_B)

# Inverse variance weighting
w_A = σ²_B / (σ²_A + σ²_B + ε)
w_B = σ²_A / (σ²_A + σ²_B + ε)

# Weighted fusion
Z = w_A · μ_A + w_B · μ_B
```

**Architecture:**
```
Input: feat_A (B, C, D), feat_B (B, C, D)

# Graph pooling (mean + max)
pooled_A = mean(feat_A, dim=1) + max(feat_A, dim=1)  # (B, D)
pooled_B = mean(feat_B, dim=1) + max(feat_B, dim=1)  # (B, D)

# Variance estimation
mean_A, var_A = VarianceEstimator(pooled_A)  # (B, D), (B, D)
mean_B, var_B = VarianceEstimator(pooled_B)

# Uncertainty-weighted fusion
fused = (mean_A * var_B + mean_B * var_A) / (var_A + var_B + ε)
```

**Implementation:**
```python
class UncertaintyFusion(nn.Module):
    def __init__(self, embed_dim, output_dim):
        self.mean_head = nn.Linear(embed_dim, output_dim)
        self.var_head = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.Softplus()  # Ensure positive variance
        )

    def forward(self, feat_a, feat_b):
        # Pooling
        pooled_a = feat_a.mean(dim=1) + feat_a.max(dim=1)[0]  # (B, D)
        pooled_b = feat_b.mean(dim=1) + feat_b.max(dim=1)[0]  # (B, D)

        # Estimate mean and variance
        mean_a, var_a = self.mean_head(pooled_a), self.var_head(pooled_a)
        mean_b, var_b = self.mean_head(pooled_b), self.var_head(pooled_b)

        # Weighted fusion
        fused = (mean_a * var_b + mean_b * var_a) / (var_a + var_b + 1e-8)
        return fused  # (B, output_dim)
```

**Fallback (use_uncertainty=False):**
```python
# Simple average pooling + concatenation
pooled_a = feat_a.mean(dim=1)  # (B, D)
pooled_b = feat_b.mean(dim=1)  # (B, D)
fused = (pooled_a + pooled_b) / 2  # (B, D)
# Or: fused = torch.cat([pooled_a, pooled_b], dim=-1)  # (B, 2D)
```

---

## Main Model: HyperEEG_Encoder

### Constructor Parameters

```python
class HyperEEG_Encoder(nn.Module):
    def __init__(
        self,
        # Input configuration
        in_channels: int = 32,          # EEG channels
        in_timepoints: int = 1024,      # Timepoints per segment
        num_classes: int = 3,           # Output classes

        # Architecture configuration
        embed_dim: int = 128,           # Feature dimension
        num_heads: int = 4,             # Attention heads
        dropout: float = 0.1,           # Dropout rate

        # Ablation switches (M1-M4)
        use_sinc: bool = True,          # M1: SincConv vs Conv1d
        use_graph: bool = True,         # M2: Graph attention vs Linear
        use_cross_attn: bool = True,    # M3: Cross-attention vs Concat
        use_uncertainty: bool = True,   # M4: Uncertainty fusion vs Average

        # SincConv parameters
        sample_rate: int = 500,         # EEG sampling rate
        sinc_kernel_size: int = 251,    # Sinc filter length
    ):
```

### Forward Flow with Shape Comments

```python
def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
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

    # ============================================================
    # Stage 2: Intra-Brain Topology (M2)
    # ============================================================
    # x1: (B, 32, 128) -> (B, 32, 128)
    # x2: (B, 32, 128) -> (B, 32, 128)
    x1 = self.intra_brain(x1)
    x2 = self.intra_brain(x2)  # Shared weights

    # ============================================================
    # Stage 3: Inter-Brain Cross Attention (M3)
    # ============================================================
    # x1, x2: (B, 32, 128) × 2 -> (B, 32, 128) × 2
    x1, x2 = self.inter_brain(x1, x2)

    # ============================================================
    # Stage 4: Uncertainty Estimation & Fusion (M4)
    # ============================================================
    # x1, x2: (B, 32, 128) × 2 -> fused: (B, 128)
    fused = self.fusion(x1, x2)

    # ============================================================
    # Classifier Head
    # ============================================================
    # fused: (B, 128) -> logits: (B, 3)
    logits = self.classifier(fused)

    return logits
```

---

## Ablation Study Configuration

### Configuration Presets

| Config Name | M1 (Sinc) | M2 (Graph) | M3 (CrossAttn) | M4 (Uncertainty) | Description |
|-------------|-----------|------------|----------------|------------------|-------------|
| `baseline` | ❌ | ❌ | ❌ | ❌ | Pure baseline: Conv1d + Linear + Concat + Average |
| `sinc_only` | ✅ | ❌ | ❌ | ❌ | Test SincConv contribution |
| `graph_only` | ❌ | ✅ | ❌ | ❌ | Test Graph attention contribution |
| `cross_only` | ❌ | ❌ | ✅ | ❌ | Test Cross-attention contribution |
| `uncert_only` | ❌ | ❌ | ❌ | ✅ | Test Uncertainty fusion contribution |
| `no_sinc` | ❌ | ✅ | ✅ | ✅ | Full model without SincConv |
| `no_graph` | ✅ | ❌ | ✅ | ✅ | Full model without Graph |
| `no_cross` | ✅ | ✅ | ❌ | ✅ | Full model without Cross-attention |
| `no_uncert` | ✅ | ✅ | ✅ | ❌ | Full model without Uncertainty |
| `full` | ✅ | ✅ | ✅ | ✅ | Complete HyperEEG model |

### Factory Function

```python
def create_hypereeg_model(config_name: str = 'full', **kwargs) -> HyperEEG_Encoder:
    """
    Create HyperEEG model with predefined ablation configurations.

    Args:
        config_name: One of ['baseline', 'sinc_only', ..., 'full']
        **kwargs: Override any parameter

    Returns:
        Configured HyperEEG_Encoder instance
    """
    configs = {
        'baseline': dict(use_sinc=False, use_graph=False, use_cross_attn=False, use_uncertainty=False),
        'full': dict(use_sinc=True, use_graph=True, use_cross_attn=True, use_uncertainty=True),
        # ... other configs
    }
    return HyperEEG_Encoder(**{**configs[config_name], **kwargs})
```

---

## Dimension Flow Summary

```
Input:
    x1, x2: (B, 32, 1024) - Raw EEG segments

Stage 1 (Temporal):
    SincConv/Conv1d: (B, 32, 1024) → (B, 32, T')
    AvgPool1d + Projection: (B, 32, T') → (B, 32, 128)

Stage 2 (Intra-Brain):
    Self-Attention: (B, 32, 128) → (B, 32, 128)
    Treats 32 channels as nodes, 128 as features

Stage 3 (Inter-Brain):
    Cross-Attention: (B, 32, 128) × 2 → (B, 32, 128) × 2
    A queries B, B queries A

Stage 4 (Fusion):
    Pooling: (B, 32, 128) → (B, 128) per person
    Variance estimation + Weighted fusion → (B, 128)

Classifier:
    MLP: (B, 128) → (B, 3)
```

---

## Additional Methods to Implement

### 1. `get_features()` - For analysis compatibility

```python
def get_features(self, x1: torch.Tensor, x2: torch.Tensor) -> dict:
    """
    Extract intermediate features for visualization/analysis.

    Returns:
        dict with keys: 'temporal_a', 'temporal_b', 'intra_a', 'intra_b',
                       'inter_a', 'inter_b', 'fused'
    """
```

### 2. `get_attention_weights()` - For interpretability

```python
def get_attention_weights(self, x1: torch.Tensor, x2: torch.Tensor) -> dict:
    """
    Extract attention weights from graph and cross-attention layers.

    Returns:
        dict with keys: 'intra_attn_a', 'intra_attn_b',
                       'cross_attn_a2b', 'cross_attn_b2a'
    """
```

---

## File Structure

```python
# 3_Models/backbones/hypereeg.py

"""
HyperEEG Encoder for Hyperscanning EEG Classification

Architecture based on fig3.EEGArch.png with 4 progressive stages:
- M1: SincConv1d (Learnable band-pass filtering)
- M2: IntraGraphBlock (Intra-brain topology)
- M3: InterBrainCrossAttn (Inter-brain interaction)
- M4: UncertaintyFusion (Uncertainty-aware fusion)

Supports ablation studies via boolean switches.
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, Optional

# =============================================================================
# M1: Learnable Temporal Encoding
# =============================================================================

class SincConv1d(nn.Module):
    """..."""

# =============================================================================
# M2: Intra-Brain Topology
# =============================================================================

class IntraGraphBlock(nn.Module):
    """..."""

# =============================================================================
# M3: Inter-Brain Interaction
# =============================================================================

class InterBrainCrossAttn(nn.Module):
    """..."""

# =============================================================================
# M4: Uncertainty Fusion
# =============================================================================

class UncertaintyFusion(nn.Module):
    """..."""

# =============================================================================
# Main Model
# =============================================================================

class HyperEEG_Encoder(nn.Module):
    """..."""

# =============================================================================
# Factory Functions
# =============================================================================

def create_hypereeg_model(config_name: str = 'full', **kwargs) -> HyperEEG_Encoder:
    """..."""

# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Test code with shape verification
    ...
```

---

## Key Implementation Notes

### 1. SincConv1d - Critical Details

```python
# Sinc function with numerical stability
def sinc(x):
    # Handle x=0 case
    mask = (x == 0)
    x = torch.where(mask, torch.ones_like(x), x)
    result = torch.sin(x) / x
    result = torch.where(mask, torch.ones_like(result), result)
    return result

# Hamming window
def hamming_window(kernel_size):
    n = torch.arange(kernel_size)
    return 0.54 - 0.46 * torch.cos(2 * math.pi * n / (kernel_size - 1))
```

### 2. Dimension Alignment for Ablation

When `use_cross_attn=False`, ensure the fusion layer still receives correct shapes:
```python
if self.use_cross_attn:
    x1, x2 = self.inter_brain(x1, x2)
else:
    # No modification, pass through
    pass  # x1, x2 remain (B, 32, 128)
```

When `use_uncertainty=False`, fusion should output same dimension:
```python
if self.use_uncertainty:
    fused = self.uncertainty_fusion(x1, x2)  # (B, 128)
else:
    fused = (x1.mean(dim=1) + x2.mean(dim=1)) / 2  # (B, 128)
```

### 3. Shared Weights

Temporal encoder and Intra-brain modules use **shared weights** between two streams:
```python
# In __init__
self.temporal_encoder = SincConv1d(...) or Conv1d(...)  # Single instance
self.intra_brain = IntraGraphBlock(...)  # Single instance

# In forward - same module applied to both
x1 = self.temporal_encoder(x1)
x2 = self.temporal_encoder(x2)  # Same weights
```

---

## Design Decisions (User Confirmed)

1. **Temporal output dimension:** ✅ **Stride convolution** - 使用 stride=8 的卷積直接降維 (1024 → 128)

2. **Intra-Graph layers:** ✅ **1 layer** - 單層 Self-Attention，參數較少，訓練穩定

3. **Residual connections:** ✅ **Yes** - 在 Stage 2, 3 加入殘差連接，改善梯度流動

4. **Pooling strategy:** ✅ **Mean + Max** - 如圖所示，結合均值和最大值 pooling

---

## Updated Architecture Details

### Stage 1: Temporal Encoding with Stride

```python
# SincConv with stride for downsampling
class SincConv1d:
    # kernel_size=251, stride=8
    # Input: (B, 32, 1024)
    # After conv: (B, 32, ~97)  # (1024 - 251) / 8 + 1
    # After projection: (B, 32, 128)
```

### Stage 2 & 3: With Residual Connections

```python
# In forward:
# Stage 2 with residual
identity = x1
x1 = self.intra_brain(x1)
x1 = x1 + identity  # Residual connection

# Stage 3 with residual
identity_a, identity_b = x1, x2
x1, x2 = self.inter_brain(x1, x2)
x1 = x1 + identity_a  # Residual
x2 = x2 + identity_b  # Residual
```

### Stage 4: Mean + Max Pooling

```python
def pool_features(self, x):
    # x: (B, C, D)
    mean_pool = x.mean(dim=1)      # (B, D)
    max_pool = x.max(dim=1)[0]     # (B, D)
    return mean_pool + max_pool    # (B, D)
```
