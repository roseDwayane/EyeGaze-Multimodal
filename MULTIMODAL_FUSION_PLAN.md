# Multimodal Fusion Plan: Eye Gaze Image × EEG Signal

## Overview

**Goal**: Fuse Eye Gaze images and EEG signals for enhanced social interaction classification (Single/Competition/Cooperation).

**Data Source**: `Data/metadata/complete_metadata.json`
- Player 1 & 2 Eye Gaze images
- Player 1 & 2 EEG signals
- Interaction labels

**Existing Models**:
- ViT for Eye Gaze images (`Models/backbones/vit.py`)
- Dual EEG Transformer (`Models/backbones/dual_eeg_transformer.py`)

---

## Architecture Strategy

### A. Late Fusion (Baseline - Simplest)

**Architecture**:
```
┌─────────────┐          ┌─────────────┐
│   ViT Model │          │  EEG Model  │
│ (Pre-trained)│          │(Pre-trained)│
└──────┬──────┘          └──────┬──────┘
       │                        │
       │ logits_img (B, 3)     │ logits_eeg (B, 3)
       │                        │
       └────────┬───────────────┘
                │
         ┌──────▼──────┐
         │Weighted Avg │
         │  or MLP     │
         └──────┬──────┘
                │
         Final Logits (B, 3)
```

**Implementation**:
- Load pre-trained ViT and EEG models
- Freeze or fine-tune
- Two strategies:
  1. **Weighted Average**: `logits = α * logits_img + β * logits_eeg`
  2. **MLP Fusion**: Concat features from last hidden layer → MLP

**Pros**: Simple, interpretable, fast to implement
**Cons**: Limited interaction between modalities

---

### B. Mid Fusion (Main Model - Recommended)

**Four-Tower Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA                                │
│  [P1_img, P2_img, P1_eeg, P2_eeg]                           │
└──────┬────────────┬──────────────┬──────────────┬───────────┘
       │            │              │              │
   ┌───▼───┐   ┌───▼───┐     ┌────▼────┐   ┌────▼────┐
   │ViT-P1 │   │ViT-P2 │     │EEG-P1   │   │EEG-P2   │
   │Encoder│   │Encoder│     │Encoder  │   │Encoder  │
   └───┬───┘   └───┬───┘     └────┬────┘   └────┬────┘
       │            │              │              │
    z_img_p1     z_img_p2      z_eeg_p1      z_eeg_p2
       │            │              │              │
       └─────┬──────┘              └──────┬───────┘
             │                            │
    ┌────────▼────────┐          ┌────────▼────────┐
    │ Intra-Modality  │          │ Intra-Modality  │
    │ Fusion (Image)  │          │ Fusion (EEG)    │
    │ 4 Operators:    │          │ 4 Operators:    │
    │ Sum, Mul, Diff, │          │ Sum, Mul, Diff, │
    │ Concat          │          │ Concat          │
    └────────┬────────┘          └────────┬────────┘
             │                            │
          z_img                        z_eeg
             │                            │
             └──────────┬─────────────────┘
                        │
              ┌─────────▼─────────┐
              │  Cross-Modal      │
              │  Interaction      │
              │  (Cross-Attn +    │
              │   IBS Token)      │
              └─────────┬─────────┘
                        │
                 z_fused (B, d)
                        │
              ┌─────────▼─────────┐
              │ Classification    │
              │ Head (MLP)        │
              └─────────┬─────────┘
                        │
                  Logits (B, 3)
```

**Key Components**:

1. **Four-Tower Encoders**:
   - **Image Encoders**: 2× ViT (shared weights or independent)
     - Input: P1_img (B, 3, H, W), P2_img (B, 3, H, W)
     - Output: z_img_p1 (B, d), z_img_p2 (B, d)

   - **EEG Encoders**: 2× Temporal Conv + Transformer (shared weights)
     - Input: P1_eeg (B, C, T), P2_eeg (B, C, T)
     - Output: z_eeg_p1 (B, d), z_eeg_p2 (B, d)

2. **Intra-Modality Fusion** (Symmetric Operators):
   ```python
   # For images
   z_img_sum = z_img_p1 + z_img_p2
   z_img_mul = z_img_p1 * z_img_p2
   z_img_diff = |z_img_p1 - z_img_p2|
   z_img = Proj([z_img_sum, z_img_mul, z_img_diff])  # (B, d)

   # For EEG (same)
   z_eeg = Proj([z_eeg_sum, z_eeg_mul, z_eeg_diff])  # (B, d)
   ```

3. **Inter-Brain Synchrony (IBS) Token**:
   ```python
   # Compute cross-brain features from raw EEG
   ibs_features = compute_IBS(P1_eeg, P2_eeg)  # PLV, power corr, etc.
   ibs_token = MLP(ibs_features)  # (B, d)
   ```

4. **Cross-Modal Interaction**:
   - **Bidirectional Cross-Attention**:
     ```python
     z_img' = z_img + CrossAttn(Q=z_img, K=z_eeg, V=z_eeg)
     z_eeg' = z_eeg + CrossAttn(Q=z_eeg, K=z_img, V=z_img)
     ```

   - **Fusion with IBS Token**:
     ```python
     z_fused = Concat([z_img', z_eeg', ibs_token])  # (B, 3d)
     ```

5. **Classification Head**:
   ```python
   logits = MLP(z_fused)  # (B, 3d) → (B, 3)
   ```

**Pros**:
- Explicit cross-modal interaction
- IBS token provides domain knowledge
- Symmetric operators preserve permutation invariance
**Cons**: More complex, requires careful tuning

---

### C. Early Fusion (Ablation Baseline)

**Architecture**:
```
┌──────────────────────────┐
│ EEG → Spectrogram        │
│ (Time-Freq Image)        │
└───────────┬──────────────┘
            │
┌───────────▼──────────────┐
│ Concat with Eye Gaze     │
│ [RGB (3ch) + TF (4ch)]   │
│ = 7 channels             │
└───────────┬──────────────┘
            │
┌───────────▼──────────────┐
│ Single ViT Encoder       │
│ (7-channel input)        │
└───────────┬──────────────┘
            │
        Logits (B, 3)
```

**Implementation**:
- Convert EEG to time-frequency representation (e.g., STFT, Wavelet)
- Stack as additional channels to Eye Gaze image
- Feed into modified ViT with 7-channel input

**Pros**: Single unified model
**Cons**: Modality differences may hurt performance

---

## Implementation Plan

### Phase 1: Late Fusion (Week 1)

**Files to Create**:
- `Models/fusion/late_fusion.py` - Late fusion model
- `Data/processed/multimodal_dataset.py` - Dataset loader for both modalities
- `Experiments/scripts/train_late_fusion.py` - Training script
- `Experiments/configs/late_fusion.yaml` - Config file

**Steps**:
1. Load pre-trained ViT and EEG models
2. Create multimodal dataset (loads both image and EEG)
3. Implement weighted average and MLP fusion
4. Train and evaluate

### Phase 2: Mid Fusion (Week 2-3)

**Files to Create**:
- `Models/fusion/mid_fusion.py` - Four-tower + cross-modal attention
- `Models/fusion/cross_modal_attention.py` - Cross-attention module
- `Models/fusion/symmetric_fusion.py` - Intra-modality fusion operators
- `Experiments/scripts/train_mid_fusion.py` - Training script
- `Experiments/configs/mid_fusion.yaml` - Config file

**Steps**:
1. Implement four-tower architecture
2. Add intra-modality fusion (symmetric operators)
3. Add IBS token generator
4. Add cross-modal attention
5. Train end-to-end

### Phase 3: Early Fusion (Week 4)

**Files to Create**:
- `Models/fusion/early_fusion.py` - Modified ViT with 7 channels
- `Experiments/scripts/train_early_fusion.py` - Training script
- `Experiments/configs/early_fusion.yaml` - Config file

**Steps**:
1. Convert EEG to spectrogram
2. Modify ViT input layer
3. Train and compare

---

## Training Strategy

### Data Pipeline

**Metadata Format** (`complete_metadata.json`):
```json
{
  "player1": "P1_001",
  "player2": "P2_003",
  "class": "Cooperation",
  "image_player1": "path/to/img1.png",
  "image_player2": "path/to/img2.png",
  "eeg_player1": "path/to/eeg1.csv",
  "eeg_player2": "path/to/eeg2.csv"
}
```

**MultimodalDataset**:
```python
def __getitem__(self, idx):
    # Load images
    img1 = load_image(self.metadata[idx]['image_player1'])
    img2 = load_image(self.metadata[idx]['image_player2'])

    # Load EEG
    eeg1 = load_eeg(self.metadata[idx]['eeg_player1'])
    eeg2 = load_eeg(self.metadata[idx]['eeg_player2'])

    # Label
    label = self.label2id[self.metadata[idx]['class']]

    return {
        'img1': img1, 'img2': img2,
        'eeg1': eeg1, 'eeg2': eeg2,
        'labels': label
    }
```

### Loss Functions

**Primary Loss**: Cross-Entropy
```python
L_ce = CrossEntropy(logits, labels)
```

**Auxiliary Losses** (for Mid Fusion):
1. **Modality-Specific Losses**:
   ```python
   L_img = CrossEntropy(logits_img_only, labels)
   L_eeg = CrossEntropy(logits_eeg_only, labels)
   ```

2. **Contrastive Loss** (align cross-modal representations):
   ```python
   L_contrast = InfoNCE(z_img, z_eeg, labels)
   ```

3. **IBS Alignment Loss**:
   ```python
   L_ibs = MSE(ibs_token, target_synchrony)
   ```

**Total Loss**:
```python
L_total = L_ce + λ_img * L_img + λ_eeg * L_eeg + λ_contrast * L_contrast
```

### Training Protocol

**Stage 1: Warm-up (Epochs 1-10)**
- Freeze pre-trained encoders
- Train only fusion layers
- Learning rate: 1e-3

**Stage 2: Fine-tuning (Epochs 11-50)**
- Unfreeze all layers
- Lower learning rate: 1e-4
- Gradually reduce λ for auxiliary losses

---

## Evaluation Metrics

1. **Accuracy**: Overall classification accuracy
2. **Macro F1**: Average F1 across 3 classes
3. **Confusion Matrix**: Class-wise performance
4. **Ablation Studies**:
   - Image-only vs EEG-only vs Multimodal
   - With/without IBS token
   - With/without cross-attention

---

## Expected Results

| Method | Expected Accuracy | Expected F1 |
|--------|------------------|-------------|
| Image-only (ViT) | ~65% | ~0.55 |
| EEG-only (Dual Transformer) | ~70% | ~0.60 |
| **Late Fusion** | **~75%** | **~0.68** |
| **Mid Fusion (Main)** | **~80%** | **~0.75** |
| Early Fusion | ~72% | ~0.65 |

---

## File Structure

```
EyeGaze-Multimodal/
│
├── Models/
│   ├── backbones/
│   │   ├── vit.py                    # Existing
│   │   └── dual_eeg_transformer.py   # Existing
│   │
│   └── fusion/
│       ├── late_fusion.py            # NEW
│       ├── mid_fusion.py             # NEW
│       ├── early_fusion.py           # NEW
│       ├── cross_modal_attention.py  # NEW
│       └── symmetric_fusion.py       # NEW
│
├── Data/
│   ├── processed/
│   │   ├── dual_image_dataset.py     # Existing
│   │   ├── dual_eeg_dataset.py       # Existing
│   │   └── multimodal_dataset.py     # NEW
│   │
│   └── metadata/
│       └── complete_metadata.json    # Existing
│
├── Experiments/
│   ├── scripts/
│   │   ├── train_vit.py              # Existing
│   │   ├── train_art.py              # Existing
│   │   ├── train_late_fusion.py      # NEW
│   │   ├── train_mid_fusion.py       # NEW
│   │   └── train_early_fusion.py     # NEW
│   │
│   └── configs/
│       ├── vit_fusion.yaml           # Existing
│       ├── dual_eeg_transformer.yaml # Existing
│       ├── late_fusion.yaml          # NEW
│       ├── mid_fusion.yaml           # NEW
│       └── early_fusion.yaml         # NEW
│
└── MULTIMODAL_FUSION_PLAN.md         # THIS FILE
```

---

## Next Steps

1. **Implement Late Fusion first** (simplest baseline)
2. **Implement Mid Fusion** (main contribution)
3. **Implement Early Fusion** (ablation)
4. **Compare all methods** and write paper

Let's start with **Late Fusion**!
