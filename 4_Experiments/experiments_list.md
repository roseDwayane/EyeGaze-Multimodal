# Experiments List

**Project Path**: `C:/Users/user/pythonproject/EyeGaze-Multimodal_new`
**Last Updated**: 2025-01-04

---

## Gaze-Only Experiments

### EXP-001: Early Fusion ViT (Channel Concatenation)

#### Overview

| Item | Value |
|------|-------|
| **Date** | 2025-01-04 |
| **Model** | EarlyFusionViT (vit_base_patch16_224) |
| **Task** | 3-class classification (Single/Competition/Cooperation) |
| **Fusion Strategy** | Channel Concatenation (6-channel input) |
| **Status** | Ready to run |

#### Training Commands

```bash
# Basic training
cd C:/Users/user/pythonproject/EyeGaze-Multimodal_new
python 4_Experiments/scripts/train_gaze_earlyfusion.py

# With specific config
python 4_Experiments/scripts/train_gaze_earlyfusion.py --config 4_Experiments/configs/gaze_earlyfusion.yaml

# Resume from checkpoint
python 4_Experiments/scripts/train_gaze_earlyfusion.py --resume 4_Experiments/runs/gaze_earlyfusion/checkpoint_epoch_10.pt
```

#### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | AdamW | |
| **Learning Rate** | 5e-5 | Linear scaling: batch 64 -> LR 2e-4 |
| **Weight Decay** | 0.01 | |
| **Batch Size** | 16 | Recommended: 64 for 24GB GPU |
| **Epochs** | 50 | No early stopping |
| **LR Scheduler** | Cosine Annealing | With linear warmup |
| **Warmup Epochs** | 5 | |
| **Mixed Precision** | FP16 | Enabled |
| **Gradient Clipping** | 1.0 | max_grad_norm |
| **Class Weights** | Weighted Loss | Inverse frequency |

#### Data Split

| Split | Samples | Pairs |
|-------|---------|-------|
| Train | 3,187 | 12-32 |
| Val | 1,276 | 33-40 |

#### Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | 86,390,787 |
| GPU Memory (FP16, batch=16) | ~4 GB |
| GPU Memory (FP16, batch=64) | ~14-15 GB |

#### Files

| Type | Path |
|------|------|
| Model | `3_Models/backbones/early_fusion_vit.py` |
| Dataset | `1_Data/datasets/gaze_pair_dataset.py` |
| Training Script | `4_Experiments/scripts/train_gaze_earlyfusion.py` |
| Config | `4_Experiments/configs/gaze_earlyfusion.yaml` |
| Checkpoints | `4_Experiments/runs/gaze_earlyfusion/` |

#### Wandb Config

| Parameter | Value |
|-----------|-------|
| Project | `Multimodal_Gaze` |
| Run Name | `early_fusion_vit` |
| Tags | `early-fusion`, `vit`, `gaze`, `channel-concat` |

---

### EXP-002: Late Fusion ViT (Siamese Encoder + Feature Fusion)

#### Overview

| Item | Value |
|------|-------|
| **Date** | 2025-01-04 |
| **Model** | LateFusionViT (vit_base_patch16_224) |
| **Task** | 3-class classification (Single/Competition/Cooperation) |
| **Fusion Strategy** | Shared encoder (Siamese) + CLS token fusion |
| **Status** | Ready to run |

#### Architecture

```
Stream 1 (B, 3, H, W) ──┐                      ┌── CLS_1 ──┐
                       ├── Shared ViT Encoder ─┤          ├── Fusion ──► Classification
Stream 2 (B, 3, H, W) ──┘                      └── CLS_2 ──┘
```

#### Fusion Modes

| Mode | Description | Output Dim |
|------|-------------|------------|
| `concat` | [CLS_1, CLS_2] | 1536 (2D) |
| `add` | CLS_1 + CLS_2 | 768 (D) |
| `subtract` | CLS_1 - CLS_2 | 768 (D) |
| `multiply` | CLS_1 * CLS_2 | 768 (D) |
| `full` | [concat, subtract, multiply] | 3072 (4D) |

#### Training Commands (Verified Working)

```bash
# Basic training (uses default config)
cd C:/Users/user/pythonproject/EyeGaze-Multimodal_new
python 4_Experiments/scripts/train_gaze_latefusion.py

# With specific config file
python 4_Experiments/scripts/train_gaze_latefusion.py --config 4_Experiments/configs/gaze_latefusion.yaml

# Resume from checkpoint (note: path includes fusion_mode subfolder)
python 4_Experiments/scripts/train_gaze_latefusion.py --resume 4_Experiments/runs/gaze_latefusion/full/checkpoint_epoch_10.pt
```

**Note**: Checkpoints are saved to `save_dir/{fusion_mode}/` automatically based on config.

#### Hyperparameters (Finalized)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | AdamW | Weight decay = 0.01 |
| **Learning Rate** | 5e-5 | Linear warmup + Cosine decay |
| **Batch Size** | 16 (32 w/ accumulation) | Effective batch = 32 on 24GB GPU |
| **Epochs** | 50 | No early stopping |
| **LR Scheduler** | Cosine Annealing | With 5 epochs linear warmup |
| **Mixed Precision** | FP16 | Enabled for memory efficiency |
| **Gradient Clipping** | 1.0 | max_grad_norm |
| **Dropout** | 0.1 | Before classification head |
| **Fusion Mode** | full | Default (3072-dim = 4D) |
| **Class Weights** | Weighted Loss | Inverse frequency for imbalance |

#### Data Split

| Split | Samples | Pairs |
|-------|---------|-------|
| Train | 3,187 | 12-32 |
| Val | 1,276 | 33-40 |

#### Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | ~86M (shared encoder) |
| Classifier Params | 3072 * 3 = 9,219 (full mode) |

#### Files

| Type | Path |
|------|------|
| Model | `3_Models/backbones/late_fusion_vit.py` |
| Dataset | `1_Data/datasets/gaze_pair_dataset.py` |
| Training Script | `4_Experiments/scripts/train_gaze_latefusion.py` |
| Config | `4_Experiments/configs/gaze_latefusion.yaml` |
| Checkpoints | `4_Experiments/runs/gaze_latefusion/{fusion_mode}/` |

#### Wandb Config

| Parameter | Value |
|-----------|-------|
| Project | `Multimodal_Gaze` |
| Run Name | `late_fusion_vit` |
| Tags | `late-fusion`, `vit`, `gaze`, `siamese` |

---

## Planned Experiments

- [x] **EXP-001**: Early Fusion ViT (channel concatenation)
- [x] **EXP-002**: Late Fusion ViT (Siamese encoder + feature fusion)
- [x] **EXP-003**: HyperEEG Encoder (Ablation Studies)
- [ ] **EXP-004**: Spatial Concatenation (Horizontal/Vertical)
- [ ] **EXP-005**: Pixel-wise operations (Add/Multiply/Subtract)
- [ ] **EXP-006**: Different ViT variants (vit_small, vit_large, DeiT)

---

## EEG-Only Experiments

### EXP-003: HyperEEG Encoder (Ablation Studies)

#### Overview

| Item | Value |
|------|-------|
| **Date** | 2026-01-07 |
| **Model** | HyperEEG Encoder (Dual-Stream Siamese) |
| **Task** | 3-class classification (Single/Competition/Cooperation) |
| **Fusion Strategy** | 4-stage progressive fusion (M1→M2→M3→M4) |
| **Status** | Completed |

#### Architecture

```
EEG_A (B, 32, 1024) ──┐
                      ├── [M1: TemporalBlock] ──► (B, 32, 128) × 2
EEG_B (B, 32, 1024) ──┘         (Shared)
                                    │
                                    ▼
                         [M2: IntraGraphBlock] ──► (B, 32, 128) × 2
                                (Shared)          Self-Attention
                                    │
                                    ▼
                        [M3: InterBrainCrossAttn] ──► (B, 32, 128) × 2
                              Cross-Attention
                                    │
                                    ▼
                         [M4: UncertaintyFusion] ──► (B, 128)
                           Inverse-Variance Weighted
                                    │
                                    ▼
                              [Classifier] ──► (B, 3)
```

#### Core Modules

| Module | Component | Function | Ablation Flag |
|--------|-----------|----------|---------------|
| **M1** | SincConv1d | Learnable band-pass filtering | `use_sinc` |
| **M2** | IntraGraphBlock | Intra-brain channel connectivity | `use_graph` |
| **M3** | InterBrainCrossAttn | Inter-brain cross-attention | `use_cross_attn` |
| **M4** | UncertaintyFusion | Inverse-variance weighted fusion | `use_uncertainty` |

#### Training Commands

```bash
# Step 1: Preprocess data (one-time, ~5-10 min)
python 2_Preprocessing/scripts/preprocess_eeg_windows.py

# Step 2a: Run single experiment
python 4_Experiments/scripts/train_eeg_hypereeg.py --ablation full
python 4_Experiments/scripts/train_eeg_hypereeg.py --ablation baseline
python 4_Experiments/scripts/train_eeg_hypereeg.py --ablation no_sinc
python 4_Experiments/scripts/train_eeg_hypereeg.py --ablation no_graph
python 4_Experiments/scripts/train_eeg_hypereeg.py --ablation no_cross
python 4_Experiments/scripts/train_eeg_hypereeg.py --ablation no_uncertainty

# Step 2b: Run all ablation experiments (automated)
python run_experiments.py
```

#### Ablation Configurations

| Config | M1 (SincConv) | M2 (Graph) | M3 (CrossAttn) | M4 (Uncertainty) |
|--------|:-------------:|:----------:|:--------------:|:----------------:|
| `full` | ✓ | ✓ | ✓ | ✓ |
| `baseline` | ✗ | ✗ | ✗ | ✗ |
| `no_sinc` | ✗ | ✓ | ✓ | ✓ |
| `no_graph` | ✓ | ✗ | ✓ | ✓ |
| `no_cross` | ✓ | ✓ | ✗ | ✓ |
| `no_uncertainty` | ✓ | ✓ | ✓ | ✗ |

#### Hyperparameters (Finalized)

**Model Architecture**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `in_channels` | 32 | EEG channels |
| `in_timepoints` | 1024 | Timepoints per window |
| `embed_dim` | 128 | Feature embedding dimension |
| `num_heads` | 4 | Attention heads |
| `dropout` | 0.1 | Dropout probability |
| `sample_rate` | 250 | EEG sampling rate (Hz) |
| `sinc_kernel_size` | 125 | SincConv kernel size |

**Training**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 150 | Training epochs |
| `batch_size` | 256 | Optimized for RTX 4070 12GB |
| `learning_rate` | 5e-4 | Initial LR |
| `weight_decay` | 0.01 | L2 regularization |
| `warmup_epochs` | 10 | Linear warmup epochs |
| `scheduler` | cosine | Cosine annealing |
| `fp16` | true | Mixed precision training |
| `max_grad_norm` | 1.0 | Gradient clipping |

**Data**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `window_size` | 1024 | Sliding window size |
| `stride` | 256 | Sliding stride (75% overlap) |
| `filter_low` | 0.5 | High-pass cutoff (Hz) |
| `filter_high` | 50.0 | Low-pass cutoff (Hz) |
| `val_pairs` | [33-40] | Validation Pair IDs |

**Data Augmentation** (Training only):

| Augmentation | Parameters |
|--------------|------------|
| Time Masking | `max_length=50, num_masks=2` |
| Channel Dropout | `prob=0.2` |
| Gaussian Noise | `std=0.05` |

#### Data Statistics

| Split | Samples | Windows | Single | Competition | Cooperation |
|-------|---------|---------|--------|-------------|-------------|
| Train | 3,187 | 28,683 | 14,346 | 7,155 | 7,182 |
| Val | 1,276 | 11,484 | 5,751 | 2,853 | 2,880 |

#### Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | ~680K |
| GPU Memory (FP16, batch=256) | ~4 GB |
| Training Time (per epoch) | ~1 min |

#### Files

| Type | Path |
|------|------|
| Model | `3_Models/backbones/hypereeg.py` |
| Training Script | `4_Experiments/scripts/train_eeg_hypereeg.py` |
| Preprocessing | `2_Preprocessing/scripts/preprocess_eeg_windows.py` |
| Config | `4_Experiments/configs/eeg_hypereeg.yaml` |
| Checkpoints | `4_Experiments/runs/eeg_hypereeg/{ablation_mode}/` |
| Preprocessed Data | `1_Data/datasets/EEGseg_preprocessed/` |

#### Wandb Config

| Parameter | Value |
|-----------|-------|
| Project | `Multimodal_EEG` |
| Run Name | `hypereeg_{ablation_mode}_{timestamp}` |
| Tags | `hypereeg`, `hyperscanning`, `dual-eeg`, `social-interaction` |

python 2_Preprocessing/scripts/preprocess_eeg_raw.py

python 4_Experiments/scripts/train_eeg_hypereeg.py --config 4_Experiments/configs/eeg_hypereeg_raw.yaml