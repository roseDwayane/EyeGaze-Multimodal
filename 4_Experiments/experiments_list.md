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
- [ ] **EXP-003**: Spatial Concatenation (Horizontal/Vertical)
- [ ] **EXP-004**: Pixel-wise operations (Add/Multiply/Subtract)
- [ ] **EXP-005**: Different ViT variants (vit_small, vit_large, DeiT)
