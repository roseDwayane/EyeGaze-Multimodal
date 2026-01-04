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

## Planned Experiments

- [ ] **EXP-002**: Late Fusion ViT (separate encoders + feature fusion)
- [ ] **EXP-003**: Spatial Concatenation (Horizontal/Vertical)
- [ ] **EXP-004**: Pixel-wise operations (Add/Multiply/Subtract)
- [ ] **EXP-005**: Different ViT variants (vit_small, vit_large, DeiT)
