# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an eye-gaze multimodal classification project using Vision Transformers (ViT) to classify dual-image eye-tracking data into three categories:
- **Single**: Single-player mode
- **Competition**: Competitive mode
- **Cooperation**: Cooperative mode

The key innovation is fusing player1 and player2 eye-gaze images using various fusion strategies before classification.

## Common Commands

### Training

```bash
# Start training from scratch
python Experiments/scripts/train_vit.py

# Resume training from last checkpoint (recommended after interruption)
python Experiments/scripts/train_vit.py --resume

# Resume from specific checkpoint
python Experiments/scripts/train_vit.py --checkpoint path/to/checkpoint-500

# Verify setup before training
python Experiments/scripts/verify_setup.py
```

### Testing Fusion Modes

```bash
# Test visual output of different fusion modes
python Data/processed/test_fusion_simple.py --concat-mode add --num-samples 5
python Data/processed/test_fusion_simple.py --concat-mode multiply --num-samples 5
python Data/processed/test_fusion_simple.py --concat-mode subtract --num-samples 5

# Output saved to: Data/processed/test_outputs/
```

### Monitoring

```bash
# View wandb URL in terminal output after training starts
# Format: wandb: 🚀 View run at https://wandb.ai/...
```

### Data Preparation

```bash
# Generate metadata JSON from raw data
python Data/metadata/generate_json.py

# Verify metadata integrity
python Data/metadata/verify_metadata.py
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Architecture

### Data Pipeline

1. **Metadata Loading** (`Experiments/scripts/train_vit.py:57-87`)
   - Uses HuggingFace `datasets.load_dataset()` to load JSON metadata
   - Applies stratified train/test split via `train_test_split()`
   - Metadata path: `Data/metadata/complete_metadata.json`

2. **Dual-Image Fusion** (`Data/processed/two_image_fusion.py`)
   - `DualImageDataset` class handles fusion of player1 + player2 images
   - Supports 5 fusion modes (configurable via `concat_mode`):
     - `horizontal`: Side-by-side concatenation (6000×1583)
     - `vertical`: Top-to-bottom concatenation (3000×3166)
     - `add`: Pixel-wise averaging (preserves common features)
     - `multiply`: Pixel-wise multiplication (emphasizes overlap)
     - `subtract`: Absolute difference (highlights differences)
   - Each mode has different characteristics for classification tasks

3. **Image Processing**
   - Uses `ViTImageProcessor` from transformers
   - Resizes to 224×224 for ViT input
   - Applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Model Architecture

- **Backbone** (`Models/backbones/vit.py`)
  - Uses `ViTForImageClassification` from HuggingFace transformers
  - Default: `google/vit-base-patch16-224` pretrained model
  - Supports freezing backbone and fine-tuning only classification head
  - `DualImageViTClassifier` wrapper handles dual-image inputs

- **Training** (`Experiments/scripts/train_vit.py`)
  - Follows HuggingFace Trainer API patterns
  - Supports automatic checkpoint resumption
  - Wandb integration for experiment tracking (replaces TensorBoard)
  - Early stopping disabled by default (runs full epochs)

### Evaluation Metrics (`metrics/classification.py`)

- Accuracy, Precision, Recall, F1 (macro and weighted)
- Confusion matrix
- Per-class metrics via `compute_per_class_metrics()`
- Compatible with HuggingFace Trainer's `compute_metrics` callback

## Configuration

Main config: `Experiments/configs/vit_single_vs_competition.yaml`

### Key Configuration Sections

**Model Settings**:
- `model.model_name`: Pretrained ViT model variant
- `model.num_labels`: 3 (Single/Competition/Cooperation)
- `model.concat_mode`: Fusion strategy (horizontal/vertical/add/multiply/subtract)
- `model.freeze_backbone`: Whether to freeze ViT backbone

**Data Settings**:
- `data.metadata_path`: Path to complete_metadata.json
- `data.image_base_path`: Base directory for eye-gaze images
- `data.train_test_split`: Test set ratio (default: 0.2)
- `data.random_seed`: Reproducibility seed (default: 42)

**Training Settings**:
- `training.output_dir`: Where checkpoints/logs are saved (e.g., `Experiments/outputs/vit_class_subtract/`)
- `training.num_train_epochs`: Total training epochs
- `training.per_device_train_batch_size`: Batch size per GPU
- `training.learning_rate`: Learning rate (default: 2.0e-5)
- `training.save_strategy`: "epoch" (saves checkpoint each epoch)
- `training.save_total_limit`: 3 (keeps only last 3 checkpoints)
- `training.metric_for_best_model`: "f1" (selects best model by F1 score)
- `training.report_to`: ["wandb"] (experiment tracking)

**Wandb Settings**:
- `wandb.project`: "eyegaze-vit-classification"
- `wandb.tags`: List of tags for filtering experiments

## Training Resume Feature

The training script supports robust checkpoint resumption:

1. **Auto-detection**: `--resume` flag automatically finds latest checkpoint in output_dir
2. **Wandb Continuity**: Attempts to resume same wandb run for continuous metrics
3. **State Preservation**: Restores model weights, optimizer state, learning rate scheduler, random seed
4. **Checkpoint Structure**: Each checkpoint contains:
   - `model.safetensors`: Model weights
   - `optimizer.pt`: Optimizer state
   - `scheduler.pt`: LR scheduler state
   - `trainer_state.json`: Training progress metadata

**Important**: When resuming, avoid changing:
- Model architecture (`model_name`, `num_labels`, `concat_mode`)
- Dataset (`metadata_path`, `train_test_split`, `random_seed`)

Safe to change: `num_train_epochs` (to extend training), logging parameters

## Experiment Workflow

Typical experiment iteration:

1. **Choose fusion mode**: Edit `concat_mode` in config YAML
2. **Test visually** (optional): Run `test_fusion_simple.py` to preview fusion output
3. **Update output_dir**: Change `training.output_dir` to match fusion mode (e.g., `vit_class_add`)
4. **Start training**: Run `train_vit.py`
5. **Monitor on wandb**: Click URL in terminal output
6. **Compare experiments**: Use wandb dashboard to compare different fusion modes

## File Organization

```
Data/
  metadata/
    complete_metadata.json       # Master dataset metadata
    generate_json.py             # Script to generate metadata
    verify_metadata.py           # Validation script
  processed/
    two_image_fusion.py          # DualImageDataset and fusion logic
    test_fusion_simple.py        # Visual testing of fusion modes
  raw/                           # Raw eye-gaze images

Models/
  backbones/
    vit.py                       # ViT model definitions

Experiments/
  configs/
    vit_single_vs_competition.yaml  # Main training config
  scripts/
    train_vit.py                 # Training script
    verify_setup.py              # Pre-training validation
  outputs/
    vit_class_{mode}/            # Per-experiment outputs
      checkpoint-{N}/            # Saved checkpoints
      logs/                      # Training logs
      wandb/                     # Wandb run data

metrics/
  classification.py              # Evaluation metrics
```

## Troubleshooting

**CUDA Out of Memory**:
- Reduce `per_device_train_batch_size` in config (try 4 instead of 8)
- Use smaller model: `google/vit-small-patch16-224`
- Enable gradient accumulation: set `gradient_accumulation_steps: 2`

**Resume Not Working**:
- Check checkpoint directory exists: `ls Experiments/outputs/vit_class_*/checkpoint-*`
- Ensure checkpoint format is `checkpoint-{number}`
- Use `--checkpoint path/to/checkpoint` to specify manually

**Image Loading Errors**:
- Verify `image_base_path` points to correct directory
- Ensure images are `.jpg` format
- Check player1/player2 names in metadata match actual filenames

**Low Accuracy**:
- Try different fusion modes (subtract often works well for Single vs Competition)
- Increase `num_train_epochs`
- Enable data augmentation: `augmentation.enabled: true`
- Adjust learning rate

## Notes

- The codebase is in Chinese (Traditional) but code/comments are mixed English/Chinese
- All training follows HuggingFace Transformers best practices
- Wandb is the primary experiment tracking tool (TensorBoard deprecated)
- Early stopping has been removed - training runs full epoch count
- Windows environment: Use standard Python paths, not Unix-style
