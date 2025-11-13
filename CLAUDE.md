# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multimodal classification project for eye-gaze and EEG signals to classify interaction modes into three categories:
- **Single**: Single-player mode
- **Competition**: Competitive mode
- **Cooperation**: Cooperative mode

### Two Training Pipelines

1. **Vision Transformer (ViT)**: Fuses player1 and player2 eye-gaze images using various fusion strategies (horizontal/vertical/add/multiply/subtract) before classification.

2. **Dual EEG Transformer (NEW)**: Fuses EEG signals from two players using:
   - Temporal convolution frontend for downsampling
   - IBS (Inter-Brain Synchrony) token for cross-brain features
   - Siamese Transformer encoder with cross-brain attention
   - Symmetric fusion for permutation-invariant classification

## Common Commands

### Training - Vision Transformer (Eye-Gaze Images)

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

### Training - Dual EEG Transformer (NEW)

```bash
# Train dual EEG transformer with IBS token
python Experiments/scripts/train_art.py --config Experiments/configs/dual_eeg_transformer.yaml

# Monitor training on wandb (URL shown in terminal)
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

### Vision Transformer Pipeline (Eye-Gaze Images)

#### Data Pipeline

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

## Dual EEG Transformer Pipeline (NEW)

### Architecture Overview

**File**: `Models/backbones/dual_eeg_transformer.py`

The Dual EEG Transformer processes paired EEG signals with the following stages:

#### 1. Temporal Convolution Frontend (`TemporalConvFrontend`)
- Input: (B, C, T) raw EEG where C=channels, T=timepoints
- Multiple Conv1d layers with stride to downsample: T → T̃
- Output: (B, T̃, d_model) embedded sequences

#### 2. IBS Token Generator (`IBSTokenGenerator`)
- Computes Inter-Brain Synchrony features from dual EEG
- Frequency bands: theta, alpha, beta, gamma
- Features per band:
  - PLV (Phase Locking Value)
  - Power correlation
  - Phase difference
- Output: (B, d_model) shared IBS token

#### 3. Token Sequence Construction
- Each player sequence: [CLS, IBS, H₁(1), H₁(2), ..., H₁(T̃)]
- CLS: Learnable classification token
- IBS: Shared Inter-Brain Synchrony token
- H: Temporal conv embeddings

#### 4. Siamese Transformer Encoder
- Shared weights process both players' sequences
- Output: Z₁, Z₂ ∈ ℝ^{(T̃+2)×d_model}

#### 5. Cross-Brain Attention (`CrossBrainAttention`)
- Bidirectional: Z₁ ↔ Z₂
- Allows information exchange between players
- Output: Z₁', Z₂' (cross-attended)

#### 6. Symmetric Fusion (`SymmetricFusion`)
- Extracts CLS tokens: cls₁, cls₂
- Symmetric operations: add, multiply, abs_diff, concat
- Ensures f(z₁, z₂) = f(z₂, z₁)
- Output: f_pair

#### 7. Classification
- Concatenate: [f_pair, mp₁', mp₂'] where mp=mean pooling
- MLP classifier → logits

### Loss Functions

**Main loss**: Cross-entropy L_ce

**Optional losses** (disabled by default, enable after baseline converges):
- Symmetry loss L_sym = ||cls₁ - cls₂||²
- IBS alignment loss L_ibs (InfoNCE between IBS token and CLS tokens)

Total: L = L_ce + λ_sym·L_sym + λ_ibs·L_ibs

### Data Format

**EEG Files**: CSV format at `Data/raw/EEG/example/`
- Format: (Channels, Timepoints) or transposed
- Sampling rate: 250 Hz (configurable)
- Preprocessing: bandpass filter (1-45 Hz), CAR, z-score normalization

**Windowing**: Sliding window approach
- window_size: 1000 samples (4 seconds @ 250Hz)
- stride: 500 samples (2 seconds overlap)
- Creates multiple training samples per trial

**Dataset**: `Data/processed/dual_eeg_dataset.py`
- DualEEGDataset class handles loading and preprocessing
- Automatically creates valid windows from all trials

## Configuration

### Vision Transformer Config

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

### Dual EEG Transformer Config

Main config: `Experiments/configs/dual_eeg_transformer.yaml`

**Key EEG-specific settings**:
- `model.in_channels`: Number of EEG channels (default: 62)
- `model.d_model`: Transformer embedding dimension (default: 256)
- `model.num_layers`: Transformer depth (default: 6)
- `model.conv_kernel_size`: Temporal conv kernel (default: 25)
- `model.conv_stride`: Downsampling rate (default: 4)
- `data.eeg_base_path`: Path to EEG CSV files
- `data.window_size`: EEG window in samples (default: 1000 = 4s @ 250Hz)
- `data.stride`: Sliding window stride (default: 500 = 2s overlap)
- `training.use_sym_loss`: Enable symmetry loss (start with false)
- `training.use_ibs_loss`: Enable IBS alignment loss (start with false)

**Important**: Start training with only L_ce (cross-entropy), enable optional losses after baseline converges.

## Experiment Workflow

### Vision Transformer (Eye-Gaze)

Typical experiment iteration:

1. **Choose fusion mode**: Edit `concat_mode` in config YAML
2. **Test visually** (optional): Run `test_fusion_simple.py` to preview fusion output
3. **Update output_dir**: Change `training.output_dir` to match fusion mode (e.g., `vit_class_add`)
4. **Start training**: Run `train_vit.py`
5. **Monitor on wandb**: Click URL in terminal output
6. **Compare experiments**: Use wandb dashboard to compare different fusion modes

### Dual EEG Transformer

Typical experiment iteration:

1. **Verify EEG data**: Ensure CSV files exist at `data.eeg_base_path`
2. **Check channels**: Confirm `model.in_channels` matches your EEG setup
3. **Baseline training**: Start with default config (only L_ce loss)
4. **Monitor convergence**: Watch F1 score on validation set
5. **Enable optional losses**: After baseline converges, try adding L_sym and L_ibs
6. **Hyperparameter tuning**: Experiment with d_model, num_layers, learning_rate

## File Organization

```
Data/
  metadata/
    complete_metadata.json       # Master dataset metadata
    generate_json.py             # Script to generate metadata
    verify_metadata.py           # Validation script
  processed/
    two_image_fusion.py          # DualImageDataset (eye-gaze) and fusion logic
    test_fusion_simple.py        # Visual testing of fusion modes
    dual_eeg_dataset.py          # DualEEGDataset (NEW)
  raw/
    Gaze/                        # Raw eye-gaze images
    EEG/example/                 # Raw EEG CSV files (NEW)

Models/
  backbones/
    vit.py                       # ViT model definitions
    dual_eeg_transformer.py      # Dual EEG Transformer (NEW)
    art.py                       # Base Transformer components

Experiments/
  configs/
    vit_single_vs_competition.yaml  # Vision Transformer config
    dual_eeg_transformer.yaml        # Dual EEG Transformer config (NEW)
  scripts/
    train_vit.py                 # ViT training script
    train_art.py                 # Dual EEG training script (NEW)
    verify_setup.py              # Pre-training validation
  outputs/
    vit_class_{mode}/            # ViT outputs by fusion mode
      checkpoint-{N}/            # Saved checkpoints
      logs/                      # Training logs
      wandb/                     # Wandb run data
    dual_eeg_transformer/        # Dual EEG outputs (NEW)
      best_model.pt              # Best model checkpoint
      checkpoint-epoch-{N}.pt    # Periodic checkpoints

metrics/
  classification.py              # Evaluation metrics
```

## Troubleshooting

### Vision Transformer Issues

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

### Dual EEG Transformer Issues

**CUDA Out of Memory**:
- Reduce `per_device_train_batch_size` (try 8 or 4)
- Reduce `model.d_model` (try 128 instead of 256)
- Reduce `data.window_size` (try 500 instead of 1000)

**EEG File Loading Errors**:
- Verify CSV files exist at `data.eeg_base_path`
- Check CSV format: should be (Channels, Timepoints) matrix
- Ensure player names in metadata match CSV filenames exactly
- Add `.csv` extension if missing

**No Valid Windows Created**:
- Check if EEG files are long enough (need > window_size samples)
- Reduce `data.window_size` or `data.stride`
- Verify EEG CSV files are not empty or corrupted

**Training Not Converging**:
- Start with smaller model (d_model=128, num_layers=4)
- Lower learning rate (try 5e-5)
- Check data preprocessing (bandpass filter, normalization)
- Verify labels are correct in metadata

**Channel Mismatch**:
- Count actual channels in CSV files
- Update `model.in_channels` to match
- Ensure all EEG files have same number of channels

**scipy Import Error** (for filtering):
```bash
pip install scipy
```

## Notes

- The codebase is in Chinese (Traditional) but code/comments are mixed English/Chinese
- All training follows HuggingFace Transformers best practices
- Wandb is the primary experiment tracking tool (TensorBoard deprecated)
- Early stopping has been removed - training runs full epoch count
- Windows environment: Use standard Python paths, not Unix-style
