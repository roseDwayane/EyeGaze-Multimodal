# ViT Image Classification Training

This experiment trains a Vision Transformer (ViT) model for classifying eye-gaze images into three categories: Single, Competition, and Cooperation.

## Data Structure

The training uses dual-image inputs where `player1` and `player2` images are concatenated horizontally before being fed to the ViT model.

**Classes:**
- Single: Single-player mode
- Competition: Competitive mode
- Cooperation: Cooperative mode

## Files

```
Experiments/
├── configs/
│   └── vit_single_vs_competition.yaml  # Training configuration
└── scripts/
    └── train_vit.py                     # Training script

Models/
└── backbones/
    └── vit.py                           # ViT model implementation

metrics/
└── classification.py                    # Evaluation metrics
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python Experiments/scripts/train_vit.py --config Experiments/configs/vit_single_vs_competition.yaml
```

### 3. Monitor Training

TensorBoard logs are saved to `Experiments/outputs/vit_dual_classification/logs/`

```bash
tensorboard --logdir Experiments/outputs/vit_dual_classification/logs/
```

## Configuration

Edit `Experiments/configs/vit_single_vs_competition.yaml` to customize:

- **Model settings**: model architecture, pretrained weights, backbone freezing
- **Data settings**: train/test split ratio, image preprocessing
- **Training settings**: learning rate, batch size, epochs, optimizer
- **Augmentation**: data augmentation options
- **Paths**: output directories

## Key Features

### Following Hugging Face Best Practices

The training script follows Hugging Face Transformers guidelines:

1. **Dataset Loading**: Uses `datasets.load_dataset()` for JSON data
2. **Train/Test Split**: Uses `dataset.train_test_split()` with stratification
3. **Trainer API**: Uses Hugging Face `Trainer` for training loop
4. **Model**: Uses `ViTForImageClassification` from transformers
5. **Metrics**: Compatible with Trainer's `compute_metrics` callback

### Dual-Image Processing

Images are concatenated horizontally (default) or vertically:
- Player1 image + Player2 image → Single concatenated image
- Concatenation mode configurable in YAML config

### Training Features

- Early stopping based on F1 score
- Model checkpointing (saves best model)
- TensorBoard logging
- Stratified train/test split
- Per-class metrics reporting
- Confusion matrix visualization

## Training Output

Results are saved to:
- **Model checkpoints**: `Experiments/outputs/vit_dual_classification/checkpoint-*/`
- **Best model**: `Experiments/outputs/vit_dual_classification/`
- **Logs**: `Experiments/outputs/vit_dual_classification/logs/`
- **Results**: `Experiments/outputs/vit_dual_classification/results/test_results.txt`

## Evaluation Metrics

The training evaluates:
- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)
- F1 Score (macro & weighted)
- Confusion Matrix
- Per-class metrics

## Customization

### Change Model Size

Edit `vit_single_vs_competition.yaml`:

```yaml
model:
  model_name: "google/vit-large-patch16-224"  # or vit-huge
```

### Adjust Image Concatenation

```yaml
model:
  concat_mode: "vertical"  # or "horizontal"
```

### Fine-tune Hyperparameters

```yaml
training:
  learning_rate: 5.0e-5
  per_device_train_batch_size: 16
  num_train_epochs: 20
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Enable gradient checkpointing
- Use a smaller model (vit-small)

### Low Accuracy
- Increase training epochs
- Adjust learning rate
- Enable data augmentation
- Check class balance in dataset

### Image Loading Errors
- Verify `image_base_path` in config
- Ensure image files exist with `.jpg` extension
- Check that player1/player2 names match metadata
