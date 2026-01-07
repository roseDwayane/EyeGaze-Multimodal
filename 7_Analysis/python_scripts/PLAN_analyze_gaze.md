# Implementation Plan: analyze_gaze.py

## Overview

This plan outlines the implementation of a comprehensive analysis script for the Gaze classification project. The script will analyze trained Early Fusion and Late Fusion ViT models.

---

## Part 0: Setup & Path Management

### 0.1 Command Line Arguments (`argparse`)

```python
--config        # Path to YAML config file (e.g., 4_Experiments/configs/gaze_earlyfusion.yaml)
--checkpoint    # Path to checkpoint file (e.g., 4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt)
--model_type    # "early" or "late" (determines which model class to load)
--fusion_mode   # Override config fusion_mode if needed (concat/add/subtract/multiply/full)
--device        # cuda or cpu (default: auto-detect)
--exp_name      # Experiment name for output folders (default: auto-generate from checkpoint path)
```

### 0.2 Project Root & Import Setup

```python
# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]  # 7_Analysis/python_scripts -> EyeGaze-Multimodal_new
sys.path.insert(0, str(project_root))

# Required imports from project
from 3_Models.backbones.early_fusion_vit import EarlyFusionViT
from 3_Models.backbones.late_fusion_vit import LateFusionViT
from 1_Data.datasets.gaze_pair_dataset import GazePairDataset, create_train_val_datasets, custom_collate_fn
```

**Issue Identified:** Python doesn't allow numeric-prefixed imports. We need to handle this:
- Option A: Rename folders (e.g., `models` instead of `3_Models`)
- Option B: Use `importlib` with dynamic loading
- **Recommended:** Use `importlib.util` to load modules with numeric prefixes

### 0.3 Output Directory Structure

```
7_Analysis/
├── raw_result/
│   └── {exp_name}/              # e.g., "earlyfusion_concat"
│       ├── metrics.csv
│       ├── conf_mat.npy
│       ├── roc_data.npz         # Contains fpr, tpr for each class
│       ├── features.npy         # CLS token features (N, D)
│       ├── tsne_coords.npy      # t-SNE projections (N, 2)
│       ├── labels.npy           # Ground truth labels
│       ├── predictions.npy      # Model predictions
│       └── pair_stats.csv       # Per-pair accuracy
│
manuscript/
├── figures/
│   └── {exp_name}/
│       ├── fig_conf_mat.pdf
│       ├── fig_roc_curves.pdf
│       ├── fig_tsne.pdf
│       ├── fig_attention_viz.png
│       ├── fig_pair_accuracy.png
│       └── fig_mechanism_analysis.pdf
│
└── tables/
    └── table_performance_{exp_name}.csv
```

---

## Part 1: Quantitative Analysis

### 1.1 Metric Calculation

**Implementation:**
1. Run inference on entire validation set
2. Collect predictions and ground truth labels
3. Calculate metrics using `sklearn.metrics`:
   - `accuracy_score`
   - `precision_score(average='macro')` and `precision_score(average='weighted')`
   - `recall_score(average='macro')` and `recall_score(average='weighted')`
   - `f1_score(average='macro')` and `f1_score(average='weighted')`

**Output Files:**
- `7_Analysis/raw_result/{exp_name}/metrics.csv` - Full metrics table
- `manuscript/tables/table_performance_{exp_name}.csv` - Formatted for thesis

**CSV Format:**
```csv
metric,value
accuracy,0.8523
precision_macro,0.8412
precision_weighted,0.8567
recall_macro,0.8234
recall_weighted,0.8523
f1_macro,0.8321
f1_weighted,0.8544
```

### 1.2 Confusion Matrix

**Implementation:**
1. Use `sklearn.metrics.confusion_matrix`
2. Normalize option for percentage display

**Output Files:**
- `7_Analysis/raw_result/{exp_name}/conf_mat.npy` - Raw numpy array (3x3)
- `manuscript/figures/{exp_name}/fig_conf_mat.pdf` - Heatmap visualization

**Plot Specifications:**
- Use `seaborn.heatmap` with `annot=True`
- Color map: Blues or viridis
- Class labels: ["Single", "Competition", "Cooperation"]
- Include both raw counts and percentages
- Figure size: 8x6 inches, 300 DPI
- Title: "Confusion Matrix - {model_type} Fusion ({fusion_mode})"

### 1.3 ROC Curves (Multi-class)

**Implementation:**
1. Get probability scores using `softmax(logits)`
2. For each class, compute One-vs-Rest ROC:
   - `sklearn.metrics.roc_curve(y_true_binary, y_prob[:, class_idx])`
   - `sklearn.metrics.auc(fpr, tpr)`
3. Compute micro/macro average ROC

**Output Files:**
- `7_Analysis/raw_result/{exp_name}/roc_data.npz` - Contains:
  - `fpr_class0`, `tpr_class0`, `auc_class0`
  - `fpr_class1`, `tpr_class1`, `auc_class1`
  - `fpr_class2`, `tpr_class2`, `auc_class2`
  - `fpr_micro`, `tpr_micro`, `auc_micro`
- `manuscript/figures/{exp_name}/fig_roc_curves.pdf`

**Plot Specifications:**
- Three class curves + micro average
- Include diagonal reference line
- Legend with AUC values
- Figure size: 8x6 inches, 300 DPI

---

## Part 2: Qualitative Analysis (Visualization)

### 2.1 Image Denormalization

**Critical:** Images were normalized with ImageNet stats during training:
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

**Denormalization function:**
```python
def denormalize(img_tensor, mean, std):
    """Denormalize tensor image for visualization."""
    img = img_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)
```

### 2.2 Attention Visualization

#### 2.2.1 Attention Rollout

**Concept:** Multiply attention matrices across all layers to get final attention map from CLS token to patches.

**Implementation for ViT:**
```python
def attention_rollout(model, img_a, img_b, fusion_mode):
    """
    For Early Fusion: fused_img -> ViT -> attention maps
    For Late Fusion: process each stream separately
    """
    # Register hooks on attention layers
    attention_weights = []

    def hook_fn(module, input, output):
        # ViT attention output shape: (B, num_heads, N, N)
        attention_weights.append(output.detach())

    # Register hooks on all attention layers
    # In timm ViT: model.blocks[i].attn.qkv or model.blocks[i].attn

    # Forward pass
    with torch.no_grad():
        _ = model(img_a, img_b)

    # Compute rollout
    # Multiply attention matrices, accounting for residual connections
    result = torch.eye(attention_weights[0].shape[-1])
    for attn in attention_weights:
        attn_avg = attn.mean(dim=1)  # Average over heads
        attn_with_residual = (attn_avg + torch.eye(attn_avg.shape[-1])) / 2
        result = torch.matmul(attn_with_residual, result)

    # Extract CLS token attention to patches
    cls_attention = result[0, 0, 1:]  # Skip CLS token itself

    # Reshape to 2D spatial map (14x14 for ViT-B/16 with 224x224 input)
    attention_map = cls_attention.reshape(14, 14)

    return attention_map
```

#### 2.2.2 Grad-CAM for ViT

**Implementation:**
```python
def grad_cam_vit(model, img_a, img_b, target_class):
    """
    Grad-CAM for Vision Transformer.
    Target the last attention layer's value projection or the last block's output.
    """
    # For ViT: typically target the last block's output (before layer norm)
    target_layer = model.backbone.blocks[-1]

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    model.zero_grad()
    logits = model(img_a, img_b)

    # Backward for target class
    one_hot = torch.zeros_like(logits)
    one_hot[0, target_class] = 1
    logits.backward(gradient=one_hot, retain_graph=True)

    # Compute Grad-CAM
    grad = gradients[0][:, 1:, :]  # Remove CLS token
    act = activations[0][:, 1:, :]  # Remove CLS token

    weights = grad.mean(dim=1, keepdim=True)  # Global average pooling over patches
    cam = (weights * act).sum(dim=-1)  # Weighted sum
    cam = F.relu(cam)  # ReLU
    cam = cam.reshape(14, 14)  # Reshape to spatial

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()

    return cam
```

#### 2.2.3 Visualization Comparison

**For Early Fusion (Concat mode):**
- Show original img_a and img_b side by side
- Show attention overlaid on the fused 6-channel representation
- Requires special handling: visualize attention on both image regions

**For Late Fusion:**
- Show attention maps for Stream 1 (img_a) and Stream 2 (img_b) separately
- Compare how model attends to each player's gaze

**Sample Selection:**
- "Easy" samples: High confidence correct predictions
- "Hard" samples: Misclassified or low confidence predictions

**Output Files:**
- `manuscript/figures/{exp_name}/fig_attention_viz.png`

**Plot Specifications:**
- Multi-panel figure (e.g., 2x4 or 3x4 grid)
- Original images + attention overlays
- Alpha blending for overlay
- Colorbar for attention intensity
- Labels for sample class, prediction, confidence

### 2.3 Feature Space Analysis (t-SNE)

**Implementation:**
```python
def extract_features(model, dataloader, device, model_type):
    """Extract CLS token features for all samples."""
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for img_a, img_b, labels in tqdm(dataloader):
            img_a, img_b = img_a.to(device), img_b.to(device)

            if model_type == 'early':
                feat = model.get_features(img_a, img_b)  # (B, 768)
            else:  # late
                feat_dict = model.get_features(img_a, img_b)
                feat = feat_dict['fused']  # Use fused features

            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)  # (N, D)
    labels = np.concatenate(labels_list, axis=0)  # (N,)

    return features, labels
```

**t-SNE Projection:**
```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
tsne_coords = tsne.fit_transform(features)  # (N, 2)
```

**Output Files:**
- `7_Analysis/raw_result/{exp_name}/features.npy` - (N, D) feature matrix
- `7_Analysis/raw_result/{exp_name}/tsne_coords.npy` - (N, 2) t-SNE coordinates
- `7_Analysis/raw_result/{exp_name}/labels.npy` - (N,) ground truth labels
- `manuscript/figures/{exp_name}/fig_tsne.pdf`

**Plot Specifications:**
- Scatter plot with class-colored points
- Class names: Single (blue), Competition (red), Cooperation (green)
- Legend with class names
- Optional: highlight misclassified samples
- Figure size: 8x8 inches, 300 DPI

---

## Part 3: Error & Mechanism Analysis

### 3.1 Pair-wise Performance Analysis

**Implementation:**
1. Modify dataset to return `pair_id` in metadata
2. Group predictions by `pair_id`
3. Calculate per-pair accuracy

**Data Collection:**
```python
def collect_pair_results(model, dataloader_with_metadata, device):
    """Collect results with pair information."""
    results = []

    for img_a, img_b, labels, metadata in dataloader_with_metadata:
        # ... inference ...
        for i in range(len(labels)):
            results.append({
                'pair_id': metadata[i]['pair'],
                'true_label': labels[i].item(),
                'pred_label': preds[i].item(),
                'correct': (labels[i] == preds[i]).item()
            })

    return pd.DataFrame(results)
```

**Per-pair Statistics:**
```python
pair_stats = results_df.groupby('pair_id').agg({
    'correct': ['sum', 'count', 'mean']
}).reset_index()
pair_stats.columns = ['pair_id', 'correct_count', 'total_count', 'accuracy']
```

**Hard Pairs Identification:**
- Pairs with accuracy < mean - 1.5 * std
- Or bottom 20% of pairs

**Output Files:**
- `7_Analysis/raw_result/{exp_name}/pair_stats.csv`
- `manuscript/figures/{exp_name}/fig_pair_accuracy.png`

**Plot Specifications:**
- Bar chart of accuracy by Pair ID
- Horizontal line for overall accuracy
- Color code hard pairs differently
- Figure size: 12x6 inches, 300 DPI

### 3.2 Interaction Mechanism Validation

#### 3.2.1 Spatial Sensitivity (Early Fusion)

**Concept:** Analyze relationship between gaze spatial overlap and classification accuracy.

**Implementation:**
```python
def compute_gaze_distance(img_a, img_b):
    """
    Compute distance between gaze centers of two heatmaps.
    Assumes heatmap values indicate gaze intensity.
    """
    # Convert to grayscale intensity
    heat_a = img_a.mean(dim=0)  # (H, W)
    heat_b = img_b.mean(dim=0)  # (H, W)

    # Find center of mass
    def center_of_mass(heatmap):
        total = heatmap.sum()
        if total == 0:
            return 112, 112  # Center as default

        y_coords = torch.arange(224).float().unsqueeze(1)
        x_coords = torch.arange(224).float().unsqueeze(0)

        cy = (heatmap * y_coords).sum() / total
        cx = (heatmap * x_coords).sum() / total
        return cy.item(), cx.item()

    cy_a, cx_a = center_of_mass(heat_a)
    cy_b, cx_b = center_of_mass(heat_b)

    distance = np.sqrt((cy_a - cy_b)**2 + (cx_a - cx_b)**2)
    return distance
```

**Analysis:**
- Bin samples by distance
- Plot accuracy vs. distance bins
- Hypothesis: Cooperation may have smaller distance (shared attention)

#### 3.2.2 Feature Correlation (Late Fusion)

**Concept:** Analyze cosine similarity between two streams' CLS tokens.

**Implementation:**
```python
def compute_feature_correlation(model, dataloader, device):
    """Compute cosine similarity between stream features."""
    similarities = []
    labels_list = []

    with torch.no_grad():
        for img_a, img_b, labels in dataloader:
            img_a, img_b = img_a.to(device), img_b.to(device)

            feat_dict = model.get_features(img_a, img_b)
            cls1 = F.normalize(feat_dict['cls1'], dim=-1)
            cls2 = F.normalize(feat_dict['cls2'], dim=-1)

            cos_sim = (cls1 * cls2).sum(dim=-1)  # (B,)

            similarities.extend(cos_sim.cpu().numpy())
            labels_list.extend(labels.numpy())

    return np.array(similarities), np.array(labels_list)
```

**Visualization:**
- Violin plot of cosine similarity by class
- Hypothesis: Cooperation has higher similarity, Competition has lower

**Output Files:**
- `manuscript/figures/{exp_name}/fig_mechanism_analysis.pdf`

---

## Class Design: `GazeAnalyzer`

```python
class GazeAnalyzer:
    """Comprehensive analyzer for Gaze classification models."""

    def __init__(self, config_path, checkpoint_path, model_type,
                 fusion_mode=None, device=None, exp_name=None):
        """Initialize analyzer with model and data."""
        self.config = self._load_config(config_path)
        self.model_type = model_type
        self.fusion_mode = fusion_mode or self.config['model']['fusion_mode']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.exp_name = exp_name or self._generate_exp_name()

        # Setup paths
        self._setup_output_dirs()

        # Load model and data
        self.model = self._load_model(checkpoint_path)
        self.val_dataset, self.val_loader = self._load_data()

        # Storage for results
        self.predictions = None
        self.labels = None
        self.probabilities = None
        self.features = None

    # --- Part 0: Setup ---
    def _load_config(self, path): ...
    def _load_model(self, checkpoint_path): ...
    def _load_data(self): ...
    def _setup_output_dirs(self): ...

    # --- Part 1: Quantitative Analysis ---
    def run_inference(self): ...
    def compute_metrics(self): ...
    def compute_confusion_matrix(self): ...
    def compute_roc_curves(self): ...
    def save_metrics(self): ...
    def plot_confusion_matrix(self): ...
    def plot_roc_curves(self): ...

    # --- Part 2: Qualitative Analysis ---
    def extract_features(self): ...
    def compute_tsne(self): ...
    def plot_tsne(self): ...
    def attention_rollout(self, sample_idx): ...
    def grad_cam(self, sample_idx, target_class): ...
    def visualize_attention(self, sample_indices): ...

    # --- Part 3: Error Analysis ---
    def analyze_pair_performance(self): ...
    def identify_hard_samples(self): ...
    def plot_pair_accuracy(self): ...
    def spatial_sensitivity_analysis(self): ...  # Early Fusion only
    def feature_correlation_analysis(self): ...  # Late Fusion only
    def plot_mechanism_analysis(self): ...

    # --- Main Entry Point ---
    def run_full_analysis(self):
        """Run all analyses and generate all outputs."""
        print("Part 1: Quantitative Analysis")
        self.run_inference()
        self.compute_metrics()
        self.compute_confusion_matrix()
        self.compute_roc_curves()

        print("Part 2: Qualitative Analysis")
        self.extract_features()
        self.compute_tsne()
        hard_samples, easy_samples = self.identify_hard_samples()
        self.visualize_attention(hard_samples + easy_samples)

        print("Part 3: Error Analysis")
        self.analyze_pair_performance()
        if self.model_type == 'early':
            self.spatial_sensitivity_analysis()
        else:
            self.feature_correlation_analysis()

        print("Saving all outputs...")
        self.save_all_outputs()

        print(f"Analysis complete! Results saved to:")
        print(f"  Raw data: 7_Analysis/raw_result/{self.exp_name}/")
        print(f"  Figures: manuscript/figures/{self.exp_name}/")
        print(f"  Tables: manuscript/tables/")
```

---

## Execution Script (Main)

```python
def main():
    parser = argparse.ArgumentParser(description='Gaze Model Analysis')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', required=True, choices=['early', 'late'])
    parser.add_argument('--fusion_mode', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--exp_name', default=None)
    args = parser.parse_args()

    analyzer = GazeAnalyzer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        fusion_mode=args.fusion_mode,
        device=args.device,
        exp_name=args.exp_name
    )

    analyzer.run_full_analysis()

if __name__ == '__main__':
    main()
```

---

## Dependencies

```python
# Standard library
import os
import sys
import json
import argparse
from pathlib import Path

# Scientific computing
import numpy as np
import pandas as pd

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Image processing
from PIL import Image
import torchvision.transforms as transforms

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.manifold import TSNE

# Progress bar
from tqdm import tqdm

# Config
import yaml
```

---

## Key Implementation Notes

### 1. Import Handling for Numeric-Prefixed Folders

```python
import importlib.util

def load_module(module_path, module_name):
    """Load a Python module from path (handles numeric prefixes)."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Example usage:
early_fusion_vit = load_module(
    project_root / "3_Models/backbones/early_fusion_vit.py",
    "early_fusion_vit"
)
EarlyFusionViT = early_fusion_vit.EarlyFusionViT
```

### 2. DataLoader with Metadata

To get `pair_id` for pair-wise analysis, we need to use `return_metadata=True`:

```python
# Modify dataset creation
val_dataset = GazePairDataset(
    metadata=val_metadata,
    image_base_path=...,
    label2id=...,
    transform=val_transform,
    return_metadata=True  # Enable metadata return
)

# Custom collate for metadata
def collate_with_metadata(batch):
    img_a = torch.stack([item[0] for item in batch])
    img_b = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    metadata = [item[3] for item in batch]
    return img_a, img_b, labels, metadata
```

### 3. ViT Attention Hook Placement

For `timm` ViT models, attention is in:
- `model.backbone.blocks[i].attn.attn_drop` (after softmax)
- Need to hook `attn` module and capture attention weights

```python
# In timm ViT, we need to modify forward or use specific hooks
# The attention output can be captured via:
model.backbone.blocks[i].attn.fused_attn = False  # Disable fused attention
# Then attention weights are computed explicitly and can be hooked
```

### 4. Thesis Figure Mapping

| Script Output | Thesis Section |
|---------------|----------------|
| `fig_conf_mat.pdf` | Results - Performance Metrics (Fig. X) |
| `fig_roc_curves.pdf` | Results - Classification Performance (Fig. X) |
| `fig_tsne.pdf` | Results - Feature Analysis (Fig. X) |
| `fig_attention_viz.png` | Results - Attention Visualization (Fig. X) |
| `fig_pair_accuracy.png` | Discussion - Per-pair Analysis (Fig. X) |
| `fig_mechanism_analysis.pdf` | Discussion - Mechanism Validation (Fig. X) |
| `table_performance_{exp}.csv` | Results - Table X |

---

## Design Decisions (User Confirmed)

1. **Statistical Tests:** ✅ Include t-test, ANOVA with p-values and effect sizes for pair-wise and mechanism analyses

2. **Export Format:** ✅ Use `.csv` files for all raw data exports (MATLAB-friendly and human-readable)

3. **Multi-checkpoint Comparison:** ✅ Support loading multiple checkpoints and generating side-by-side comparison figures

4. **Learning Curves:** ✅ Parse wandb logs or checkpoint history to plot training/validation curves over epochs

---

## Additional Implementation Details (Based on User Decisions)

### Statistical Tests Implementation

```python
from scipy import stats

def paired_analysis_stats(pair_stats_df):
    """Statistical analysis for pair-wise performance."""
    # ANOVA: Compare performance across pairs
    groups = [group['accuracy'].values for name, group in pair_stats_df.groupby('pair_id')]
    f_stat, p_value_anova = stats.f_oneway(*groups)

    # Effect size (eta-squared)
    ss_between = sum(len(g) * (np.mean(g) - pair_stats_df['accuracy'].mean())**2 for g in groups)
    ss_total = sum((pair_stats_df['accuracy'] - pair_stats_df['accuracy'].mean())**2)
    eta_squared = ss_between / ss_total

    return {
        'anova_f': f_stat,
        'anova_p': p_value_anova,
        'eta_squared': eta_squared
    }

def mechanism_stats(similarities, labels, class_names):
    """Statistical tests for mechanism validation."""
    # Group by class
    class_groups = {name: similarities[labels == i] for i, name in enumerate(class_names)}

    # Pairwise t-tests with Bonferroni correction
    results = []
    comparisons = [(0, 1), (0, 2), (1, 2)]  # Single vs Comp, Single vs Coop, Comp vs Coop
    alpha = 0.05 / len(comparisons)  # Bonferroni correction

    for i, j in comparisons:
        t_stat, p_value = stats.ttest_ind(
            class_groups[class_names[i]],
            class_groups[class_names[j]]
        )
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(class_groups[class_names[i]]) +
                              np.var(class_groups[class_names[j]])) / 2)
        cohen_d = (np.mean(class_groups[class_names[i]]) -
                   np.mean(class_groups[class_names[j]])) / pooled_std

        results.append({
            'comparison': f"{class_names[i]} vs {class_names[j]}",
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohen_d': cohen_d
        })

    return pd.DataFrame(results)
```

**Output:** `7_Analysis/raw_result/{exp_name}/statistical_tests.csv`

---

### CSV Export Format (Updated)

All raw data will be exported as CSV for MATLAB compatibility:

```python
# Confusion matrix as CSV
conf_mat_df = pd.DataFrame(
    conf_mat,
    index=['True_Single', 'True_Competition', 'True_Cooperation'],
    columns=['Pred_Single', 'Pred_Competition', 'Pred_Cooperation']
)
conf_mat_df.to_csv(output_path / 'conf_mat.csv')

# ROC data as CSV
roc_df = pd.DataFrame({
    'class': [],
    'fpr': [],
    'tpr': [],
    'threshold': []
})
# Flatten and save all classes
for i, name in enumerate(class_names):
    class_roc = pd.DataFrame({
        'class': name,
        'fpr': fpr[i],
        'tpr': tpr[i],
        'threshold': thresholds[i]
    })
    roc_df = pd.concat([roc_df, class_roc])
roc_df.to_csv(output_path / 'roc_data.csv', index=False)

# Features as CSV (with labels)
features_df = pd.DataFrame(features)
features_df['label'] = labels
features_df['pair_id'] = pair_ids
features_df.to_csv(output_path / 'features.csv', index=False)

# t-SNE coordinates as CSV
tsne_df = pd.DataFrame({
    'tsne_1': tsne_coords[:, 0],
    'tsne_2': tsne_coords[:, 1],
    'label': labels,
    'label_name': [class_names[l] for l in labels]
})
tsne_df.to_csv(output_path / 'tsne_coords.csv', index=False)
```

---

### Multi-Checkpoint Comparison Mode

**New CLI Arguments:**
```python
parser.add_argument('--compare', action='store_true',
                    help='Enable multi-checkpoint comparison mode')
parser.add_argument('--checkpoints', nargs='+',
                    help='List of checkpoint paths for comparison')
parser.add_argument('--labels', nargs='+',
                    help='Labels for each checkpoint (e.g., "concat" "multiply")')
```

**Comparison Outputs:**
```
manuscript/figures/comparison/
├── fig_compare_conf_mat.pdf      # Side-by-side confusion matrices
├── fig_compare_roc.pdf           # Overlaid ROC curves
├── fig_compare_tsne.pdf          # Multi-panel t-SNE plots
├── fig_compare_attention.png     # Attention comparison grid
└── fig_compare_metrics.pdf       # Bar chart comparing metrics

manuscript/tables/
└── table_comparison.csv          # All metrics for all models in one table
```

**Comparison Implementation:**
```python
class MultiModelAnalyzer:
    """Compare multiple trained models."""

    def __init__(self, checkpoint_configs: List[Dict]):
        """
        checkpoint_configs: List of {
            'checkpoint': path,
            'config': path,
            'model_type': 'early' or 'late',
            'label': 'Display name'
        }
        """
        self.analyzers = []
        for cfg in checkpoint_configs:
            analyzer = GazeAnalyzer(
                config_path=cfg['config'],
                checkpoint_path=cfg['checkpoint'],
                model_type=cfg['model_type'],
                exp_name=cfg['label']
            )
            self.analyzers.append(analyzer)

    def run_all_inferences(self):
        """Run inference for all models."""
        for analyzer in self.analyzers:
            analyzer.run_inference()

    def compare_metrics(self):
        """Generate comparison table."""
        all_metrics = []
        for analyzer in self.analyzers:
            metrics = analyzer.compute_metrics()
            metrics['model'] = analyzer.exp_name
            all_metrics.append(metrics)
        return pd.DataFrame(all_metrics)

    def plot_comparison_confusion_matrices(self):
        """Side-by-side confusion matrices."""
        n_models = len(self.analyzers)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

        for idx, analyzer in enumerate(self.analyzers):
            ax = axes[idx] if n_models > 1 else axes
            cm = analyzer.compute_confusion_matrix()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_title(f'{analyzer.exp_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        plt.savefig('manuscript/figures/comparison/fig_compare_conf_mat.pdf', dpi=300)

    def plot_overlaid_roc(self):
        """Overlay ROC curves from all models."""
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = plt.cm.tab10.colors
        for idx, analyzer in enumerate(self.analyzers):
            fpr, tpr, auc_score = analyzer.compute_roc_curves()
            ax.plot(fpr['micro'], tpr['micro'],
                    label=f'{analyzer.exp_name} (AUC={auc_score["micro"]:.3f})',
                    color=colors[idx], linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison (Micro-Average)')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('manuscript/figures/comparison/fig_compare_roc.pdf', dpi=300)
```

---

### Learning Curves Implementation

**Source Options:**
1. **Wandb API** (if enabled during training)
2. **Checkpoint metadata** (if saved during training)

**Wandb Approach:**
```python
import wandb

def fetch_wandb_history(project_name, run_name):
    """Fetch training history from wandb."""
    api = wandb.Api()
    runs = api.runs(project_name)

    for run in runs:
        if run.name == run_name:
            history = run.history()
            return history[['epoch', 'train_loss', 'val_loss',
                           'train_acc', 'val_acc', 'val_f1']]

    raise ValueError(f"Run {run_name} not found in project {project_name}")

def plot_learning_curves(history_df, save_path):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(history_df['epoch'], history_df['train_acc'], label='Train')
    axes[1].plot(history_df['epoch'], history_df['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()

    # F1 curve
    axes[2].plot(history_df['epoch'], history_df['val_f1'], label='Val F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Curve')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
```

**Checkpoint-based Approach (Fallback):**
```python
def extract_history_from_checkpoints(checkpoint_dir):
    """Extract metrics from periodic checkpoints."""
    checkpoint_files = sorted(Path(checkpoint_dir).glob('checkpoint_epoch_*.pt'))

    history = []
    for ckpt_path in checkpoint_files:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt['epoch']
        # Assumes metrics were saved in checkpoint
        if 'metrics' in ckpt:
            history.append({
                'epoch': epoch,
                **ckpt['metrics']
            })

    return pd.DataFrame(history)
```

**Output:**
- `manuscript/figures/{exp_name}/fig_learning_curves.pdf`
- `7_Analysis/raw_result/{exp_name}/training_history.csv`

---

## Estimated File Structure Output (Updated)

```
7_Analysis/
├── python_scripts/
│   └── analyze_gaze.py          # Main analysis script (~1000-1200 lines)
│
├── raw_result/
│   ├── earlyfusion_concat/
│   │   ├── metrics.csv                  # Performance metrics
│   │   ├── conf_mat.csv                 # Confusion matrix (CSV format)
│   │   ├── roc_data.csv                 # ROC curve data (fpr, tpr per class)
│   │   ├── features.csv                 # CLS token features (N, D+metadata)
│   │   ├── tsne_coords.csv              # t-SNE 2D projections with labels
│   │   ├── predictions.csv              # All predictions with ground truth
│   │   ├── pair_stats.csv               # Per-pair accuracy statistics
│   │   ├── statistical_tests.csv        # t-test, ANOVA results with p-values
│   │   └── training_history.csv         # Learning curve data (from wandb/ckpts)
│   │
│   └── latefusion_full/
│       └── ... (same structure)

manuscript/
├── figures/
│   ├── earlyfusion_concat/
│   │   ├── fig_conf_mat.pdf             # Confusion matrix heatmap
│   │   ├── fig_roc_curves.pdf           # Multi-class ROC curves
│   │   ├── fig_tsne.pdf                 # Feature space visualization
│   │   ├── fig_attention_viz.png        # Attention map comparison
│   │   ├── fig_pair_accuracy.png        # Per-pair accuracy bar chart
│   │   ├── fig_mechanism_analysis.pdf   # Spatial/correlation analysis
│   │   └── fig_learning_curves.pdf      # Training/validation curves
│   │
│   ├── latefusion_full/
│   │   └── ... (same structure)
│   │
│   └── comparison/                       # Multi-model comparison outputs
│       ├── fig_compare_conf_mat.pdf
│       ├── fig_compare_roc.pdf
│       ├── fig_compare_tsne.pdf
│       ├── fig_compare_attention.png
│       └── fig_compare_metrics.pdf
│
└── tables/
    ├── table_performance_earlyfusion_concat.csv
    ├── table_performance_latefusion_full.csv
    └── table_comparison.csv              # Side-by-side all models
```

---

## Example Usage Commands

### Single Model Analysis
```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_earlyfusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
    --model_type early \
    --exp_name earlyfusion_concat
```

### Multi-Model Comparison
```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --compare \
    --checkpoints \
        4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
        4_Experiments/runs/gaze_latefusion/full/best_model.pt \
    --configs \
        4_Experiments/configs/gaze_earlyfusion.yaml \
        4_Experiments/configs/gaze_latefusion.yaml \
    --model_types early late \
    --labels "Early-Concat" "Late-Full"
```

### With Wandb Learning Curves
```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_earlyfusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
    --model_type early \
    --wandb_project Multimodal_Gaze \
    --wandb_run early_fusion_vit
```
