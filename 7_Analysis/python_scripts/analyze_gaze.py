"""
Comprehensive Gaze Model Analysis Script

This script performs quantitative and qualitative analysis of trained
Early Fusion and Late Fusion ViT models for gaze-based interaction classification.

Output locations:
- Raw data (CSV): 7_Analysis/raw_result/{exp_name}/
- Publication figures (PDF/PNG): 7_Analysis/figures/{exp_name}/
- Tables: 7_Analysis/tables/

Author: Generated for Gaze-EEG Multimodal Project
"""

# =============================================================================
# Part 0: Setup & Path Management
# =============================================================================

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # 7_Analysis/python_scripts -> EyeGaze-Multimodal_new
sys.path.insert(0, str(PROJECT_ROOT))

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
import matplotlib.patches as mpatches
import seaborn as sns

# Machine learning metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# Statistical tests
from scipy import stats

# Progress bar
from tqdm import tqdm

# Config
import yaml

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300


def load_module_from_path(module_path: Path, module_name: str):
    """
    Load a Python module from a file path (handles numeric prefixes like '3_Models').

    Args:
        module_path: Path to the .py file
        module_name: Name to give the module

    Returns:
        Loaded module object
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load project modules dynamically
early_fusion_vit_module = load_module_from_path(
    PROJECT_ROOT / "3_Models" / "backbones" / "early_fusion_vit.py",
    "early_fusion_vit"
)
late_fusion_vit_module = load_module_from_path(
    PROJECT_ROOT / "3_Models" / "backbones" / "late_fusion_vit.py",
    "late_fusion_vit"
)
gaze_dataset_module = load_module_from_path(
    PROJECT_ROOT / "1_Data" / "datasets" / "gaze_pair_dataset.py",
    "gaze_pair_dataset"
)

EarlyFusionViT = early_fusion_vit_module.EarlyFusionViT
LateFusionViT = late_fusion_vit_module.LateFusionViT
GazePairDataset = gaze_dataset_module.GazePairDataset
create_train_val_datasets = gaze_dataset_module.create_train_val_datasets


# =============================================================================
# Constants
# =============================================================================

CLASS_NAMES = ["Single", "Competition", "Cooperation"]
CLASS_COLORS = {
    "Single": "#2196F3",       # Blue
    "Competition": "#F44336",  # Red
    "Cooperation": "#4CAF50"   # Green
}

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Utility Functions
# =============================================================================

def denormalize_image(img_tensor: torch.Tensor,
                      mean: List[float] = IMAGENET_MEAN,
                      std: List[float] = IMAGENET_STD) -> torch.Tensor:
    """
    Denormalize tensor image for visualization.

    Args:
        img_tensor: (C, H, W) normalized tensor
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized tensor clamped to [0, 1]
    """
    img = img_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to (H, W, C) numpy array for plotting."""
    img = denormalize_image(tensor)
    return img.permute(1, 2, 0).cpu().numpy()


def create_output_dirs(project_root: Path, exp_name: str) -> Dict[str, Path]:
    """
    Create output directories for analysis results.

    Returns:
        Dict with paths: raw_result, figures, tables
    """
    paths = {
        'raw_result': project_root / "7_Analysis" / "raw_result" / exp_name,
        'figures': project_root / "7_Analysis" / "figures" / exp_name,
        'tables': project_root / "7_Analysis" / "tables",
        'comparison_figures': project_root / "7_Analysis" / "figures" / "comparison"
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


# =============================================================================
# Part 1: Quantitative Analysis
# =============================================================================

class QuantitativeAnalyzer:
    """Handles metric calculation, confusion matrix, and ROC curves."""

    def __init__(self, class_names: List[str] = CLASS_NAMES):
        self.class_names = class_names
        self.n_classes = len(class_names)

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.

        Returns:
            Dict with accuracy, precision, recall, f1 (macro and weighted)
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Per-class metrics
        for i, name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            metrics[f'precision_{name}'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics[f'recall_{name}'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics[f'f1_{name}'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        return metrics

    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(y_true, y_pred, labels=range(self.n_classes))

    def compute_roc_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Compute ROC curve data for multi-class classification.

        Args:
            y_true: Ground truth labels (N,)
            y_prob: Predicted probabilities (N, n_classes)

        Returns:
            Dict with fpr, tpr, thresholds, auc for each class and micro/macro average
        """
        # Binarize labels for one-vs-rest
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))

        roc_data = {}

        # Per-class ROC
        for i, name in enumerate(self.class_names):
            fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[name] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }

        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_data['micro'] = {
            'fpr': fpr_micro,
            'tpr': tpr_micro,
            'auc': auc(fpr_micro, tpr_micro)
        }

        # Macro-average ROC (average of per-class)
        all_fpr = np.unique(np.concatenate([roc_data[name]['fpr'] for name in self.class_names]))
        mean_tpr = np.zeros_like(all_fpr)
        for name in self.class_names:
            mean_tpr += np.interp(all_fpr, roc_data[name]['fpr'], roc_data[name]['tpr'])
        mean_tpr /= self.n_classes
        roc_data['macro'] = {
            'fpr': all_fpr,
            'tpr': mean_tpr,
            'auc': auc(all_fpr, mean_tpr)
        }

        return roc_data

    def save_metrics_csv(self, metrics: Dict[str, float], save_path: Path):
        """Save metrics to CSV."""
        df = pd.DataFrame([metrics]).T
        df.columns = ['value']
        df.index.name = 'metric'
        df.to_csv(save_path)
        print(f"  Saved metrics to {save_path}")

    def save_confusion_matrix_csv(self, cm: np.ndarray, save_path: Path):
        """Save confusion matrix to CSV."""
        df = pd.DataFrame(
            cm,
            index=[f'True_{name}' for name in self.class_names],
            columns=[f'Pred_{name}' for name in self.class_names]
        )
        df.to_csv(save_path)
        print(f"  Saved confusion matrix to {save_path}")

    def save_roc_data_csv(self, roc_data: Dict[str, Any], save_path: Path):
        """Save ROC curve data to CSV."""
        rows = []
        for key, data in roc_data.items():
            if key in ['micro', 'macro'] or key in self.class_names:
                for i in range(len(data['fpr'])):
                    rows.append({
                        'class': key,
                        'fpr': data['fpr'][i],
                        'tpr': data['tpr'][i],
                        'auc': data['auc']
                    })
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"  Saved ROC data to {save_path}")

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Path, title: str = "Confusion Matrix"):
        """
        Plot and save confusion matrix heatmap.
        Generates Fig. X for Results section.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Normalize for percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot with both raw counts and percentages
        sns.heatmap(
            cm_norm, annot=False, fmt='.1f', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax, cbar_kws={'label': 'Percentage (%)'}
        )

        # Add annotations with counts and percentages
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                text = f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)'
                color = 'white' if cm_norm[i, j] > 50 else 'black'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                       color=color, fontsize=10)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved confusion matrix plot to {save_path}")

    def plot_roc_curves(self, roc_data: Dict[str, Any], save_path: Path,
                        title: str = "ROC Curves"):
        """
        Plot multi-class ROC curves.
        Generates Fig. X for Results section.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = list(CLASS_COLORS.values())

        # Plot per-class ROC
        for i, name in enumerate(self.class_names):
            data = roc_data[name]
            ax.plot(data['fpr'], data['tpr'], color=colors[i], lw=2,
                   label=f'{name} (AUC = {data["auc"]:.3f})')

        # Plot micro-average
        micro = roc_data['micro']
        ax.plot(micro['fpr'], micro['tpr'], color='navy', linestyle='--', lw=2,
               label=f'Micro-avg (AUC = {micro["auc"]:.3f})')

        # Plot macro-average
        macro = roc_data['macro']
        ax.plot(macro['fpr'], macro['tpr'], color='darkorange', linestyle=':', lw=2,
               label=f'Macro-avg (AUC = {macro["auc"]:.3f})')

        # Diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved ROC curves to {save_path}")


# =============================================================================
# Part 2: Qualitative Analysis
# =============================================================================

class AttentionAnalyzer:
    """Handles attention visualization: Rollout and Grad-CAM."""

    def __init__(self, model: nn.Module, model_type: str, device: str = 'cuda'):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.attention_weights = []

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights from ViT blocks."""
        self.attention_weights = []
        handles = []

        # Access the backbone's blocks
        if self.model_type == 'early':
            blocks = self.model.backbone.blocks
        else:
            blocks = self.model.backbone.blocks

        def create_hook(block_idx):
            def hook_fn(module, input, output):
                # For timm ViT, we need to capture the attention output
                # This assumes standard attention module structure
                # output shape: (B, N, D) from the attention block
                pass  # We'll use a different approach
            return hook_fn

        # Note: timm ViT doesn't expose attention weights directly
        # We'll use attention rollout approximation instead
        return handles

    def compute_attention_rollout(self, img_a: torch.Tensor, img_b: torch.Tensor,
                                  head_fusion: str = 'mean') -> np.ndarray:
        """
        Compute attention rollout for visualization.

        For ViT, we approximate attention by using gradient-based attribution
        since timm models don't expose attention weights by default.

        Args:
            img_a, img_b: Input images (1, 3, H, W)
            head_fusion: How to fuse attention heads ('mean', 'max', 'min')

        Returns:
            attention_map: (H, W) numpy array
        """
        self.model.eval()

        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)

        # Use gradient-based attribution as approximation
        img_a.requires_grad = True
        img_b.requires_grad = True

        # Forward pass
        if self.model_type == 'early':
            logits = self.model(img_a, img_b)
        else:
            logits = self.model(img_a, img_b)

        # Get predicted class
        pred_class = logits.argmax(dim=1).item()

        # Backward pass for the predicted class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, pred_class] = 1
        logits.backward(gradient=one_hot)

        # Get gradients and compute saliency
        if self.model_type == 'early':
            # For early fusion, combine gradients from both images
            grad_a = img_a.grad.abs()
            grad_b = img_b.grad.abs()
            saliency = (grad_a + grad_b).mean(dim=1)[0]  # Average across channels
        else:
            # For late fusion, return separate saliency maps
            grad_a = img_a.grad.abs().mean(dim=1)[0]
            grad_b = img_b.grad.abs().mean(dim=1)[0]
            saliency = grad_a  # Return first stream by default

        # Normalize
        saliency = saliency.cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        return saliency

    def compute_grad_cam(self, img_a: torch.Tensor, img_b: torch.Tensor,
                         target_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Grad-CAM for ViT model.

        Args:
            img_a, img_b: Input images (1, 3, H, W)
            target_class: Target class for CAM (None = predicted class)

        Returns:
            cam_a, cam_b: Grad-CAM maps for each image (H, W)
        """
        self.model.eval()

        img_a = img_a.clone().to(self.device).requires_grad_(True)
        img_b = img_b.clone().to(self.device).requires_grad_(True)

        activations = []
        gradients = []

        # Get the last transformer block
        if self.model_type == 'early':
            target_layer = self.model.backbone.blocks[-1]
        else:
            target_layer = self.model.backbone.blocks[-1]

        # Register hooks
        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward
            logits = self.model(img_a, img_b)

            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            # Backward
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            # Compute Grad-CAM
            grad = gradients[0][:, 1:, :]  # Remove CLS token, shape: (1, N-1, D)
            act = activations[0][:, 1:, :]  # Remove CLS token

            # Global average pooling over the feature dimension
            weights = grad.mean(dim=2, keepdim=True)  # (1, N-1, 1)
            cam = (weights * act).sum(dim=2)  # (1, N-1)
            cam = F.relu(cam)

            # Reshape to 2D spatial map (14x14 for ViT-B/16 with 224x224 input)
            n_patches = int(np.sqrt(cam.shape[1]))
            cam = cam.reshape(1, n_patches, n_patches)

            # Upsample to image size
            cam = F.interpolate(cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            # Normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        finally:
            handle_fwd.remove()
            handle_bwd.remove()

        return cam, cam  # Return same CAM for both (model processes them together)

    def visualize_attention(self, img_a: torch.Tensor, img_b: torch.Tensor,
                           true_label: int, pred_label: int, confidence: float,
                           save_path: Path, sample_idx: int):
        """
        Create comprehensive attention visualization figure.
        Generates Fig. X for Results section - Attention Visualization.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Convert images for display
        img_a_np = tensor_to_numpy_image(img_a.squeeze(0))
        img_b_np = tensor_to_numpy_image(img_b.squeeze(0))

        # Compute attention maps
        saliency = self.compute_attention_rollout(img_a.clone(), img_b.clone())
        cam_a, cam_b = self.compute_grad_cam(img_a.clone(), img_b.clone())

        # Row 1: Original images and gradient saliency
        axes[0, 0].imshow(img_a_np)
        axes[0, 0].set_title('Player 1 Gaze Heatmap')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img_b_np)
        axes[0, 1].set_title('Player 2 Gaze Heatmap')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(saliency, cmap='hot')
        axes[0, 2].set_title('Gradient Saliency')
        axes[0, 2].axis('off')

        # Row 2: Attention overlays
        axes[1, 0].imshow(img_a_np)
        axes[1, 0].imshow(cam_a, cmap='jet', alpha=0.5)
        axes[1, 0].set_title('Grad-CAM on Player 1')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(img_b_np)
        axes[1, 1].imshow(cam_b, cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Grad-CAM on Player 2')
        axes[1, 1].axis('off')

        # Combined overlay
        combined_img = (img_a_np + img_b_np) / 2
        combined_cam = (cam_a + cam_b) / 2
        axes[1, 2].imshow(combined_img)
        axes[1, 2].imshow(combined_cam, cmap='jet', alpha=0.5)
        axes[1, 2].set_title('Combined Attention')
        axes[1, 2].axis('off')

        # Add title with prediction info
        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_label]
        correct = "Correct" if true_label == pred_label else "Incorrect"
        fig.suptitle(f'Sample {sample_idx}: True={true_name}, Pred={pred_name} ({correct}, Conf={confidence:.2%})',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class FeatureAnalyzer:
    """Handles feature extraction and t-SNE visualization."""

    def __init__(self, model: nn.Module, model_type: str, device: str = 'cuda'):
        self.model = model
        self.model_type = model_type
        self.device = device

    def extract_features(self, dataloader: DataLoader,
                        return_metadata: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[List]]:
        """
        Extract CLS token features from all samples.

        Args:
            dataloader: DataLoader yielding (img_a, img_b, labels) or with metadata
            return_metadata: If True, also return metadata list

        Returns:
            features: (N, D) feature matrix
            labels: (N,) label array
            metadata: Optional list of metadata dicts
        """
        self.model.eval()

        all_features = []
        all_labels = []
        all_metadata = [] if return_metadata else None

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if return_metadata:
                    img_a, img_b, labels, metadata = batch
                    all_metadata.extend(metadata)
                else:
                    img_a, img_b, labels = batch

                img_a = img_a.to(self.device)
                img_b = img_b.to(self.device)

                # Extract features
                if self.model_type == 'early':
                    features = self.model.get_features(img_a, img_b)  # (B, D)
                else:
                    feat_dict = self.model.get_features(img_a, img_b)
                    features = feat_dict['fused']  # Use fused features

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, labels, all_metadata

    def compute_tsne(self, features: np.ndarray,
                     perplexity: int = 30,
                     n_iter: int = 1000,
                     random_state: int = 42) -> np.ndarray:
        """
        Compute t-SNE projection of features.

        Returns:
            tsne_coords: (N, 2) t-SNE coordinates
        """
        print(f"  Computing t-SNE (perplexity={perplexity}, n_iter={n_iter})...")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            max_iter=n_iter,
            random_state=random_state,
            init='pca'
        )

        tsne_coords = tsne.fit_transform(features)
        return tsne_coords

    def save_features_csv(self, features: np.ndarray, labels: np.ndarray,
                          metadata: Optional[List], save_path: Path):
        """Save features with labels and metadata to CSV."""
        df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])
        df['label'] = labels
        df['label_name'] = [CLASS_NAMES[l] for l in labels]

        if metadata:
            df['pair_id'] = [m.get('pair', -1) for m in metadata]
            df['player1'] = [m.get('player1', '') for m in metadata]
            df['player2'] = [m.get('player2', '') for m in metadata]

        df.to_csv(save_path, index=False)
        print(f"  Saved features to {save_path}")

    def save_tsne_csv(self, tsne_coords: np.ndarray, labels: np.ndarray,
                      predictions: np.ndarray, save_path: Path):
        """Save t-SNE coordinates to CSV."""
        df = pd.DataFrame({
            'tsne_1': tsne_coords[:, 0],
            'tsne_2': tsne_coords[:, 1],
            'label': labels,
            'label_name': [CLASS_NAMES[l] for l in labels],
            'prediction': predictions,
            'correct': labels == predictions
        })
        df.to_csv(save_path, index=False)
        print(f"  Saved t-SNE coordinates to {save_path}")

    def plot_tsne(self, tsne_coords: np.ndarray, labels: np.ndarray,
                  predictions: np.ndarray, save_path: Path,
                  title: str = "t-SNE Feature Visualization"):
        """
        Plot t-SNE scatter plot colored by class.
        Generates Fig. X for Results section - Feature Analysis.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Colored by true labels
        for i, name in enumerate(CLASS_NAMES):
            mask = labels == i
            axes[0].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                          c=CLASS_COLORS[name], label=name, alpha=0.6, s=30)

        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        axes[0].set_title('Feature Space (True Labels)')
        axes[0].legend()

        # Plot 2: Highlight misclassified samples
        correct_mask = labels == predictions

        # Plot correct predictions lightly
        for i, name in enumerate(CLASS_NAMES):
            mask = (labels == i) & correct_mask
            axes[1].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                          c=CLASS_COLORS[name], alpha=0.3, s=20)

        # Plot misclassified samples with markers
        incorrect_mask = ~correct_mask
        axes[1].scatter(tsne_coords[incorrect_mask, 0], tsne_coords[incorrect_mask, 1],
                       c='red', marker='x', s=50, linewidths=2, label='Misclassified')

        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        axes[1].set_title('Misclassified Samples Highlighted')
        axes[1].legend()

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved t-SNE plot to {save_path}")


# =============================================================================
# Part 3: Error & Mechanism Analysis
# =============================================================================

class ErrorAnalyzer:
    """Handles pair-wise analysis and mechanism validation."""

    def __init__(self, class_names: List[str] = CLASS_NAMES):
        self.class_names = class_names

    def analyze_pair_performance(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-pair accuracy statistics.

        Args:
            predictions_df: DataFrame with columns: pair_id, true_label, pred_label, correct

        Returns:
            pair_stats: DataFrame with pair_id, correct_count, total_count, accuracy
        """
        pair_stats = predictions_df.groupby('pair_id').agg({
            'correct': ['sum', 'count', 'mean']
        }).reset_index()

        pair_stats.columns = ['pair_id', 'correct_count', 'total_count', 'accuracy']
        pair_stats = pair_stats.sort_values('accuracy')

        return pair_stats

    def identify_hard_pairs(self, pair_stats: pd.DataFrame,
                            threshold_percentile: float = 20) -> List[int]:
        """
        Identify pairs with lowest accuracy (hard pairs).

        Args:
            pair_stats: DataFrame from analyze_pair_performance
            threshold_percentile: Bottom X% are considered hard

        Returns:
            List of hard pair IDs
        """
        threshold = np.percentile(pair_stats['accuracy'], threshold_percentile)
        hard_pairs = pair_stats[pair_stats['accuracy'] <= threshold]['pair_id'].tolist()
        return hard_pairs

    def compute_pair_statistics(self, pair_stats: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute statistical summary for pair-wise analysis.

        Returns:
            Dict with ANOVA results, effect sizes, etc.
        """
        # ANOVA test: are accuracies different across pairs?
        # Note: This tests if pair identity affects accuracy
        # (We group samples, not pairs, so we need per-sample data)

        stats_results = {
            'mean_accuracy': pair_stats['accuracy'].mean(),
            'std_accuracy': pair_stats['accuracy'].std(),
            'min_accuracy': pair_stats['accuracy'].min(),
            'max_accuracy': pair_stats['accuracy'].max(),
            'n_pairs': len(pair_stats),
            'hard_pairs_count': len(self.identify_hard_pairs(pair_stats))
        }

        return stats_results

    def save_pair_stats_csv(self, pair_stats: pd.DataFrame, save_path: Path):
        """Save pair statistics to CSV."""
        pair_stats.to_csv(save_path, index=False)
        print(f"  Saved pair statistics to {save_path}")

    def plot_pair_accuracy(self, pair_stats: pd.DataFrame, save_path: Path,
                          overall_accuracy: float, title: str = "Per-Pair Accuracy"):
        """
        Plot bar chart of accuracy by pair ID.
        Generates Fig. X for Discussion section - Per-pair Analysis.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Sort by accuracy for visualization
        pair_stats_sorted = pair_stats.sort_values('accuracy', ascending=False)

        # Color bars: red for hard pairs (below threshold)
        threshold = pair_stats['accuracy'].mean() - 1.5 * pair_stats['accuracy'].std()
        colors = ['#e74c3c' if acc < threshold else '#3498db'
                  for acc in pair_stats_sorted['accuracy']]

        bars = ax.bar(range(len(pair_stats_sorted)), pair_stats_sorted['accuracy'],
                     color=colors, alpha=0.8)

        # Add overall accuracy line
        ax.axhline(y=overall_accuracy, color='green', linestyle='--', linewidth=2,
                  label=f'Overall Accuracy: {overall_accuracy:.2%}')

        # Add threshold line
        if threshold > 0:
            ax.axhline(y=threshold, color='red', linestyle=':', linewidth=1,
                      label=f'Hard Pair Threshold: {threshold:.2%}')

        ax.set_xlabel('Pair (sorted by accuracy)')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_xticks(range(len(pair_stats_sorted)))
        ax.set_xticklabels(pair_stats_sorted['pair_id'], rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved pair accuracy plot to {save_path}")


class MechanismAnalyzer:
    """Analyzes interaction mechanisms: spatial sensitivity and feature correlation."""

    def __init__(self, model_type: str, class_names: List[str] = CLASS_NAMES):
        self.model_type = model_type
        self.class_names = class_names

    def compute_gaze_distance(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        """
        Compute distance between gaze centers of two heatmaps.

        Args:
            img_a, img_b: (3, H, W) tensors

        Returns:
            distance: Euclidean distance between centers of mass
        """
        def center_of_mass(heatmap: torch.Tensor) -> Tuple[float, float]:
            """Compute center of mass of a heatmap."""
            # Convert to grayscale intensity
            gray = heatmap.mean(dim=0)  # (H, W)
            total = gray.sum()

            if total < 1e-8:
                return 112.0, 112.0  # Center as default

            h, w = gray.shape
            y_coords = torch.arange(h, dtype=torch.float32, device=gray.device).unsqueeze(1)
            x_coords = torch.arange(w, dtype=torch.float32, device=gray.device).unsqueeze(0)

            cy = (gray * y_coords).sum() / total
            cx = (gray * x_coords).sum() / total

            return cy.item(), cx.item()

        cy_a, cx_a = center_of_mass(img_a)
        cy_b, cx_b = center_of_mass(img_b)

        distance = np.sqrt((cy_a - cy_b)**2 + (cx_a - cx_b)**2)
        return distance

    def analyze_spatial_sensitivity(self, dataloader: DataLoader,
                                    predictions: np.ndarray,
                                    labels: np.ndarray) -> pd.DataFrame:
        """
        Analyze relationship between gaze spatial overlap and accuracy.
        For Early Fusion models.

        Returns:
            DataFrame with distance, correct, label columns
        """
        results = []
        idx = 0

        for batch in tqdm(dataloader, desc="Computing gaze distances"):
            if len(batch) == 4:
                img_a, img_b, batch_labels, _ = batch
            else:
                img_a, img_b, batch_labels = batch

            for i in range(len(batch_labels)):
                distance = self.compute_gaze_distance(img_a[i], img_b[i])
                results.append({
                    'distance': distance,
                    'correct': int(predictions[idx] == labels[idx]),
                    'label': labels[idx],
                    'label_name': self.class_names[labels[idx]]
                })
                idx += 1

        return pd.DataFrame(results)

    def compute_feature_correlation(self, model: nn.Module,
                                   dataloader: DataLoader,
                                   device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity between stream features (Late Fusion).

        Returns:
            similarities: (N,) array of cosine similarities
            labels: (N,) array of labels
        """
        model.eval()
        similarities = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing feature correlations"):
                if len(batch) == 4:
                    img_a, img_b, labels, _ = batch
                else:
                    img_a, img_b, labels = batch

                img_a = img_a.to(device)
                img_b = img_b.to(device)

                # Get features from both streams
                feat_dict = model.get_features(img_a, img_b)
                cls1 = F.normalize(feat_dict['cls1'], dim=-1)
                cls2 = F.normalize(feat_dict['cls2'], dim=-1)

                # Cosine similarity
                cos_sim = (cls1 * cls2).sum(dim=-1)  # (B,)

                similarities.extend(cos_sim.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(similarities), np.array(all_labels)

    def statistical_tests_by_class(self, values: np.ndarray,
                                   labels: np.ndarray) -> pd.DataFrame:
        """
        Perform statistical tests comparing values across classes.

        Returns:
            DataFrame with pairwise t-test results
        """
        results = []

        # Group by class
        class_groups = {name: values[labels == i] for i, name in enumerate(self.class_names)}

        # ANOVA
        f_stat, anova_p = stats.f_oneway(*[class_groups[name] for name in self.class_names])
        results.append({
            'comparison': 'ANOVA (all classes)',
            't_stat': f_stat,
            'p_value': anova_p,
            'significant': anova_p < 0.05,
            'effect_size': 'eta_squared (see below)'
        })

        # Pairwise t-tests with Bonferroni correction
        comparisons = [(0, 1), (0, 2), (1, 2)]
        alpha = 0.05 / len(comparisons)  # Bonferroni

        for i, j in comparisons:
            name_i, name_j = self.class_names[i], self.class_names[j]

            t_stat, p_value = stats.ttest_ind(class_groups[name_i], class_groups[name_j])

            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(class_groups[name_i]) + np.var(class_groups[name_j])) / 2)
            if pooled_std > 0:
                cohen_d = (np.mean(class_groups[name_i]) - np.mean(class_groups[name_j])) / pooled_std
            else:
                cohen_d = 0.0

            results.append({
                'comparison': f'{name_i} vs {name_j}',
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'effect_size': f"Cohen's d = {cohen_d:.3f}"
            })

        return pd.DataFrame(results)

    def plot_mechanism_analysis(self, data: pd.DataFrame,
                                analysis_type: str,
                                save_path: Path,
                                stats_df: pd.DataFrame):
        """
        Plot mechanism analysis figures.
        Generates Fig. X for Discussion section - Mechanism Validation.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if analysis_type == 'spatial':
            # Left: Accuracy vs Distance (binned)
            data['distance_bin'] = pd.cut(data['distance'], bins=10)
            bin_stats = data.groupby('distance_bin')['correct'].mean()

            axes[0].bar(range(len(bin_stats)), bin_stats.values, alpha=0.7)
            axes[0].set_xlabel('Gaze Distance Bin')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Accuracy vs. Gaze Distance')
            axes[0].set_xticklabels([f'{int(b.left)}-{int(b.right)}' for b in bin_stats.index],
                                   rotation=45, ha='right')

            # Right: Distance distribution by class
            for name in self.class_names:
                class_data = data[data['label_name'] == name]['distance']
                sns.kdeplot(class_data, ax=axes[1], label=name, shade=True, alpha=0.3)

            axes[1].set_xlabel('Gaze Distance')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Gaze Distance Distribution by Class')
            axes[1].legend()

        else:  # correlation
            # Left: Violin plot of similarity by class
            sns.violinplot(data=data, x='label_name', y='similarity', ax=axes[0],
                          palette=CLASS_COLORS)
            axes[0].set_xlabel('Class')
            axes[0].set_ylabel('Feature Cosine Similarity')
            axes[0].set_title('Inter-stream Feature Correlation by Class')

            # Right: Box plot with individual points
            sns.boxplot(data=data, x='label_name', y='similarity', ax=axes[1],
                       palette=CLASS_COLORS, showfliers=False)
            sns.stripplot(data=data, x='label_name', y='similarity', ax=axes[1],
                         color='black', alpha=0.2, size=2)
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Feature Cosine Similarity')
            axes[1].set_title('Feature Correlation Distribution')

        # Add statistical test results as text
        stats_text = "Statistical Tests:\n"
        for _, row in stats_df.iterrows():
            sig = "*" if row['significant'] else ""
            stats_text += f"{row['comparison']}: p={row['p_value']:.4f}{sig}\n"

        fig.text(0.02, 0.02, stats_text, fontsize=8, family='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Mechanism Analysis ({analysis_type.capitalize()})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved mechanism analysis plot to {save_path}")


# =============================================================================
# Learning Curves
# =============================================================================

class LearningCurveAnalyzer:
    """Handles learning curve visualization from wandb or checkpoints."""

    def __init__(self):
        self.history = None

    def fetch_wandb_history(self, project_name: str, run_name: str) -> pd.DataFrame:
        """
        Fetch training history from wandb.

        Args:
            project_name: Wandb project name
            run_name: Run name to fetch

        Returns:
            DataFrame with epoch, train_loss, val_loss, etc.
        """
        try:
            import wandb
            api = wandb.Api()
            runs = api.runs(project_name)

            for run in runs:
                if run.name == run_name:
                    history = run.history()
                    # Select relevant columns if they exist
                    cols = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1']
                    available_cols = [c for c in cols if c in history.columns]
                    self.history = history[available_cols]
                    return self.history

            print(f"  Warning: Run '{run_name}' not found in project '{project_name}'")
            return None

        except ImportError:
            print("  Warning: wandb not installed. Cannot fetch learning curves.")
            return None
        except Exception as e:
            print(f"  Warning: Failed to fetch wandb history: {e}")
            return None

    def extract_from_checkpoints(self, checkpoint_dir: Path) -> pd.DataFrame:
        """
        Extract metrics from periodic checkpoints.

        Args:
            checkpoint_dir: Directory containing checkpoint_epoch_*.pt files

        Returns:
            DataFrame with available metrics per epoch
        """
        checkpoint_files = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))

        if not checkpoint_files:
            print(f"  Warning: No checkpoint files found in {checkpoint_dir}")
            return None

        history = []
        for ckpt_path in checkpoint_files:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                epoch = ckpt.get('epoch', 0)

                row = {'epoch': epoch}

                # Check for various metric storage patterns
                if 'metrics' in ckpt:
                    row.update(ckpt['metrics'])
                if 'val_metrics' in ckpt:
                    row.update({f'val_{k}': v for k, v in ckpt['val_metrics'].items()})
                if 'train_loss' in ckpt:
                    row['train_loss'] = ckpt['train_loss']
                if 'val_loss' in ckpt:
                    row['val_loss'] = ckpt['val_loss']
                if 'best_metric' in ckpt:
                    row['best_metric'] = ckpt['best_metric']

                history.append(row)

            except Exception as e:
                print(f"  Warning: Failed to load {ckpt_path}: {e}")
                continue

        if history:
            self.history = pd.DataFrame(history).sort_values('epoch')
            return self.history
        return None

    def save_history_csv(self, save_path: Path):
        """Save training history to CSV."""
        if self.history is not None:
            self.history.to_csv(save_path, index=False)
            print(f"  Saved training history to {save_path}")

    def plot_learning_curves(self, save_path: Path, title: str = "Learning Curves"):
        """
        Plot training and validation curves.
        Generates Fig. X for Results section - Training Progress.
        """
        if self.history is None or len(self.history) == 0:
            print("  Warning: No history data available for learning curves")
            return

        # Determine available metrics
        has_loss = 'train_loss' in self.history.columns and 'val_loss' in self.history.columns
        has_acc = 'train_acc' in self.history.columns or 'val_acc' in self.history.columns
        has_f1 = 'val_f1' in self.history.columns

        n_plots = sum([has_loss, has_acc, has_f1])
        if n_plots == 0:
            print("  Warning: No plottable metrics found in history")
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        if has_loss:
            ax = axes[plot_idx]
            if 'train_loss' in self.history.columns:
                ax.plot(self.history['epoch'], self.history['train_loss'],
                       label='Train', color='blue')
            if 'val_loss' in self.history.columns:
                ax.plot(self.history['epoch'], self.history['val_loss'],
                       label='Validation', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curves')
            ax.legend()
            plot_idx += 1

        if has_acc:
            ax = axes[plot_idx]
            if 'train_acc' in self.history.columns:
                ax.plot(self.history['epoch'], self.history['train_acc'],
                       label='Train', color='blue')
            if 'val_acc' in self.history.columns:
                ax.plot(self.history['epoch'], self.history['val_acc'],
                       label='Validation', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curves')
            ax.legend()
            plot_idx += 1

        if has_f1:
            ax = axes[plot_idx]
            ax.plot(self.history['epoch'], self.history['val_f1'],
                   label='Val F1', color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.set_title('Validation F1 Curve')
            ax.legend()

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved learning curves to {save_path}")


# =============================================================================
# Multi-Model Comparison
# =============================================================================

class MultiModelComparator:
    """Compare multiple trained models side by side."""

    def __init__(self, analyzers: List['GazeAnalyzer'], labels: List[str]):
        """
        Args:
            analyzers: List of GazeAnalyzer instances
            labels: Display labels for each model
        """
        self.analyzers = analyzers
        self.labels = labels

    def compare_metrics(self) -> pd.DataFrame:
        """Generate comparison table of all metrics."""
        all_metrics = []
        for analyzer, label in zip(self.analyzers, self.labels):
            metrics = analyzer.quant_analyzer.compute_metrics(
                analyzer.labels, analyzer.predictions
            )
            metrics['model'] = label
            all_metrics.append(metrics)

        df = pd.DataFrame(all_metrics)
        # Reorder columns
        cols = ['model'] + [c for c in df.columns if c != 'model']
        return df[cols]

    def save_comparison_csv(self, save_path: Path):
        """Save comparison table to CSV."""
        df = self.compare_metrics()
        df.to_csv(save_path, index=False)
        print(f"  Saved comparison table to {save_path}")

    def plot_comparison_confusion_matrices(self, save_path: Path):
        """Plot side-by-side confusion matrices."""
        n_models = len(self.analyzers)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (analyzer, label) in enumerate(zip(self.analyzers, self.labels)):
            cm = analyzer.quant_analyzer.compute_confusion_matrix(
                analyzer.labels, analyzer.predictions
            )

            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            ax.set_title(f'{label}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison confusion matrices to {save_path}")

    def plot_comparison_roc(self, save_path: Path):
        """Plot overlaid ROC curves from all models."""
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = plt.cm.tab10.colors

        for idx, (analyzer, label) in enumerate(zip(self.analyzers, self.labels)):
            roc_data = analyzer.quant_analyzer.compute_roc_data(
                analyzer.labels, analyzer.probabilities
            )
            micro = roc_data['micro']
            ax.plot(micro['fpr'], micro['tpr'],
                   label=f'{label} (AUC={micro["auc"]:.3f})',
                   color=colors[idx], linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison (Micro-Average)')
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison ROC curves to {save_path}")

    def plot_comparison_metrics_bar(self, save_path: Path):
        """Plot bar chart comparing key metrics."""
        df = self.compare_metrics()

        metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        available_metrics = [m for m in metrics_to_plot if m in df.columns]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(available_metrics))
        width = 0.8 / len(self.labels)

        for idx, label in enumerate(self.labels):
            model_data = df[df['model'] == label]
            values = [model_data[m].values[0] for m in available_metrics]
            offset = (idx - len(self.labels)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=label, alpha=0.8)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison metrics bar chart to {save_path}")


# =============================================================================
# Main Analyzer Class
# =============================================================================

class GazeAnalyzer:
    """
    Comprehensive analyzer for Gaze classification models.

    Encapsulates all analysis functionality:
    - Quantitative: metrics, confusion matrix, ROC curves
    - Qualitative: attention visualization, t-SNE
    - Error analysis: pair-wise, mechanism validation
    - Learning curves
    """

    def __init__(self,
                 config_path: str,
                 checkpoint_path: str,
                 model_type: str,
                 fusion_mode: Optional[str] = None,
                 device: Optional[str] = None,
                 exp_name: Optional[str] = None):
        """
        Initialize analyzer with model and data.

        Args:
            config_path: Path to YAML config file
            checkpoint_path: Path to model checkpoint
            model_type: 'early' or 'late'
            fusion_mode: Override config fusion_mode if needed
            device: 'cuda' or 'cpu' (auto-detect if None)
            exp_name: Experiment name for output folders
        """
        # Load config
        self.config = self._load_config(config_path)
        self.model_type = model_type
        self.fusion_mode = fusion_mode or self.config['model']['fusion_mode']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate experiment name
        self.exp_name = exp_name or f"{model_type}fusion_{self.fusion_mode}"

        # Setup output directories
        self.paths = create_output_dirs(PROJECT_ROOT, self.exp_name)

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Load data
        print("Loading validation dataset...")
        self.val_dataset, self.val_loader = self._load_data()
        self.val_loader_with_metadata = self._load_data_with_metadata()

        # Initialize analyzers
        self.quant_analyzer = QuantitativeAnalyzer()
        self.attention_analyzer = AttentionAnalyzer(self.model, model_type, self.device)
        self.feature_analyzer = FeatureAnalyzer(self.model, model_type, self.device)
        self.error_analyzer = ErrorAnalyzer()
        self.mechanism_analyzer = MechanismAnalyzer(model_type)
        self.learning_analyzer = LearningCurveAnalyzer()

        # Storage for results
        self.predictions = None
        self.labels = None
        self.probabilities = None
        self.features = None
        self.metadata = None

    def _load_config(self, path: str) -> Dict:
        """Load YAML config file."""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Get config from checkpoint or use loaded config
        model_config = checkpoint.get('config', self.config).get('model', self.config['model'])

        if self.model_type == 'early':
            model = EarlyFusionViT(
                model_name=model_config.get('name', 'vit_base_patch16_224'),
                num_classes=model_config.get('num_classes', 3),
                pretrained=False,  # Don't load pretrained, we have checkpoint
                fusion_mode=model_config.get('fusion_mode', self.fusion_mode),
                weight_init_strategy=model_config.get('weight_init_strategy', 'duplicate')
            )
        else:
            model = LateFusionViT(
                model_name=model_config.get('name', 'vit_base_patch16_224'),
                num_classes=model_config.get('num_classes', 3),
                pretrained=False,
                fusion_mode=model_config.get('fusion_mode', self.fusion_mode),
                dropout=model_config.get('dropout', 0.1)
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _load_data(self) -> Tuple[GazePairDataset, DataLoader]:
        """Load validation dataset."""
        data_config = self.config['data']

        # Create validation transform
        val_transform = transforms.Compose([
            transforms.Resize((data_config.get('image_size', 224),
                              data_config.get('image_size', 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Load metadata
        metadata_path = PROJECT_ROOT / data_config['metadata_path']
        with open(metadata_path, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)

        # Filter validation pairs
        val_pairs = data_config.get('val_pairs', [33, 34, 35, 36, 37, 38, 39, 40])
        val_metadata = [m for m in all_metadata if m['pair'] in val_pairs]

        # Create dataset
        label2id = data_config.get('label2id', {"Single": 0, "Competition": 1, "Cooperation": 2})

        val_dataset = GazePairDataset(
            metadata=val_metadata,
            image_base_path=data_config['image_base_path'],
            image_extension=data_config.get('image_extension', '.jpg'),
            label2id=label2id,
            transform=val_transform,
            return_metadata=False
        )

        # Custom collate function
        def collate_fn(batch):
            img_a = torch.stack([item[0] for item in batch])
            img_b = torch.stack([item[1] for item in batch])
            labels = torch.tensor([item[2] for item in batch])
            return img_a, img_b, labels

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training'].get('batch_size', 16),
            shuffle=False,
            num_workers=0,  # Windows compatibility
            collate_fn=collate_fn
        )

        return val_dataset, val_loader

    def _load_data_with_metadata(self) -> DataLoader:
        """Load validation dataset with metadata for pair analysis."""
        data_config = self.config['data']

        val_transform = transforms.Compose([
            transforms.Resize((data_config.get('image_size', 224),
                              data_config.get('image_size', 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        metadata_path = PROJECT_ROOT / data_config['metadata_path']
        with open(metadata_path, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)

        val_pairs = data_config.get('val_pairs', [33, 34, 35, 36, 37, 38, 39, 40])
        val_metadata = [m for m in all_metadata if m['pair'] in val_pairs]

        label2id = data_config.get('label2id', {"Single": 0, "Competition": 1, "Cooperation": 2})

        val_dataset = GazePairDataset(
            metadata=val_metadata,
            image_base_path=data_config['image_base_path'],
            image_extension=data_config.get('image_extension', '.jpg'),
            label2id=label2id,
            transform=val_transform,
            return_metadata=True  # Enable metadata
        )

        def collate_with_metadata(batch):
            img_a = torch.stack([item[0] for item in batch])
            img_b = torch.stack([item[1] for item in batch])
            labels = torch.tensor([item[2] for item in batch])
            metadata = [item[3] for item in batch]
            return img_a, img_b, labels, metadata

        return DataLoader(
            val_dataset,
            batch_size=self.config['training'].get('batch_size', 16),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_with_metadata
        )

    def run_inference(self):
        """Run model inference on validation set."""
        print("\n[Step 1] Running inference...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        all_metadata = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader_with_metadata, desc="Inference"):
                img_a, img_b, labels, metadata = batch
                img_a = img_a.to(self.device)
                img_b = img_b.to(self.device)

                logits = self.model(img_a, img_b)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.append(probs.cpu().numpy())
                all_metadata.extend(metadata)

        self.predictions = np.array(all_predictions)
        self.labels = np.array(all_labels)
        self.probabilities = np.concatenate(all_probs, axis=0)
        self.metadata = all_metadata

        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': self.labels,
            'pred_label': self.predictions,
            'correct': self.labels == self.predictions,
            'confidence': self.probabilities.max(axis=1),
            'pair_id': [m['pair'] for m in self.metadata]
        })
        for i, name in enumerate(CLASS_NAMES):
            predictions_df[f'prob_{name}'] = self.probabilities[:, i]

        predictions_df.to_csv(self.paths['raw_result'] / 'predictions.csv', index=False)
        print(f"  Saved predictions to {self.paths['raw_result'] / 'predictions.csv'}")

    def run_quantitative_analysis(self):
        """Run Part 1: Quantitative analysis."""
        print("\n[Step 2] Quantitative Analysis...")

        # Compute metrics
        metrics = self.quant_analyzer.compute_metrics(self.labels, self.predictions)
        self.quant_analyzer.save_metrics_csv(metrics, self.paths['raw_result'] / 'metrics.csv')
        self.quant_analyzer.save_metrics_csv(metrics, self.paths['tables'] / f'table_performance_{self.exp_name}.csv')

        # Confusion matrix
        cm = self.quant_analyzer.compute_confusion_matrix(self.labels, self.predictions)
        self.quant_analyzer.save_confusion_matrix_csv(cm, self.paths['raw_result'] / 'conf_mat.csv')
        self.quant_analyzer.plot_confusion_matrix(
            cm, self.paths['figures'] / 'fig_conf_mat.pdf',
            title=f'Confusion Matrix - {self.exp_name}'
        )

        # ROC curves
        roc_data = self.quant_analyzer.compute_roc_data(self.labels, self.probabilities)
        self.quant_analyzer.save_roc_data_csv(roc_data, self.paths['raw_result'] / 'roc_data.csv')
        self.quant_analyzer.plot_roc_curves(
            roc_data, self.paths['figures'] / 'fig_roc_curves.pdf',
            title=f'ROC Curves - {self.exp_name}'
        )

        # Print summary
        print(f"\n  Performance Summary:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"    F1 (Weighted): {metrics['f1_weighted']:.4f}")

    def run_qualitative_analysis(self):
        """Run Part 2: Qualitative analysis (features and attention)."""
        print("\n[Step 3] Qualitative Analysis...")

        # Extract features
        features, labels, metadata = self.feature_analyzer.extract_features(
            self.val_loader_with_metadata, return_metadata=True
        )
        self.features = features

        # Save features
        self.feature_analyzer.save_features_csv(
            features, labels, metadata,
            self.paths['raw_result'] / 'features.csv'
        )

        # t-SNE
        tsne_coords = self.feature_analyzer.compute_tsne(features)
        self.feature_analyzer.save_tsne_csv(
            tsne_coords, self.labels, self.predictions,
            self.paths['raw_result'] / 'tsne_coords.csv'
        )
        self.feature_analyzer.plot_tsne(
            tsne_coords, self.labels, self.predictions,
            self.paths['figures'] / 'fig_tsne.pdf',
            title=f't-SNE Feature Visualization - {self.exp_name}'
        )

        # Attention visualization for selected samples
        print("  Generating attention visualizations...")
        self._visualize_attention_samples()

    def _visualize_attention_samples(self):
        """Generate attention visualization for hard and easy samples."""
        # Find hard samples (misclassified or low confidence)
        confidences = self.probabilities.max(axis=1)
        incorrect_mask = self.predictions != self.labels

        # Get indices
        incorrect_indices = np.where(incorrect_mask)[0]
        correct_indices = np.where(~incorrect_mask)[0]

        # Select samples
        n_samples = min(3, len(incorrect_indices))
        hard_indices = incorrect_indices[:n_samples] if len(incorrect_indices) > 0 else []

        n_easy = min(3, len(correct_indices))
        # Get highest confidence correct predictions
        correct_confidences = confidences[correct_indices]
        easy_indices = correct_indices[np.argsort(correct_confidences)[-n_easy:]]

        # Generate visualizations
        fig, axes = plt.subplots(len(hard_indices) + len(easy_indices), 4,
                                figsize=(16, 4*(len(hard_indices) + len(easy_indices))))

        if len(hard_indices) + len(easy_indices) == 0:
            print("  Warning: No samples available for attention visualization")
            return

        sample_indices = list(hard_indices) + list(easy_indices)

        for row, idx in enumerate(sample_indices):
            # Get sample
            if hasattr(self.val_dataset, '__getitem__'):
                sample = self.val_dataset[idx]
                if len(sample) == 4:
                    img_a, img_b, label, _ = sample
                else:
                    img_a, img_b, label = sample
            else:
                continue

            img_a = img_a.unsqueeze(0)
            img_b = img_b.unsqueeze(0)

            # Generate attention
            self.attention_analyzer.visualize_attention(
                img_a, img_b,
                label, self.predictions[idx], confidences[idx],
                self.paths['figures'] / f'fig_attention_sample_{idx}.png',
                idx
            )

        # Create combined visualization
        print(f"  Saved attention visualizations to {self.paths['figures']}")

    def run_error_analysis(self):
        """Run Part 3: Error and mechanism analysis."""
        print("\n[Step 4] Error & Mechanism Analysis...")

        # Pair-wise analysis
        predictions_df = pd.DataFrame({
            'pair_id': [m['pair'] for m in self.metadata],
            'true_label': self.labels,
            'pred_label': self.predictions,
            'correct': self.labels == self.predictions
        })

        pair_stats = self.error_analyzer.analyze_pair_performance(predictions_df)
        self.error_analyzer.save_pair_stats_csv(
            pair_stats, self.paths['raw_result'] / 'pair_stats.csv'
        )

        overall_accuracy = (self.predictions == self.labels).mean()
        self.error_analyzer.plot_pair_accuracy(
            pair_stats, self.paths['figures'] / 'fig_pair_accuracy.png',
            overall_accuracy, title=f'Per-Pair Accuracy - {self.exp_name}'
        )

        # Mechanism analysis
        if self.model_type == 'early':
            print("  Analyzing spatial sensitivity (Early Fusion)...")
            spatial_df = self.mechanism_analyzer.analyze_spatial_sensitivity(
                self.val_loader, self.predictions, self.labels
            )
            spatial_df.to_csv(self.paths['raw_result'] / 'spatial_analysis.csv', index=False)

            stats_df = self.mechanism_analyzer.statistical_tests_by_class(
                spatial_df['distance'].values, spatial_df['label'].values
            )
            stats_df.to_csv(self.paths['raw_result'] / 'statistical_tests.csv', index=False)

            self.mechanism_analyzer.plot_mechanism_analysis(
                spatial_df, 'spatial',
                self.paths['figures'] / 'fig_mechanism_analysis.pdf',
                stats_df
            )
        else:
            print("  Analyzing feature correlation (Late Fusion)...")
            similarities, labels = self.mechanism_analyzer.compute_feature_correlation(
                self.model, self.val_loader, self.device
            )

            corr_df = pd.DataFrame({
                'similarity': similarities,
                'label': labels,
                'label_name': [CLASS_NAMES[l] for l in labels]
            })
            corr_df.to_csv(self.paths['raw_result'] / 'correlation_analysis.csv', index=False)

            stats_df = self.mechanism_analyzer.statistical_tests_by_class(similarities, labels)
            stats_df.to_csv(self.paths['raw_result'] / 'statistical_tests.csv', index=False)

            self.mechanism_analyzer.plot_mechanism_analysis(
                corr_df, 'correlation',
                self.paths['figures'] / 'fig_mechanism_analysis.pdf',
                stats_df
            )

    def run_learning_curves(self, wandb_project: Optional[str] = None,
                           wandb_run: Optional[str] = None,
                           checkpoint_dir: Optional[str] = None):
        """Generate learning curves from wandb or checkpoints."""
        print("\n[Step 5] Learning Curves...")

        history = None

        # Try wandb first
        if wandb_project and wandb_run:
            history = self.learning_analyzer.fetch_wandb_history(wandb_project, wandb_run)

        # Fallback to checkpoints
        if history is None and checkpoint_dir:
            history = self.learning_analyzer.extract_from_checkpoints(Path(checkpoint_dir))

        # Auto-detect checkpoint directory from checkpoint path
        if history is None:
            ckpt_dir = Path(self.config.get('checkpoint', {}).get('save_dir', ''))
            if ckpt_dir.exists():
                history = self.learning_analyzer.extract_from_checkpoints(ckpt_dir)

        if history is not None:
            self.learning_analyzer.save_history_csv(
                self.paths['raw_result'] / 'training_history.csv'
            )
            self.learning_analyzer.plot_learning_curves(
                self.paths['figures'] / 'fig_learning_curves.pdf',
                title=f'Learning Curves - {self.exp_name}'
            )
        else:
            print("  Warning: Could not load training history for learning curves")

    def run_full_analysis(self,
                         wandb_project: Optional[str] = None,
                         wandb_run: Optional[str] = None):
        """
        Run complete analysis pipeline.

        Args:
            wandb_project: Wandb project name (optional)
            wandb_run: Wandb run name (optional)
        """
        print(f"\n{'='*60}")
        print(f"Starting Full Analysis: {self.exp_name}")
        print(f"{'='*60}")

        # Step 1: Inference
        self.run_inference()

        # Step 2: Quantitative analysis
        self.run_quantitative_analysis()

        # Step 3: Qualitative analysis
        self.run_qualitative_analysis()

        # Step 4: Error analysis
        self.run_error_analysis()

        # Step 5: Learning curves
        self.run_learning_curves(wandb_project, wandb_run)

        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}")
        print(f"\nResults saved to:")
        print(f"  Raw data: {self.paths['raw_result']}")
        print(f"  Figures: {self.paths['figures']}")
        print(f"  Tables: {self.paths['tables']}")


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Gaze Model Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model analysis
  python analyze_gaze.py --config configs/gaze_earlyfusion.yaml \\
                         --checkpoint runs/best_model.pt \\
                         --model_type early

  # Multi-model comparison
  python analyze_gaze.py --compare \\
                         --checkpoints model1.pt model2.pt \\
                         --configs config1.yaml config2.yaml \\
                         --model_types early late \\
                         --labels "Early-Concat" "Late-Full"
        """
    )

    # Single model arguments
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['early', 'late'],
                       help='Model type: early or late fusion')
    parser.add_argument('--fusion_mode', type=str, default=None,
                       help='Override fusion mode from config')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu (auto-detect if not specified)')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name for output folders')

    # Learning curves
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name for learning curves')
    parser.add_argument('--wandb_run', type=str, default=None,
                       help='Wandb run name for learning curves')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory with checkpoint_epoch_*.pt files')

    # Multi-model comparison
    parser.add_argument('--compare', action='store_true',
                       help='Enable multi-model comparison mode')
    parser.add_argument('--checkpoints', nargs='+',
                       help='List of checkpoint paths for comparison')
    parser.add_argument('--configs', nargs='+',
                       help='List of config paths for comparison')
    parser.add_argument('--model_types', nargs='+',
                       help='Model types for each checkpoint')
    parser.add_argument('--labels', nargs='+',
                       help='Display labels for each model')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.compare:
        # Multi-model comparison mode
        if not all([args.checkpoints, args.configs, args.model_types, args.labels]):
            print("Error: --compare mode requires --checkpoints, --configs, --model_types, and --labels")
            return

        if not (len(args.checkpoints) == len(args.configs) == len(args.model_types) == len(args.labels)):
            print("Error: All lists must have the same length")
            return

        print(f"\n{'='*60}")
        print("Multi-Model Comparison Mode")
        print(f"{'='*60}")

        # Create analyzers for each model
        analyzers = []
        for ckpt, cfg, mtype, label in zip(args.checkpoints, args.configs,
                                           args.model_types, args.labels):
            print(f"\nLoading model: {label}")
            analyzer = GazeAnalyzer(
                config_path=cfg,
                checkpoint_path=ckpt,
                model_type=mtype,
                exp_name=label.replace(' ', '_').lower()
            )
            analyzer.run_inference()
            analyzers.append(analyzer)

        # Run comparison
        comparator = MultiModelComparator(analyzers, args.labels)

        comparison_dir = PROJECT_ROOT / "7_Analysis" / "figures" / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        comparator.save_comparison_csv(PROJECT_ROOT / "7_Analysis" / "tables" / "table_comparison.csv")
        comparator.plot_comparison_confusion_matrices(comparison_dir / "fig_compare_conf_mat.pdf")
        comparator.plot_comparison_roc(comparison_dir / "fig_compare_roc.pdf")
        comparator.plot_comparison_metrics_bar(comparison_dir / "fig_compare_metrics.pdf")

        print(f"\n{'='*60}")
        print("Comparison Complete!")
        print(f"{'='*60}")
        print(f"Results saved to: {comparison_dir}")

    else:
        # Single model analysis
        if not all([args.config, args.checkpoint, args.model_type]):
            print("Error: Single model mode requires --config, --checkpoint, and --model_type")
            return

        analyzer = GazeAnalyzer(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            fusion_mode=args.fusion_mode,
            device=args.device,
            exp_name=args.exp_name
        )

        analyzer.run_full_analysis(
            wandb_project=args.wandb_project,
            wandb_run=args.wandb_run
        )


if __name__ == '__main__':
    main()
