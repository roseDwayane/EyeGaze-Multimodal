"""
EEG Model Analysis Script

Main script for analyzing trained DualEEG models. Performs:
1. Core metrics (Confusion Matrix, Classification Report)
2. Frequency sensitivity analysis (band masking)
3. IBS connectivity matrix extraction and visualization
4. Cross-attention weight analysis
5. t-SNE/UMAP feature embedding visualization
6. Grad-CAM analysis (experimental)

Usage:
    python 7_Analysis/python_scripts/analyze_eeg.py \\
        --checkpoint 4_Experiments/runs/dualEEG/old_eeg/best_model.pt \\
        --output_dir 7_Analysis/outputs \\
        --analyses all

Author: Analysis Pipeline for EyeGaze-Multimodal Project
"""

import sys
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import using importlib for folders starting with numbers
import importlib.util


def import_module_from_path(module_name: str, file_path: str):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import custom modules
# First import the art module (Transformer components)
_art_module = import_module_from_path(
    "art",
    str(PROJECT_ROOT / "3_Models" / "backbones" / "art.py")
)

# The checkpoint was trained with DualEEGTransformer from dual_eeg_transformer.py
_model_module = import_module_from_path(
    "dual_eeg_transformer",
    str(PROJECT_ROOT / "3_Models" / "backbones" / "dual_eeg_transformer.py")
)
DualEEGTransformer = _model_module.DualEEGTransformer

_io_utils = import_module_from_path(
    "io_utils",
    str(PROJECT_ROOT / "6_Utils" / "io_utils.py")
)

_metrics = import_module_from_path(
    "eeg_metrics",
    str(PROJECT_ROOT / "5_Metrics" / "eeg_metrics.py")
)


from datasets import load_dataset

# Import DualEEGDataset
_dataset_module = import_module_from_path(
    "dual_eeg_dataset",
    str(PROJECT_ROOT / "1_Data" / "processed" / "dual_eeg_dataset.py")
)
DualEEGDataset = _dataset_module.DualEEGDataset
collate_fn = _dataset_module.collate_fn



# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CLASS_NAMES = ['Single', 'Competition', 'Cooperation']
DEFAULT_BAND_NAMES = ['broadband', 'delta', 'theta', 'alpha', 'beta', 'gamma']
DEFAULT_FEATURE_NAMES = ['PLV', 'PLI', 'wPLI', 'Coherence', 'Power_Corr', 'Phase_Diff', 'Time_Corr']


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_config(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to best_model.pt
        device: Device to load model to

    Returns:
        Tuple of (model, config)
    """
    print(f"[Analysis] Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})

    # Extract model config
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    ablation_config = config.get('ablation', {})

    # Infer max_len from checkpoint pos_embed shape
    pos_embed_shape = checkpoint['model_state_dict']['pos_embed.pos_embed.weight'].shape
    max_len = pos_embed_shape[0]
    print(f"[Analysis] Inferred max_len from checkpoint: {max_len}")

    # Create model with same configuration (DualEEGTransformer)
    # Note: config uses 'num_labels' instead of 'num_classes'
    model = DualEEGTransformer(
        in_channels=model_config.get('in_channels', 32),
        num_classes=model_config.get('num_labels', model_config.get('num_classes', 3)),
        d_model=model_config.get('d_model', 256),
        num_layers=model_config.get('num_layers', 6),
        num_heads=model_config.get('num_heads', 8),
        d_ff=model_config.get('d_ff', 1024),
        dropout=model_config.get('dropout', 0.1),
        max_len=max_len,  # Use inferred max_len
        conv_kernel_size=model_config.get('conv_kernel_size', 25),
        conv_stride=model_config.get('conv_stride', 4),
        conv_layers=model_config.get('conv_layers', 2),
        sampling_rate=data_config.get('sampling_rate', 256),
        # Spectrogram params
        use_spectrogram=model_config.get('use_spectrogram', True),
        spec_n_fft=model_config.get('spec_n_fft', 128),
        spec_hop_length=model_config.get('spec_hop_length', 64),
        spec_freq_bins=model_config.get('spec_freq_bins', 64),
        # IBS params
        use_robust_ibs=model_config.get('use_robust_ibs', True),
        use_ibs=ablation_config.get('use_ibs', True),
        use_cross_attention=ablation_config.get('use_cross_attention', True),
        ibs_instance_norm=ablation_config.get('ibs_instance_norm', True),
        ibs_feature_type=ablation_config.get('ibs_feature_type', 'all'),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"[Analysis] Model loaded successfully (epoch {checkpoint.get('epoch', 'N/A')})")
    best_metric = checkpoint.get('best_metric', 'N/A')
    if isinstance(best_metric, (int, float)):
        print(f"[Analysis] Best metric: {best_metric:.4f}")
    else:
        print(f"[Analysis] Best metric: {best_metric}")

    return model, config


def tuple_collate_fn(batch):
    """Wrapper to convert dict to tuple for eeg_metrics compatibility."""
    batch_dict = collate_fn(batch)
    return batch_dict['eeg1'], batch_dict['eeg2'], batch_dict['labels'], batch_dict['dataset_idx']


def create_test_dataloader(
    config: Dict,
    batch_size: int = 32
) -> DataLoader:
    """
    Create test DataLoader based on config, matching training logic.

    Args:
        config: Training config dict
        batch_size: Batch size for inference

    Returns:
        DataLoader for test/validation set
    """
    data_config = config.get('data', {})

    # Paths
    metadata_path = data_config.get('metadata_path', '1_Data/metadata/complete_metadata.json')
    eeg_base_path = data_config.get('eeg_base_path', '1_Data/datasets/EEGseg')
    
    print(f"[Analysis] Loading metadata from {metadata_path}")
    datasets = load_dataset("json", data_files=str(metadata_path), split="train")

    # Split
    test_size = data_config.get('train_test_split', 0.2)
    seed = data_config.get('random_seed', 42)

    print(f"[Analysis] Splitting dataset (seed={seed}, test_size={test_size})...")
    try:
        split_datasets = datasets.train_test_split(
            test_size=test_size,
            seed=seed,
            stratify_by_column='class'
        )
    except (ValueError, KeyError):
        print("[Analysis] Stratified split failed, using random split")
        split_datasets = datasets.train_test_split(
            test_size=test_size,
            seed=seed
        )
    
    val_data = split_datasets['test']
    print(f"[Analysis] Test samples: {len(val_data)}")

    # Create Dataset
    # We rely on DualEEGDataset being imported dynamically
    val_dataset = DualEEGDataset(
        dataset=val_data,
        eeg_base_path=eeg_base_path,
        label2id=data_config.get('label2id', {'Single': 0, 'Competition': 1, 'Cooperation': 2}),
        window_size=data_config.get('window_size', 1024),
        stride=data_config.get('stride', 512),
        sampling_rate=data_config.get('sampling_rate', 256),
        filter_low=data_config.get('filter_low', 1.0),
        filter_high=data_config.get('filter_high', 45.0),
        enable_preprocessing=data_config.get('enable_preprocessing', False)
    )

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for analysis to avoid issues
        collate_fn=tuple_collate_fn,
        pin_memory=True
    )

    return dataloader


# =============================================================================
# Analysis Functions
# =============================================================================

def run_core_metrics_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    class_names: List[str] = None
) -> Dict:
    """
    Run core classification metrics analysis.

    Args:
        model: Trained model
        dataloader: Test DataLoader
        device: Device
        output_dir: Output directory
        class_names: List of class names

    Returns:
        Dictionary of results
    """
    print("\n" + "=" * 60)
    print("Running Core Metrics Analysis")
    print("=" * 60)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Run inference
    inference_results = _metrics.run_inference(model, dataloader, device)

    y_true = inference_results['y_true']
    y_pred = inference_results['y_pred']
    y_prob = inference_results['y_prob']

    # Compute metrics
    metrics = _metrics.compute_classification_metrics(y_true, y_pred, class_names)

    print(f"\n[Results] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[Results] Macro F1: {metrics['macro_f1']:.4f}")
    print(f"[Results] Confusion Matrix:\n{metrics['confusion_matrix']}")

    # Save results
    output_path = output_dir / 'core_metrics'
    output_path.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix
    _io_utils.save_confusion_matrix(
        metrics['confusion_matrix'],
        output_path / 'confusion_matrix.csv',
        class_names
    )

    # Save classification report
    _io_utils.save_classification_report(
        y_true, y_pred,
        output_path / 'classification_report.csv',
        class_names
    )

    # Save overall metrics
    _io_utils.save_overall_metrics(
        {
            'Accuracy': metrics['accuracy'],
            'Macro_Precision': metrics['macro_precision'],
            'Macro_Recall': metrics['macro_recall'],
            'Macro_F1': metrics['macro_f1']
        },
        output_path / 'overall_metrics.csv'
    )

    # Save predictions
    _io_utils.save_predictions(
        y_true, y_pred, y_prob,
        output_path / 'predictions.csv',
        class_names
    )

    # Generate visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        metrics['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix_heatmap.png', dpi=150)
    plt.close()

    print(f"[Analysis] Results saved to {output_path}")

    return {
        'accuracy': metrics['accuracy'],
        'f1': metrics['macro_f1'],
        'y_true': y_true,
        'y_pred': y_pred
    }


def run_frequency_sensitivity_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    baseline_accuracy: float,
    baseline_f1: float
) -> None:
    """
    Run frequency band sensitivity analysis.

    Args:
        model: Trained model
        dataloader: Test DataLoader
        device: Device
        output_dir: Output directory
        baseline_accuracy: Baseline accuracy (no masking)
        baseline_f1: Baseline F1 score
    """
    print("\n" + "=" * 60)
    print("Running Frequency Sensitivity Analysis")
    print("=" * 60)

    # Check if model has IBS matrix generator
    if not hasattr(model, 'ibs_matrix_generator'):
        print("[Warning] Model does not have IBS matrix generator. Skipping.")
        return

    # Run sensitivity analysis
    sensitivity_results = _metrics.compute_frequency_sensitivity(
        model, dataloader, device, DEFAULT_BAND_NAMES
    )

    # Save results
    output_path = output_dir / 'frequency_sensitivity'
    output_path.mkdir(parents=True, exist_ok=True)

    _io_utils.save_frequency_sensitivity(
        sensitivity_results,
        output_path / 'frequency_sensitivity.csv',
        baseline_accuracy,
        baseline_f1
    )

    # Generate visualization
    bands = list(sensitivity_results.keys())
    acc_drops = [baseline_accuracy - sensitivity_results[b]['accuracy'] for b in bands]
    f1_drops = [baseline_f1 - sensitivity_results[b]['f1'] for b in bands]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy drop
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(bands)))
    axes[0].bar(bands, acc_drops, color=colors)
    axes[0].set_xlabel('Frequency Band')
    axes[0].set_ylabel('Accuracy Drop')
    axes[0].set_title('Accuracy Drop when Band is Masked')
    axes[0].tick_params(axis='x', rotation=45)

    # F1 drop
    axes[1].bar(bands, f1_drops, color=colors)
    axes[1].set_xlabel('Frequency Band')
    axes[1].set_ylabel('F1 Drop')
    axes[1].set_title('F1 Score Drop when Band is Masked')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path / 'frequency_sensitivity_barplot.png', dpi=150)
    plt.close()

    print(f"[Analysis] Frequency sensitivity results saved to {output_path}")


def run_ibs_connectivity_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    class_names: List[str] = None
) -> None:
    """
    Run IBS connectivity matrix analysis.

    Args:
        model: Trained model
        dataloader: Test DataLoader
        device: Device
        output_dir: Output directory
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("Running IBS Connectivity Analysis")
    print("=" * 60)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Check if model has IBS matrix generator
    if not hasattr(model, 'ibs_matrix_generator'):
        print("[Warning] Model does not have IBS matrix generator. Skipping.")
        return

    # Extract IBS matrices
    ibs_results = _metrics.extract_ibs_matrices(model, dataloader, device)
    matrices = ibs_results['matrices']  # (N, 6, 7, C, C)
    y_true = ibs_results['y_true']
    y_pred = ibs_results['y_pred']

    print(f"[Analysis] Extracted IBS matrices: {matrices.shape}")

    # Setup output directories
    output_path = output_dir / 'ibs_connectivity'
    mean_by_class_path = output_path / 'ibs_mean_by_class'
    diff_path = output_path / 'ibs_difference_coop_vs_comp'

    for p in [output_path, mean_by_class_path, diff_path]:
        p.mkdir(parents=True, exist_ok=True)

    # Save channel names
    # CORRECTED CHANNEL MAPPING (2026-01-19)
    # The order was corrected based on the specific electrode layout:
    # 1->Fp1, 2->Fz, ..., 32->FP2
    channel_names = [
        'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3',
        'T7', 'TP9', 'CP5', 'CP1', 'PZ', 'P3', 'P7', 'O1',
        'OZ', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'CZ',
        'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2'
    ]
    # channel_names = _metrics.get_channel_names(matrices.shape[-1])
    _io_utils.save_channel_names(channel_names, output_path / 'channel_names.csv')

    # Compute mean matrices by class
    mean_matrices = _metrics.compute_mean_ibs_by_class(matrices, y_true, class_names)

    # Save mean matrices for each class/band/feature
    band_names = DEFAULT_BAND_NAMES
    feature_names = DEFAULT_FEATURE_NAMES

    for class_name, class_matrix in mean_matrices.items():
        for band_idx, band_name in enumerate(band_names):
            for feat_idx, feat_name in enumerate(feature_names):
                matrix = class_matrix[band_idx, feat_idx]  # (C, C)
                filename = f'{class_name}_{band_name}_{feat_name}.csv'
                _io_utils.save_ibs_matrix(matrix, mean_by_class_path / filename)

    # Compute and save difference matrices (Cooperation - Competition)
    if 'Cooperation' in mean_matrices and 'Competition' in mean_matrices:
        diff_matrix = _metrics.compute_ibs_difference(
            mean_matrices['Cooperation'],
            mean_matrices['Competition']
        )

        for band_idx, band_name in enumerate(band_names):
            for feat_idx, feat_name in enumerate(feature_names):
                matrix = diff_matrix[band_idx, feat_idx]
                filename = f'diff_{band_name}_{feat_name}.csv'
                _io_utils.save_ibs_matrix(matrix, diff_path / filename)

    # Generate visualization for key bands/features
    key_visualizations = [
        ('theta', 'PLV'),
        ('alpha', 'PLV'),
        ('theta', 'Coherence'),
    ]

    for band_name, feat_name in key_visualizations:
        band_idx = band_names.index(band_name)
        feat_idx = feature_names.index(feat_name)

        # Plot mean matrices for each class
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, class_name in enumerate(class_names):
            matrix = mean_matrices[class_name][band_idx, feat_idx]
            im = axes[i].imshow(matrix, cmap='RdBu_r', aspect='auto')
            axes[i].set_title(f'{class_name}\n{band_name} {feat_name}')
            axes[i].set_xlabel('Channel (Brain 2)')
            axes[i].set_ylabel('Channel (Brain 1)')
            plt.colorbar(im, ax=axes[i], fraction=0.046)

        plt.tight_layout()
        plt.savefig(output_path / f'ibs_mean_{band_name}_{feat_name}.png', dpi=150)
        plt.close()

        # Plot difference matrix
        if 'Cooperation' in mean_matrices and 'Competition' in mean_matrices:
            fig, ax = plt.subplots(figsize=(8, 6))
            diff = diff_matrix[band_idx, feat_idx]
            vmax = np.abs(diff).max()
            im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Cooperation - Competition\n{band_name} {feat_name}')
            ax.set_xlabel('Channel (Brain 2)')
            ax.set_ylabel('Channel (Brain 1)')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(output_path / f'ibs_diff_coop_vs_comp_{band_name}_{feat_name}.png', dpi=150)
            plt.close()

    print(f"[Analysis] IBS connectivity results saved to {output_path}")


def run_embedding_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    class_names: List[str] = None
) -> None:
    """
    Run t-SNE and UMAP embedding analysis.

    Args:
        model: Trained model
        dataloader: Test DataLoader
        device: Device
        output_dir: Output directory
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("Running Embedding Analysis (t-SNE/UMAP)")
    print("=" * 60)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Extract features
    features = _metrics.extract_features_for_embedding(model, dataloader, device)

    output_path = output_dir / 'feature_embeddings'
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each feature type
    feature_types = ['z_fuse', 'ibs_token']

    for feat_type in feature_types:
        if feat_type not in features:
            continue

        feat = features[feat_type]
        y_true = features['y_true']
        y_pred = features['y_pred']

        print(f"[Analysis] Processing {feat_type} (shape: {feat.shape})")

        # t-SNE
        print(f"  Computing t-SNE...")
        tsne_coords = _metrics.compute_tsne(feat)

        if tsne_coords is not None:
            tsne_df = pd.DataFrame({
                'Sample_ID': list(range(len(y_true))),
                'True_Label': [class_names[y] for y in y_true],
                'Pred_Label': [class_names[y] for y in y_pred],
                'TSNE_1': tsne_coords[:, 0],
                'TSNE_2': tsne_coords[:, 1]
            })
            _io_utils.save_embedding_results(tsne_df, output_path / f'tsne_{feat_type}.csv')

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for i, class_name in enumerate(class_names):
                mask = y_true == i
                ax.scatter(
                    tsne_coords[mask, 0],
                    tsne_coords[mask, 1],
                    c=colors[i],
                    label=class_name,
                    alpha=0.6,
                    s=20
                )
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title(f't-SNE of {feat_type}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_path / f'tsne_{feat_type}.png', dpi=150)
            plt.close()

        # UMAP
        print(f"  Computing UMAP...")
        umap_coords = _metrics.compute_umap(feat)

        if umap_coords is not None:
            umap_df = pd.DataFrame({
                'Sample_ID': list(range(len(y_true))),
                'True_Label': [class_names[y] for y in y_true],
                'Pred_Label': [class_names[y] for y in y_pred],
                'UMAP_1': umap_coords[:, 0],
                'UMAP_2': umap_coords[:, 1]
            })
            _io_utils.save_embedding_results(umap_df, output_path / f'umap_{feat_type}.csv')

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            for i, class_name in enumerate(class_names):
                mask = y_true == i
                ax.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    c=colors[i],
                    label=class_name,
                    alpha=0.6,
                    s=20
                )
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'UMAP of {feat_type}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_path / f'umap_{feat_type}.png', dpi=150)
            plt.close()

    print(f"[Analysis] Embedding results saved to {output_path}")


def run_attention_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    class_names: List[str] = None
) -> None:
    """
    Run cross-attention weight analysis.
    """
    print("\n" + "=" * 60)
    print("Running Attention Analysis")
    print("=" * 60)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Extract weights
    # Returns {'mean_map': np.ndarray, 'diagonal_stats': Dict}
    results = _metrics.extract_attention_weights(
        model, dataloader, device, max_samples=200
    )

    if not results:
        print("[Warning] No attention weights extracted.")
        return

    output_path = output_dir / 'attention_weights'
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Visualize Mean Attention Map
    if results.get('mean_map') is not None:
        mean_map = results['mean_map']
        
        # Save raw
        np.save(output_path / 'mean_attention_map.npy', mean_map)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 7))
        # Downsample for visualization if too large
        if mean_map.shape[0] > 100:
            # Simple decimation
            viz_map = mean_map[::2, ::2] 
        else:
            viz_map = mean_map
            
        im = ax.imshow(viz_map, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_title('Mean Cross-Attention Map (Z1 <-> Z2)')
        ax.set_xlabel('Key Sequence Position')
        ax.set_ylabel('Query Sequence Position')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_path / 'mean_attention_heatmap.png', dpi=150)
        plt.close()

    # 2. Visualize Diagonal Profiles (Synchronization)
    diag_stats = results.get('diagonal_stats', {})
    if diag_stats:
        # Save stats to CSV
        rows = []
        for name, stats in diag_stats.items():
            rows.append({
                'Class': name,
                'Mean_Diagonal_Value': stats['mean_diagonal_value'],
                'Sample_Count': stats['count']
            })
        pd.DataFrame(rows).to_csv(output_path / 'attention_diagonal_summary.csv', index=False)
        
        # Plot profiles
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, stats in diag_stats.items():
            profile = stats['mean_diagonal_vector']
            x = np.arange(len(profile))
            # Smooth slightly for visualization
            ax.plot(x, profile, label=f'{name} (Mean={stats["mean_diagonal_value"]:.4f})', alpha=0.8)
            
        ax.set_title('Cross-Attention Diagonal Profile (Time Synchronization)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Weight')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'attention_diagonal_profile.png', dpi=150)
        plt.close()

    print(f"[Analysis] Attention analysis results saved to {output_path}")


def run_gradcam_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    class_names: List[str] = None
) -> None:
    """
    Run Grad-CAM analysis on spectrogram.
    """
    print("\n" + "=" * 60)
    print("Running Grad-CAM Analysis")
    print("=" * 60)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Compute Grad-CAM
    # Returns Dict[class_idx, heatmap_array]
    cams = _metrics.compute_gradcam_batch(
        model, dataloader, device, target_class=None, max_samples=200
    )

    if not cams:
        print("[Warning] No Grad-CAM results generated. Check model architecture.")
        return

    output_path = output_dir / 'gradcam'
    output_path.mkdir(parents=True, exist_ok=True)

    # Visualization
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5))
    if len(class_names) == 1:
        axes = [axes]

    for i, class_name in enumerate(class_names):
        if i in cams:
            heatmap = cams[i]
            
            # Save raw data
            np.savetxt(output_path / f'gradcam_{class_name}.csv', heatmap, delimiter=',')
            
            # Plot
            ax = axes[i]
            # Heatmap (Freq x Time)
            # Origin lower because frequency usually goes up
            im = ax.imshow(heatmap, cmap='jet', aspect='auto', origin='lower')
            ax.set_title(f'{class_name}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            plt.colorbar(im, ax=ax)
        else:
            axes[i].text(0.5, 0.5, 'No samples', ha='center')
            axes[i].set_title(f'{class_name}')

    plt.suptitle('Average Grad-CAM by Class (Spectrogram)')
    plt.tight_layout()
    plt.savefig(output_path / 'gradcam_average.png', dpi=150)
    plt.close()

    print(f"[Analysis] Grad-CAM results saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze trained EEG model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (best_model.pt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='7_Analysis/outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--analyses',
        type=str,
        default='all',
        help='Comma-separated list of analyses to run: metrics,frequency,ibs,embedding,attention,gradcam'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device to run analysis on'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EEG Model Analysis")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Analyses: {args.analyses}")

    # Load model and config
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    model, config = load_model_and_config(str(checkpoint_path), device)

    # Create dataloader
    dataloader = create_test_dataloader(config, args.batch_size)

    # Determine which analyses to run
    if args.analyses.lower() == 'all':
        analyses = ['metrics', 'frequency', 'ibs', 'embedding', 'attention', 'gradcam']
    else:
        analyses = [a.strip().lower() for a in args.analyses.split(',')]

    # Run analyses
    class_names = config.get('data', {}).get('class_names', DEFAULT_CLASS_NAMES)

    baseline_results = None

    if 'metrics' in analyses:
        baseline_results = run_core_metrics_analysis(
            model, dataloader, device, output_dir, class_names
        )

    if 'frequency' in analyses:
        if baseline_results is None:
            # Need to run metrics first to get baseline
            baseline_results = run_core_metrics_analysis(
                model, dataloader, device, output_dir, class_names
            )
        run_frequency_sensitivity_analysis(
            model, dataloader, device, output_dir,
            baseline_results['accuracy'],
            baseline_results['f1']
        )

    if 'ibs' in analyses:
        run_ibs_connectivity_analysis(
            model, dataloader, device, output_dir, class_names
        )

    if 'embedding' in analyses:
        run_embedding_analysis(
            model, dataloader, device, output_dir, class_names
        )

    if 'attention' in analyses:
        run_attention_analysis(
            model, dataloader, device, output_dir, class_names
        )

    if 'gradcam' in analyses:
        run_gradcam_analysis(
            model, dataloader, device, output_dir, class_names
        )

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
