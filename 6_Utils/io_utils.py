"""
CSV Export Utilities for EEG Analysis

This module provides functions to export analysis results to CSV format
for subsequent processing in MATLAB or other tools.

Author: Analysis Pipeline for EyeGaze-Multimodal Project
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def ensure_output_dirs(base_path: str) -> Dict[str, Path]:
    """
    Create all necessary output subdirectories.

    Args:
        base_path: Base output directory path

    Returns:
        Dictionary mapping directory names to Path objects
    """
    base = Path(base_path)

    subdirs = {
        'core_metrics': base / 'core_metrics',
        'frequency_sensitivity': base / 'frequency_sensitivity',
        'ibs_connectivity': base / 'ibs_connectivity',
        'ibs_mean_by_class': base / 'ibs_connectivity' / 'ibs_mean_by_class',
        'ibs_difference': base / 'ibs_connectivity' / 'ibs_difference_coop_vs_comp',
        'attention_weights': base / 'attention_weights',
        'attention_mean_by_class': base / 'attention_weights' / 'attention_mean_by_class',
        'feature_embeddings': base / 'feature_embeddings',
        'gradcam': base / 'gradcam',
        'gradcam_mean_by_class': base / 'gradcam' / 'gradcam_mean_by_class',
    }

    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)

    return subdirs


def save_confusion_matrix(
    cm: np.ndarray,
    path: Union[str, Path],
    class_names: List[str]
) -> None:
    """
    Save confusion matrix to CSV.

    Args:
        cm: Confusion matrix array (N x N)
        path: Output file path
        class_names: List of class names for row/column labels

    Output CSV format:
        Predicted_Single, Predicted_Competition, Predicted_Cooperation
        (rows are True labels)
    """
    # Create DataFrame with labeled rows and columns
    columns = [f'Predicted_{name}' for name in class_names]
    index = [f'True_{name}' for name in class_names]

    df = pd.DataFrame(cm, index=index, columns=columns)
    df.to_csv(path)
    print(f"[io_utils] Saved confusion matrix to {path}")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Union[str, Path],
    class_names: List[str]
) -> pd.DataFrame:
    """
    Save per-class classification metrics to CSV.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        path: Output file path
        class_names: List of class names

    Returns:
        DataFrame with metrics

    Output CSV format:
        Class, Precision, Recall, F1, Support
    """
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Support': support.astype(int)
    })

    df.to_csv(path, index=False)
    print(f"[io_utils] Saved classification report to {path}")
    return df


def save_overall_metrics(
    metrics: Dict[str, float],
    path: Union[str, Path]
) -> None:
    """
    Save overall metrics to CSV.

    Args:
        metrics: Dictionary of metric names to values
        path: Output file path

    Output CSV format:
        Metric, Value
    """
    df = pd.DataFrame([
        {'Metric': k, 'Value': v} for k, v in metrics.items()
    ])
    df.to_csv(path, index=False)
    print(f"[io_utils] Saved overall metrics to {path}")


def save_frequency_sensitivity(
    sensitivity_data: Dict[str, Dict[str, float]],
    path: Union[str, Path],
    baseline_accuracy: float,
    baseline_f1: float
) -> None:
    """
    Save frequency band sensitivity analysis results.

    Args:
        sensitivity_data: Dict mapping band name to {'accuracy': ..., 'f1': ...}
        path: Output file path
        baseline_accuracy: Original model accuracy (no masking)
        baseline_f1: Original model F1 score

    Output CSV format:
        Band, Masked_Accuracy, Masked_F1, Accuracy_Drop, F1_Drop
    """
    rows = []
    for band, metrics in sensitivity_data.items():
        rows.append({
            'Band': band,
            'Masked_Accuracy': metrics['accuracy'],
            'Masked_F1': metrics['f1'],
            'Accuracy_Drop': baseline_accuracy - metrics['accuracy'],
            'F1_Drop': baseline_f1 - metrics['f1']
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[io_utils] Saved frequency sensitivity to {path}")


def save_ibs_long_format(
    ibs_data: List[Dict],
    path: Union[str, Path]
) -> None:
    """
    Save IBS connectivity data in flattened long format.

    Args:
        ibs_data: List of dicts with keys:
            'subject_id', 'true_label', 'pred_label', 'band', 'feature',
            'channel_1', 'channel_2', 'value'
        path: Output file path

    Output CSV format:
        Subject_ID, True_Label, Pred_Label, Band, Feature, Channel_1, Channel_2, Value
    """
    df = pd.DataFrame(ibs_data)
    df.to_csv(path, index=False)
    print(f"[io_utils] Saved IBS long format ({len(df)} rows) to {path}")


def save_ibs_matrix(
    matrix: np.ndarray,
    path: Union[str, Path]
) -> None:
    """
    Save a single IBS connectivity matrix (32x32) as CSV.
    No header, pure numeric values for MATLAB readmatrix().

    Args:
        matrix: 2D array (C x C)
        path: Output file path
    """
    np.savetxt(path, matrix, delimiter=',', fmt='%.6f')
    print(f"[io_utils] Saved IBS matrix {matrix.shape} to {path}")


def save_channel_names(
    channel_names: List[str],
    path: Union[str, Path]
) -> None:
    """
    Save channel name mapping.

    Args:
        channel_names: List of channel names
        path: Output file path

    Output CSV format:
        Index, Channel_Name
    """
    df = pd.DataFrame({
        'Index': list(range(len(channel_names))),
        'Channel_Name': channel_names
    })
    df.to_csv(path, index=False)
    print(f"[io_utils] Saved channel names to {path}")


def save_attention_summary(
    summary_data: List[Dict],
    path: Union[str, Path]
) -> None:
    """
    Save attention weight summary statistics.

    Args:
        summary_data: List of dicts with keys:
            'Class', 'Mean_Diagonal', 'Std_Diagonal', 'Mean_OffDiag', 'Std_OffDiag'
        path: Output file path
    """
    df = pd.DataFrame(summary_data)
    df.to_csv(path, index=False)
    print(f"[io_utils] Saved attention summary to {path}")


def save_attention_matrix(
    matrix: np.ndarray,
    path: Union[str, Path]
) -> None:
    """
    Save attention weight matrix as CSV.
    No header for MATLAB readmatrix().

    Args:
        matrix: 2D array (T x T)
        path: Output file path
    """
    np.savetxt(path, matrix, delimiter=',', fmt='%.6f')


def save_embedding_results(
    embedding_df: pd.DataFrame,
    path: Union[str, Path]
) -> None:
    """
    Save t-SNE/UMAP embedding results.

    Args:
        embedding_df: DataFrame with columns:
            'Sample_ID', 'True_Label', 'Pred_Label', 'Dim_1', 'Dim_2'
        path: Output file path
    """
    embedding_df.to_csv(path, index=False)
    print(f"[io_utils] Saved embedding results ({len(embedding_df)} samples) to {path}")


def save_gradcam_results(
    gradcam_matrix: np.ndarray,
    path: Union[str, Path]
) -> None:
    """
    Save Grad-CAM time-frequency heatmap as CSV.
    No header for MATLAB readmatrix().

    Args:
        gradcam_matrix: 2D array (F x T) - frequency bins x time bins
        path: Output file path
    """
    np.savetxt(path, gradcam_matrix, delimiter=',', fmt='%.6f')


def save_gradcam_metadata(
    freq_axis: np.ndarray,
    time_axis: np.ndarray,
    path: Union[str, Path],
    sampling_rate: int = 256,
    n_fft: int = 128,
    hop_length: int = 64
) -> None:
    """
    Save Grad-CAM axis metadata.

    Args:
        freq_axis: Frequency bin centers in Hz
        time_axis: Time bin centers in seconds
        path: Output file path
        sampling_rate: EEG sampling rate
        n_fft: FFT window size
        hop_length: STFT hop length
    """
    rows = []

    # Frequency axis
    for i, freq in enumerate(freq_axis):
        rows.append({
            'Axis': 'Frequency_Hz',
            'Index': i,
            'Value': freq
        })

    # Time axis
    for i, time in enumerate(time_axis):
        rows.append({
            'Axis': 'Time_Sec',
            'Index': i,
            'Value': time
        })

    # Metadata
    rows.append({'Axis': 'Param', 'Index': 'sampling_rate', 'Value': sampling_rate})
    rows.append({'Axis': 'Param', 'Index': 'n_fft', 'Value': n_fft})
    rows.append({'Axis': 'Param', 'Index': 'hop_length', 'Value': hop_length})

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[io_utils] Saved Grad-CAM metadata to {path}")


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    path: Union[str, Path],
    class_names: List[str],
    sample_ids: Optional[List] = None
) -> None:
    """
    Save all predictions with probabilities.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (N x num_classes), optional
        path: Output file path
        class_names: List of class names
        sample_ids: Optional sample identifiers
    """
    data = {
        'Sample_ID': sample_ids if sample_ids else list(range(len(y_true))),
        'True_Label': [class_names[int(y)] for y in y_true],
        'Pred_Label': [class_names[int(y)] for y in y_pred],
        'True_Label_ID': y_true,
        'Pred_Label_ID': y_pred,
        'Correct': (y_true == y_pred).astype(int)
    }

    if y_prob is not None:
        for i, name in enumerate(class_names):
            data[f'Prob_{name}'] = y_prob[:, i]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"[io_utils] Saved predictions ({len(df)} samples) to {path}")
