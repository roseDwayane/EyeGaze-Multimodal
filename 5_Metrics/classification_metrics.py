"""
Classification Metrics Module

Provides comprehensive evaluation metrics for multi-class classification:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC Curves and AUC

Author: CNElab
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CLASS_NAMES = ["Single", "Competition", "Cooperation"]


# =============================================================================
# Classification Metrics Calculator
# =============================================================================

class ClassificationMetrics:
    """
    Comprehensive classification metrics calculator.

    Computes and saves various metrics for multi-class classification tasks.

    Parameters
    ----------
    class_names : list of str, optional
        Names of classes. Default: ["Single", "Competition", "Cooperation"]

    Attributes
    ----------
    class_names : list
        Class names
    n_classes : int
        Number of classes

    Examples
    --------
    >>> metrics_calc = ClassificationMetrics()
    >>> y_true = np.array([0, 1, 2, 0, 1])
    >>> y_pred = np.array([0, 1, 1, 0, 1])
    >>> metrics = metrics_calc.compute_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.n_classes = len(self.class_names)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels, shape (N,)
        y_pred : np.ndarray
            Predicted labels, shape (N,)

        Returns
        -------
        dict
            Dictionary containing:
            - accuracy: Overall accuracy
            - precision_macro/weighted: Precision scores
            - recall_macro/weighted: Recall scores
            - f1_macro/weighted: F1 scores
            - precision_{class}: Per-class precision
            - recall_{class}: Per-class recall
            - f1_{class}: Per-class F1
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(
                y_true, y_pred, average='macro', zero_division=0
            ),
            'precision_weighted': precision_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'recall_macro': recall_score(
                y_true, y_pred, average='macro', zero_division=0
            ),
            'recall_weighted': recall_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
            'f1_macro': f1_score(
                y_true, y_pred, average='macro', zero_division=0
            ),
            'f1_weighted': f1_score(
                y_true, y_pred, average='weighted', zero_division=0
            ),
        }

        # Per-class metrics
        for i, name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            metrics[f'precision_{name}'] = precision_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
            metrics[f'recall_{name}'] = recall_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
            metrics[f'f1_{name}'] = f1_score(
                y_true_binary, y_pred_binary, zero_division=0
            )

        return metrics

    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels

        Returns
        -------
        np.ndarray
            Confusion matrix of shape (n_classes, n_classes)
        """
        return confusion_matrix(y_true, y_pred, labels=range(self.n_classes))

    def compute_roc_data(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute ROC curve data for multi-class classification.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels, shape (N,)
        y_prob : np.ndarray
            Predicted probabilities, shape (N, n_classes)

        Returns
        -------
        dict
            Dictionary with ROC data for each class and micro/macro averages:
            - {class_name}: {'fpr', 'tpr', 'thresholds', 'auc'}
            - 'micro': {'fpr', 'tpr', 'auc'}
            - 'macro': {'fpr', 'tpr', 'auc'}
        """
        # Binarize labels for one-vs-rest
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))

        # Handle binary classification case
        if self.n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

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
        fpr_micro, tpr_micro, _ = roc_curve(
            y_true_bin.ravel(), y_prob.ravel()
        )
        roc_data['micro'] = {
            'fpr': fpr_micro,
            'tpr': tpr_micro,
            'auc': auc(fpr_micro, tpr_micro)
        }

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate(
            [roc_data[name]['fpr'] for name in self.class_names]
        ))
        mean_tpr = np.zeros_like(all_fpr)
        for name in self.class_names:
            mean_tpr += np.interp(
                all_fpr, roc_data[name]['fpr'], roc_data[name]['tpr']
            )
        mean_tpr /= self.n_classes

        roc_data['macro'] = {
            'fpr': all_fpr,
            'tpr': mean_tpr,
            'auc': auc(all_fpr, mean_tpr)
        }

        return roc_data

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get formatted classification report.

        Returns
        -------
        str
            Formatted classification report string
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )

    # =========================================================================
    # CSV Export Methods
    # =========================================================================

    def save_metrics_csv(
        self,
        metrics: Dict[str, float],
        save_path: Path
    ):
        """
        Save metrics dictionary to CSV file.

        Parameters
        ----------
        metrics : dict
            Metrics dictionary from compute_metrics()
        save_path : Path
            Output file path
        """
        df = pd.DataFrame([metrics]).T
        df.columns = ['value']
        df.index.name = 'metric'
        df.to_csv(save_path)
        print(f"  Saved metrics to {save_path}")

    def save_confusion_matrix_csv(
        self,
        cm: np.ndarray,
        save_path: Path
    ):
        """
        Save confusion matrix to CSV file.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix from compute_confusion_matrix()
        save_path : Path
            Output file path
        """
        df = pd.DataFrame(
            cm,
            index=[f'True_{name}' for name in self.class_names],
            columns=[f'Pred_{name}' for name in self.class_names]
        )
        df.to_csv(save_path)
        print(f"  Saved confusion matrix to {save_path}")

    def save_roc_data_csv(
        self,
        roc_data: Dict[str, Any],
        save_path: Path
    ):
        """
        Save ROC curve data to CSV file.

        Parameters
        ----------
        roc_data : dict
            ROC data from compute_roc_data()
        save_path : Path
            Output file path
        """
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

    def save_predictions_csv(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        save_path: Path,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Save predictions with probabilities to CSV.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        y_prob : np.ndarray
            Predicted probabilities
        save_path : Path
            Output file path
        metadata : list of dict, optional
            Additional metadata per sample
        """
        df = pd.DataFrame({
            'true_label': y_true,
            'true_label_name': [self.class_names[l] for l in y_true],
            'pred_label': y_pred,
            'pred_label_name': [self.class_names[l] for l in y_pred],
            'correct': y_true == y_pred,
            'confidence': y_prob.max(axis=1)
        })

        # Add per-class probabilities
        for i, name in enumerate(self.class_names):
            df[f'prob_{name}'] = y_prob[:, i]

        # Add metadata if provided
        if metadata:
            for key in metadata[0].keys():
                df[key] = [m.get(key, None) for m in metadata]

        df.to_csv(save_path, index=False)
        print(f"  Saved predictions to {save_path}")


# =============================================================================
# Utility Functions
# =============================================================================

def compute_class_weights(
    labels: np.ndarray,
    n_classes: int = 3
) -> np.ndarray:
    """
    Compute class weights for imbalanced datasets.

    Uses inverse frequency weighting.

    Parameters
    ----------
    labels : np.ndarray
        Array of class labels
    n_classes : int
        Number of classes

    Returns
    -------
    np.ndarray
        Class weights, shape (n_classes,)
    """
    counts = np.bincount(labels, minlength=n_classes)
    weights = len(labels) / (n_classes * counts + 1e-6)
    return weights / weights.sum() * n_classes


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute accuracy for each class separately.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Class names

    Returns
    -------
    dict
        Per-class accuracy
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc[name] = (y_pred[mask] == i).mean()
        else:
            per_class_acc[name] = 0.0

    return per_class_acc


if __name__ == '__main__':
    # Quick test
    print("=" * 60)
    print("Testing Classification Metrics")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.randint(0, 3, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, 20, replace=False)
    y_pred[error_idx] = np.random.randint(0, 3, 20)

    # Generate probabilities
    y_prob = np.random.rand(n_samples, 3)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    # Test metrics
    calc = ClassificationMetrics()

    metrics = calc.compute_metrics(y_true, y_pred)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (Macro): {metrics['f1_macro']:.4f}")

    cm = calc.compute_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    roc_data = calc.compute_roc_data(y_true, y_prob)
    print(f"\nROC AUC (Micro): {roc_data['micro']['auc']:.4f}")
    print(f"ROC AUC (Macro): {roc_data['macro']['auc']:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
