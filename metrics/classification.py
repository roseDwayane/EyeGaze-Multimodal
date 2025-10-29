"""
Classification metrics for model evaluation
Compatible with Hugging Face Trainer
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred, id2label: Optional[Dict[int, str]] = None) -> Dict[str, float]:
    """
    Compute classification metrics for Hugging Face Trainer

    Args:
        eval_pred: EvalPrediction object containing predictions and labels
        id2label: Optional mapping from label ID to label name

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred

    # Get predicted class (argmax of logits)
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)

    # Precision, Recall, F1 (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro',
        zero_division=0
    )

    # Per-class metrics (weighted average)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
    }

    # Log detailed results
    logger.info(f"\nEvaluation Metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (macro): {precision:.4f}")
    logger.info(f"Recall (macro): {recall:.4f}")
    logger.info(f"F1 (macro): {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Classification report (if label names available)
    if id2label is not None:
        target_names = [id2label[i] for i in sorted(id2label.keys())]
        report = classification_report(
            labels,
            predictions,
            target_names=target_names,
            digits=4
        )
        logger.info(f"\nClassification Report:\n{report}")

    return metrics


def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix

    Args:
        predictions: Predicted labels
        labels: True labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(labels, predictions)


def compute_per_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: Optional[Dict[int, str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics

    Args:
        predictions: Predicted labels
        labels: True labels
        id2label: Optional mapping from label ID to label name

    Returns:
        Dictionary of per-class metrics
    """
    # Get unique classes
    unique_classes = np.unique(np.concatenate([predictions, labels]))

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=unique_classes,
        average=None,
        zero_division=0
    )

    per_class_metrics = {}

    for i, class_id in enumerate(unique_classes):
        class_name = id2label[class_id] if id2label else f"Class_{class_id}"

        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    return per_class_metrics


def print_metrics_summary(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: Optional[Dict[int, str]] = None
):
    """
    Print a comprehensive metrics summary

    Args:
        predictions: Predicted labels
        labels: True labels
        id2label: Optional mapping from label ID to label name
    """
    print("=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)

    # Overall metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (macro)")
    print(f"  Recall:    {recall:.4f} (macro)")
    print(f"  F1 Score:  {f1:.4f} (macro)")

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    per_class = compute_per_class_metrics(predictions, labels, id2label)

    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-" * 80)

    for class_name, metrics in per_class.items():
        print(
            f"{class_name:<20} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f} "
            f"{metrics['support']:<12}"
        )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(f"\nConfusion Matrix:")

    if id2label:
        class_names = [id2label[i] for i in sorted(id2label.keys())]
        header = "True \\ Pred  " + "  ".join([f"{name[:8]:>8}" for name in class_names])
        print(header)
        print("-" * len(header))

        for i, class_name in enumerate(class_names):
            row = f"{class_name[:12]:<12}  " + "  ".join([f"{cm[i, j]:>8}" for j in range(len(class_names))])
            print(row)
    else:
        print(cm)

    print("=" * 80)


class MetricsTracker:
    """
    Track metrics across training epochs
    """

    def __init__(self):
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': [],
            'learning_rate': []
        }

    def update(self, metrics: Dict[str, float], prefix: str = 'eval'):
        """
        Update metrics history

        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for metric names ('train' or 'eval')
        """
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}"
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(value)

    def get_best_epoch(self, metric: str = 'eval_f1') -> int:
        """
        Get the epoch with the best metric value

        Args:
            metric: Metric name to use

        Returns:
            Best epoch index
        """
        if metric not in self.history:
            raise ValueError(f"Metric '{metric}' not found in history")

        values = self.history[metric]
        return int(np.argmax(values))

    def get_metric_history(self, metric: str) -> list:
        """
        Get the history of a specific metric

        Args:
            metric: Metric name

        Returns:
            List of metric values
        """
        return self.history.get(metric, [])

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked metrics

        Returns:
            Dictionary containing metric statistics
        """
        summary = {}

        for metric, values in self.history.items():
            if len(values) > 0:
                summary[metric] = {
                    'last': values[-1],
                    'best': max(values) if 'loss' not in metric else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }

        return summary
