"""
Error Analysis and Mechanism Validation Module

Provides tools for analyzing model errors and validating interaction mechanisms:
- Pair-wise performance analysis
- Hard sample identification
- Spatial sensitivity analysis (Early Fusion)
- Feature correlation analysis (Late Fusion)
- Statistical hypothesis testing

Author: CNElab
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
from scipy import stats


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CLASS_NAMES = ["Single", "Competition", "Cooperation"]


# =============================================================================
# Error Analyzer
# =============================================================================

class ErrorAnalyzer:
    """
    Pair-wise error analysis for classification models.

    Analyzes model performance at the pair level, identifies hard pairs,
    and computes statistical summaries.

    Parameters
    ----------
    class_names : list of str, optional
        Names of classes. Default: ["Single", "Competition", "Cooperation"]

    Examples
    --------
    >>> analyzer = ErrorAnalyzer()
    >>> pair_stats = analyzer.analyze_pair_performance(predictions_df)
    >>> hard_pairs = analyzer.identify_hard_pairs(pair_stats)
    """

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or DEFAULT_CLASS_NAMES

    def analyze_pair_performance(
        self,
        predictions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute per-pair accuracy statistics.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with columns: pair_id, true_label, pred_label, correct

        Returns
        -------
        pd.DataFrame
            Pair statistics with columns: pair_id, correct_count, total_count, accuracy
        """
        pair_stats = predictions_df.groupby('pair_id').agg({
            'correct': ['sum', 'count', 'mean']
        }).reset_index()

        pair_stats.columns = ['pair_id', 'correct_count', 'total_count', 'accuracy']
        pair_stats = pair_stats.sort_values('accuracy')

        return pair_stats

    def identify_hard_pairs(
        self,
        pair_stats: pd.DataFrame,
        threshold_percentile: float = 20
    ) -> List[int]:
        """
        Identify pairs with lowest accuracy (hard pairs).

        Parameters
        ----------
        pair_stats : pd.DataFrame
            DataFrame from analyze_pair_performance()
        threshold_percentile : float
            Bottom X% are considered hard

        Returns
        -------
        list
            List of hard pair IDs
        """
        threshold = np.percentile(pair_stats['accuracy'], threshold_percentile)
        hard_pairs = pair_stats[pair_stats['accuracy'] <= threshold]['pair_id'].tolist()
        return hard_pairs

    def compute_pair_statistics(
        self,
        pair_stats: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute statistical summary for pair-wise analysis.

        Parameters
        ----------
        pair_stats : pd.DataFrame
            DataFrame from analyze_pair_performance()

        Returns
        -------
        dict
            Statistical summary with mean, std, min, max accuracy
        """
        stats_results = {
            'mean_accuracy': pair_stats['accuracy'].mean(),
            'std_accuracy': pair_stats['accuracy'].std(),
            'min_accuracy': pair_stats['accuracy'].min(),
            'max_accuracy': pair_stats['accuracy'].max(),
            'n_pairs': len(pair_stats),
            'hard_pairs_count': len(self.identify_hard_pairs(pair_stats))
        }

        return stats_results

    def analyze_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Analyze error distribution by class and confusion patterns.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        metadata : list of dict, optional
            Sample metadata

        Returns
        -------
        pd.DataFrame
            Error analysis DataFrame
        """
        errors = []

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                error = {
                    'sample_idx': i,
                    'true_label': true,
                    'true_name': self.class_names[true],
                    'pred_label': pred,
                    'pred_name': self.class_names[pred],
                    'confusion_type': f'{self.class_names[true]}_as_{self.class_names[pred]}'
                }

                if metadata:
                    error.update(metadata[i])

                errors.append(error)

        return pd.DataFrame(errors)

    def compute_confusion_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze confusion patterns between classes.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels

        Returns
        -------
        pd.DataFrame
            Confusion analysis with counts and percentages
        """
        confusions = []

        for i, true_name in enumerate(self.class_names):
            true_mask = y_true == i
            true_count = true_mask.sum()

            for j, pred_name in enumerate(self.class_names):
                if i != j:  # Only analyze errors
                    confusion_mask = true_mask & (y_pred == j)
                    confusion_count = confusion_mask.sum()

                    if true_count > 0:
                        confusions.append({
                            'true_class': true_name,
                            'pred_class': pred_name,
                            'count': confusion_count,
                            'percentage': confusion_count / true_count * 100
                        })

        return pd.DataFrame(confusions).sort_values('count', ascending=False)

    # =========================================================================
    # CSV Export Methods
    # =========================================================================

    def save_pair_stats_csv(
        self,
        pair_stats: pd.DataFrame,
        save_path: Path
    ):
        """Save pair statistics to CSV."""
        pair_stats.to_csv(save_path, index=False)
        print(f"  Saved pair statistics to {save_path}")

    def save_error_analysis_csv(
        self,
        error_df: pd.DataFrame,
        save_path: Path
    ):
        """Save error analysis to CSV."""
        error_df.to_csv(save_path, index=False)
        print(f"  Saved error analysis to {save_path}")


# =============================================================================
# Mechanism Analyzer
# =============================================================================

class MechanismAnalyzer:
    """
    Mechanism validation for gaze interaction models.

    Analyzes spatial sensitivity (Early Fusion) and feature correlation
    (Late Fusion) to validate model behavior.

    Parameters
    ----------
    model_type : str
        Model type: 'early' or 'late'
    class_names : list of str, optional
        Names of classes

    Examples
    --------
    >>> analyzer = MechanismAnalyzer('early')
    >>> spatial_df = analyzer.analyze_spatial_sensitivity(dataloader, predictions, labels)
    >>> stats = analyzer.statistical_tests_by_class(values, labels)
    """

    def __init__(
        self,
        model_type: str,
        class_names: List[str] = None
    ):
        self.model_type = model_type
        self.class_names = class_names or DEFAULT_CLASS_NAMES

    def compute_gaze_distance(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor
    ) -> float:
        """
        Compute distance between gaze centers of two heatmaps.

        Parameters
        ----------
        img_a, img_b : torch.Tensor
            Heatmap tensors, shape (3, H, W)

        Returns
        -------
        float
            Euclidean distance between centers of mass
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

    def compute_gaze_overlap(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        threshold: float = 0.1
    ) -> float:
        """
        Compute overlap ratio between two gaze heatmaps.

        Parameters
        ----------
        img_a, img_b : torch.Tensor
            Heatmap tensors, shape (3, H, W)
        threshold : float
            Threshold for binarizing heatmaps

        Returns
        -------
        float
            IoU (Intersection over Union) between heatmaps
        """
        # Convert to grayscale and normalize
        gray_a = img_a.mean(dim=0)
        gray_b = img_b.mean(dim=0)

        gray_a = (gray_a - gray_a.min()) / (gray_a.max() - gray_a.min() + 1e-8)
        gray_b = (gray_b - gray_b.min()) / (gray_b.max() - gray_b.min() + 1e-8)

        # Binarize
        mask_a = (gray_a > threshold).float()
        mask_b = (gray_b > threshold).float()

        # Compute IoU
        intersection = (mask_a * mask_b).sum()
        union = mask_a.sum() + mask_b.sum() - intersection

        if union < 1e-8:
            return 0.0

        return (intersection / union).item()

    def analyze_spatial_sensitivity(
        self,
        dataloader,
        predictions: np.ndarray,
        labels: np.ndarray,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze relationship between gaze spatial overlap and accuracy.

        For Early Fusion models: tests whether spatial proximity affects prediction.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader yielding batches
        predictions : np.ndarray
            Model predictions
        labels : np.ndarray
            Ground truth labels
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        pd.DataFrame
            DataFrame with distance, overlap, correct, label columns
        """
        results = []
        idx = 0

        iterator = tqdm(dataloader, desc="Analyzing spatial sensitivity") if show_progress else dataloader

        for batch in iterator:
            if len(batch) == 4:
                img_a, img_b, batch_labels, _ = batch
            else:
                img_a, img_b, batch_labels = batch

            for i in range(len(batch_labels)):
                distance = self.compute_gaze_distance(img_a[i], img_b[i])
                overlap = self.compute_gaze_overlap(img_a[i], img_b[i])

                results.append({
                    'distance': distance,
                    'overlap': overlap,
                    'correct': int(predictions[idx] == labels[idx]),
                    'label': labels[idx],
                    'label_name': self.class_names[labels[idx]]
                })
                idx += 1

        return pd.DataFrame(results)

    def compute_feature_correlation(
        self,
        model: nn.Module,
        dataloader,
        device: str = 'cuda',
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity between stream features (Late Fusion).

        Parameters
        ----------
        model : nn.Module
            Late Fusion ViT model
        dataloader : DataLoader
            DataLoader yielding batches
        device : str
            Computation device
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        similarities : np.ndarray
            Cosine similarity values, shape (N,)
        labels : np.ndarray
            Ground truth labels, shape (N,)
        """
        model.eval()
        similarities = []
        all_labels = []

        iterator = tqdm(dataloader, desc="Computing feature correlations") if show_progress else dataloader

        with torch.no_grad():
            for batch in iterator:
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

    def statistical_tests_by_class(
        self,
        values: np.ndarray,
        labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Perform statistical tests comparing values across classes.

        Includes one-way ANOVA and pairwise t-tests with Bonferroni correction.

        Parameters
        ----------
        values : np.ndarray
            Values to compare (e.g., distances, similarities)
        labels : np.ndarray
            Class labels

        Returns
        -------
        pd.DataFrame
            Statistical test results with p-values and effect sizes
        """
        results = []

        # Group by class
        class_groups = {
            name: values[labels == i]
            for i, name in enumerate(self.class_names)
        }

        # Check if all groups have data
        valid_groups = {k: v for k, v in class_groups.items() if len(v) > 0}
        if len(valid_groups) < 2:
            return pd.DataFrame()

        # ANOVA
        group_values = [class_groups[name] for name in self.class_names if len(class_groups[name]) > 0]
        f_stat, anova_p = stats.f_oneway(*group_values)

        # Eta-squared effect size
        grand_mean = values.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_values)
        ss_total = ((values - grand_mean)**2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        results.append({
            'comparison': 'ANOVA (all classes)',
            'statistic': f_stat,
            'p_value': anova_p,
            'significant': anova_p < 0.05,
            'effect_size': f'eta^2 = {eta_squared:.3f}'
        })

        # Pairwise t-tests with Bonferroni correction
        comparisons = [(0, 1), (0, 2), (1, 2)]
        alpha = 0.05 / len(comparisons)  # Bonferroni

        for i, j in comparisons:
            name_i, name_j = self.class_names[i], self.class_names[j]

            if len(class_groups[name_i]) == 0 or len(class_groups[name_j]) == 0:
                continue

            t_stat, p_value = stats.ttest_ind(
                class_groups[name_i], class_groups[name_j]
            )

            # Cohen's d effect size
            pooled_std = np.sqrt(
                (np.var(class_groups[name_i]) + np.var(class_groups[name_j])) / 2
            )
            if pooled_std > 0:
                cohen_d = (
                    np.mean(class_groups[name_i]) - np.mean(class_groups[name_j])
                ) / pooled_std
            else:
                cohen_d = 0.0

            results.append({
                'comparison': f'{name_i} vs {name_j}',
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'effect_size': f"Cohen's d = {cohen_d:.3f}"
            })

        return pd.DataFrame(results)

    def compute_class_statistics(
        self,
        values: np.ndarray,
        labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics per class.

        Parameters
        ----------
        values : np.ndarray
            Values to analyze
        labels : np.ndarray
            Class labels

        Returns
        -------
        pd.DataFrame
            Per-class statistics (mean, std, min, max, etc.)
        """
        stats_list = []

        for i, name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                class_values = values[mask]
                stats_list.append({
                    'class': name,
                    'n_samples': len(class_values),
                    'mean': class_values.mean(),
                    'std': class_values.std(),
                    'min': class_values.min(),
                    'max': class_values.max(),
                    'median': np.median(class_values),
                    'q25': np.percentile(class_values, 25),
                    'q75': np.percentile(class_values, 75)
                })

        return pd.DataFrame(stats_list)

    # =========================================================================
    # CSV Export Methods
    # =========================================================================

    def save_spatial_analysis_csv(
        self,
        spatial_df: pd.DataFrame,
        save_path: Path
    ):
        """Save spatial analysis to CSV."""
        spatial_df.to_csv(save_path, index=False)
        print(f"  Saved spatial analysis to {save_path}")

    def save_correlation_analysis_csv(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        save_path: Path
    ):
        """Save correlation analysis to CSV."""
        df = pd.DataFrame({
            'similarity': similarities,
            'label': labels,
            'label_name': [self.class_names[l] for l in labels]
        })
        df.to_csv(save_path, index=False)
        print(f"  Saved correlation analysis to {save_path}")

    def save_statistical_tests_csv(
        self,
        stats_df: pd.DataFrame,
        save_path: Path
    ):
        """Save statistical tests to CSV."""
        stats_df.to_csv(save_path, index=False)
        print(f"  Saved statistical tests to {save_path}")


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=" * 60)
    print("Testing Error Analysis Module")
    print("=" * 60)

    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 300
    n_pairs = 10

    y_true = np.random.randint(0, 3, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, 50, replace=False)
    y_pred[error_idx] = np.random.randint(0, 3, 50)

    pair_ids = np.random.randint(1, n_pairs + 1, n_samples)

    # Test ErrorAnalyzer
    print("\n[1] Testing ErrorAnalyzer...")

    error_analyzer = ErrorAnalyzer()

    predictions_df = pd.DataFrame({
        'pair_id': pair_ids,
        'true_label': y_true,
        'pred_label': y_pred,
        'correct': y_true == y_pred
    })

    pair_stats = error_analyzer.analyze_pair_performance(predictions_df)
    print(f"  Pair statistics shape: {pair_stats.shape}")

    hard_pairs = error_analyzer.identify_hard_pairs(pair_stats)
    print(f"  Hard pairs: {hard_pairs}")

    summary = error_analyzer.compute_pair_statistics(pair_stats)
    print(f"  Mean accuracy: {summary['mean_accuracy']:.4f}")

    # Test confusion analysis
    confusion_df = error_analyzer.compute_confusion_analysis(y_true, y_pred)
    print(f"  Top confusion: {confusion_df.iloc[0]['true_class']} -> {confusion_df.iloc[0]['pred_class']}")

    # Test MechanismAnalyzer
    print("\n[2] Testing MechanismAnalyzer...")

    mechanism_analyzer = MechanismAnalyzer('early')

    # Mock spatial values
    distances = np.random.rand(n_samples) * 100
    labels = y_true

    stats_df = mechanism_analyzer.statistical_tests_by_class(distances, labels)
    print(f"  Statistical tests: {len(stats_df)} comparisons")

    class_stats = mechanism_analyzer.compute_class_statistics(distances, labels)
    print(f"  Class statistics: {len(class_stats)} classes")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
