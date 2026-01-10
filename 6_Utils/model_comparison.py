"""
Multi-Model Comparison Module

Provides tools for comparing multiple trained models:
- Metric comparison tables
- Side-by-side visualizations
- Statistical significance testing

Author: CNElab
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CLASS_NAMES = ["Single", "Competition", "Cooperation"]

DEFAULT_COLORS = {
    "Single": "#2196F3",       # Blue
    "Competition": "#F44336",  # Red
    "Cooperation": "#4CAF50"   # Green
}


# =============================================================================
# Model Results Container
# =============================================================================

class ModelResults:
    """
    Container for storing model evaluation results.

    Attributes
    ----------
    name : str
        Model display name
    labels : np.ndarray
        Ground truth labels
    predictions : np.ndarray
        Model predictions
    probabilities : np.ndarray
        Prediction probabilities
    metrics : dict
        Computed metrics
    """

    def __init__(
        self,
        name: str,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        self.name = name
        self.labels = labels
        self.predictions = predictions
        self.probabilities = probabilities
        self.metrics = metrics or {}

    def accuracy(self) -> float:
        """Compute accuracy."""
        return (self.labels == self.predictions).mean()


# =============================================================================
# Multi-Model Comparator
# =============================================================================

class MultiModelComparator:
    """
    Compare multiple trained models side by side.

    Generates comparison tables, visualizations, and statistical tests
    to evaluate relative model performance.

    Parameters
    ----------
    results : list of ModelResults
        List of model results to compare
    class_names : list of str, optional
        Class names for labeling

    Examples
    --------
    >>> results1 = ModelResults('Early-Concat', labels, preds1, probs1, metrics1)
    >>> results2 = ModelResults('Late-Full', labels, preds2, probs2, metrics2)
    >>> comparator = MultiModelComparator([results1, results2])
    >>> comparison_df = comparator.compare_metrics()
    """

    def __init__(
        self,
        results: List[ModelResults],
        class_names: List[str] = None
    ):
        self.results = results
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.n_classes = len(self.class_names)

    @property
    def model_names(self) -> List[str]:
        """Get list of model names."""
        return [r.name for r in self.results]

    def compare_metrics(
        self,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate comparison table of metrics.

        Parameters
        ----------
        metrics : list of str, optional
            Specific metrics to include. If None, includes all available.

        Returns
        -------
        pd.DataFrame
            Comparison table with models as rows, metrics as columns
        """
        all_metrics = []

        for result in self.results:
            row = {'model': result.name}

            if metrics is None:
                # Include all available metrics
                row.update(result.metrics)
            else:
                # Include only specified metrics
                for m in metrics:
                    if m in result.metrics:
                        row[m] = result.metrics[m]

            # Always compute accuracy if not present
            if 'accuracy' not in row:
                row['accuracy'] = result.accuracy()

            all_metrics.append(row)

        df = pd.DataFrame(all_metrics)

        # Reorder columns: model first, then sorted metrics
        cols = ['model'] + sorted([c for c in df.columns if c != 'model'])
        return df[cols]

    def rank_models(
        self,
        metric: str = 'accuracy',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank models by a specific metric.

        Parameters
        ----------
        metric : str
            Metric to rank by
        ascending : bool
            If True, lower is better

        Returns
        -------
        pd.DataFrame
            Ranked comparison table
        """
        df = self.compare_metrics()

        if metric not in df.columns:
            print(f"  Warning: Metric '{metric}' not found")
            return df

        df = df.sort_values(metric, ascending=ascending)
        df['rank'] = range(1, len(df) + 1)

        return df

    def compute_statistical_significance(
        self,
        metric: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Compute pairwise statistical significance tests.

        Uses McNemar's test for comparing classifier predictions.

        Parameters
        ----------
        metric : str
            Not directly used, comparison is on predictions

        Returns
        -------
        pd.DataFrame
            Pairwise p-values matrix
        """
        n_models = len(self.results)
        p_values = np.ones((n_models, n_models))

        for i in range(n_models):
            for j in range(i + 1, n_models):
                # McNemar's test contingency table
                pred_i = self.results[i].predictions
                pred_j = self.results[j].predictions
                labels = self.results[i].labels

                # Correct/incorrect by each model
                correct_i = pred_i == labels
                correct_j = pred_j == labels

                # Contingency table
                # b: i correct, j incorrect
                # c: i incorrect, j correct
                b = np.sum(correct_i & ~correct_j)
                c = np.sum(~correct_i & correct_j)

                # McNemar's test (with continuity correction)
                if b + c > 0:
                    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                    p_value = 1 - stats.chi2.cdf(chi2, df=1)
                else:
                    p_value = 1.0

                p_values[i, j] = p_value
                p_values[j, i] = p_value

        return pd.DataFrame(
            p_values,
            index=self.model_names,
            columns=self.model_names
        )

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_metrics_comparison(
        self,
        save_path: Path,
        metrics: Optional[List[str]] = None,
        title: str = "Model Performance Comparison"
    ):
        """
        Plot grouped bar chart comparing key metrics.

        Parameters
        ----------
        save_path : Path
            Output file path
        metrics : list of str, optional
            Metrics to compare
        title : str
            Plot title
        """
        df = self.compare_metrics(metrics)

        if metrics is None:
            # Default metrics to plot
            metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            print("  Warning: No plottable metrics found")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(available_metrics))
        width = 0.8 / len(self.model_names)
        colors = plt.cm.tab10.colors

        for idx, result in enumerate(self.results):
            values = [df[df['model'] == result.name][m].values[0]
                     for m in available_metrics]
            offset = (idx - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=result.name,
                         color=colors[idx % len(colors)], alpha=0.8)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(
                    f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=8
                )

        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved metrics comparison to {save_path}")

    def plot_confusion_matrices(
        self,
        save_path: Path,
        title: str = "Confusion Matrix Comparison"
    ):
        """
        Plot side-by-side confusion matrices.

        Parameters
        ----------
        save_path : Path
            Output file path
        title : str
            Plot title
        """
        from sklearn.metrics import confusion_matrix

        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, result in enumerate(self.results):
            cm = confusion_matrix(
                result.labels, result.predictions,
                labels=range(self.n_classes)
            )

            ax = axes[idx]
            sns.heatmap(
                cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            ax.set_title(f'{result.name}\nAcc: {result.accuracy():.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved confusion matrices to {save_path}")

    def plot_roc_comparison(
        self,
        save_path: Path,
        title: str = "ROC Curve Comparison"
    ):
        """
        Plot overlaid ROC curves from all models.

        Parameters
        ----------
        save_path : Path
            Output file path
        title : str
            Plot title
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10.colors

        for idx, result in enumerate(self.results):
            if result.probabilities is None:
                continue

            # Compute micro-average ROC
            y_true_bin = label_binarize(result.labels, classes=range(self.n_classes))
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), result.probabilities.ravel())
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr, tpr,
                label=f'{result.name} (AUC={roc_auc:.3f})',
                color=colors[idx % len(colors)],
                linewidth=2
            )

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved ROC comparison to {save_path}")

    def plot_per_class_comparison(
        self,
        save_path: Path,
        metric: str = 'f1',
        title: str = "Per-Class Performance Comparison"
    ):
        """
        Plot per-class metric comparison.

        Parameters
        ----------
        save_path : Path
            Output file path
        metric : str
            Metric to compare ('f1', 'precision', 'recall')
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(self.class_names))
        width = 0.8 / len(self.model_names)
        colors = plt.cm.tab10.colors

        for idx, result in enumerate(self.results):
            values = []
            for class_name in self.class_names:
                key = f'{metric}_{class_name}'
                if key in result.metrics:
                    values.append(result.metrics[key])
                else:
                    values.append(0)

            offset = (idx - len(self.model_names) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=result.name,
                  color=colors[idx % len(colors)], alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved per-class comparison to {save_path}")

    def plot_radar_chart(
        self,
        save_path: Path,
        metrics: Optional[List[str]] = None,
        title: str = "Model Comparison Radar Chart"
    ):
        """
        Plot radar chart comparing models across metrics.

        Parameters
        ----------
        save_path : Path
            Output file path
        metrics : list of str, optional
            Metrics to include
        title : str
            Plot title
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

        df = self.compare_metrics(metrics)
        available_metrics = [m for m in metrics if m in df.columns]

        if len(available_metrics) < 3:
            print("  Warning: Need at least 3 metrics for radar chart")
            return

        n_metrics = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.tab10.colors

        for idx, result in enumerate(self.results):
            values = [df[df['model'] == result.name][m].values[0]
                     for m in available_metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=result.name,
                   color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n').title() for m in available_metrics])
        ax.set_ylim(0, 1)
        ax.set_title(title, y=1.08, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved radar chart to {save_path}")

    # =========================================================================
    # CSV Export
    # =========================================================================

    def save_comparison_csv(self, save_path: Path):
        """Save comparison table to CSV."""
        df = self.compare_metrics()
        df.to_csv(save_path, index=False)
        print(f"  Saved comparison table to {save_path}")

    def save_significance_csv(self, save_path: Path):
        """Save statistical significance matrix to CSV."""
        df = self.compute_statistical_significance()
        df.to_csv(save_path)
        print(f"  Saved significance tests to {save_path}")

    def generate_latex_table(
        self,
        metrics: Optional[List[str]] = None,
        caption: str = "Model Performance Comparison",
        label: str = "tab:model_comparison"
    ) -> str:
        """
        Generate LaTeX table code for the comparison.

        Parameters
        ----------
        metrics : list of str, optional
            Metrics to include
        caption : str
            Table caption
        label : str
            Table label

        Returns
        -------
        str
            LaTeX table code
        """
        df = self.compare_metrics(metrics)

        # Build LaTeX table
        n_cols = len(df.columns)
        col_format = 'l' + 'c' * (n_cols - 1)

        lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            f'\\begin{{tabular}}{{{col_format}}}',
            '\\toprule'
        ]

        # Header row
        headers = df.columns.tolist()
        header_line = ' & '.join([h.replace('_', ' ').title() for h in headers]) + ' \\\\'
        lines.append(header_line)
        lines.append('\\midrule')

        # Data rows
        for _, row in df.iterrows():
            values = []
            for i, val in enumerate(row):
                if i == 0:  # Model name
                    values.append(str(val))
                else:  # Metrics
                    values.append(f'{val:.4f}')
            lines.append(' & '.join(values) + ' \\\\')

        lines.extend([
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}'
        ])

        return '\n'.join(lines)


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=" * 60)
    print("Testing Model Comparison Module")
    print("=" * 60)

    # Generate synthetic results
    np.random.seed(42)
    n_samples = 300

    labels = np.random.randint(0, 3, n_samples)

    # Model 1: Good accuracy
    preds1 = labels.copy()
    error_idx = np.random.choice(n_samples, 30, replace=False)
    preds1[error_idx] = np.random.randint(0, 3, 30)
    probs1 = np.random.rand(n_samples, 3)
    probs1 = probs1 / probs1.sum(axis=1, keepdims=True)

    # Model 2: Lower accuracy
    preds2 = labels.copy()
    error_idx = np.random.choice(n_samples, 60, replace=False)
    preds2[error_idx] = np.random.randint(0, 3, 60)
    probs2 = np.random.rand(n_samples, 3)
    probs2 = probs2 / probs2.sum(axis=1, keepdims=True)

    # Create results
    results1 = ModelResults(
        'Early-Concat', labels, preds1, probs1,
        {'accuracy': (preds1 == labels).mean(), 'f1_macro': 0.85}
    )
    results2 = ModelResults(
        'Late-Full', labels, preds2, probs2,
        {'accuracy': (preds2 == labels).mean(), 'f1_macro': 0.78}
    )

    # Test comparator
    print("\n[1] Testing MultiModelComparator...")

    comparator = MultiModelComparator([results1, results2])

    comparison_df = comparator.compare_metrics()
    print(f"  Comparison table shape: {comparison_df.shape}")
    print(f"  Models: {comparator.model_names}")

    ranked = comparator.rank_models('accuracy')
    print(f"  Top model: {ranked.iloc[0]['model']}")

    sig_df = comparator.compute_statistical_significance()
    print(f"  Significance matrix shape: {sig_df.shape}")

    # Test LaTeX generation
    latex = comparator.generate_latex_table(['accuracy', 'f1_macro'])
    print(f"  LaTeX table lines: {len(latex.split(chr(10)))}")

    print("\n[2] Note: Visualization tests require matplotlib display")
    print("  Skipping plot generation in test mode...")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
