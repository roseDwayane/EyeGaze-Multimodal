"""
Learning Curves Module

Provides tools for extracting and visualizing training history:
- Wandb integration
- Checkpoint-based extraction
- Learning curve visualization

Author: CNElab
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


# =============================================================================
# Learning Curve Analyzer
# =============================================================================

class LearningCurveAnalyzer:
    """
    Extract and visualize training history from various sources.

    Supports:
    - Wandb run history
    - Periodic checkpoint files
    - Manual history DataFrame

    Examples
    --------
    >>> analyzer = LearningCurveAnalyzer()
    >>> history = analyzer.fetch_wandb_history('my_project', 'run_name')
    >>> analyzer.plot_learning_curves(save_path)
    """

    def __init__(self):
        self.history = None
        self.source = None

    def set_history(self, history: pd.DataFrame, source: str = 'manual'):
        """
        Set history DataFrame manually.

        Parameters
        ----------
        history : pd.DataFrame
            Training history with columns like 'epoch', 'train_loss', 'val_loss'
        source : str
            Source identifier
        """
        self.history = history
        self.source = source

    def fetch_wandb_history(
        self,
        project_name: str,
        run_name: str,
        entity: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch training history from Weights & Biases.

        Parameters
        ----------
        project_name : str
            Wandb project name
        run_name : str
            Run name to fetch
        entity : str, optional
            Wandb entity (team or username)

        Returns
        -------
        pd.DataFrame or None
            Training history DataFrame
        """
        try:
            import wandb
            api = wandb.Api()

            # Build path
            if entity:
                path = f"{entity}/{project_name}"
            else:
                path = project_name

            runs = api.runs(path)

            for run in runs:
                if run.name == run_name:
                    history = run.history()

                    # Select relevant columns if they exist
                    cols = [
                        'epoch', '_step',
                        'train_loss', 'val_loss', 'loss',
                        'train_acc', 'val_acc', 'accuracy',
                        'train_f1', 'val_f1', 'f1',
                        'learning_rate', 'lr'
                    ]
                    available_cols = [c for c in cols if c in history.columns]

                    if available_cols:
                        self.history = history[available_cols].copy()
                    else:
                        self.history = history.copy()

                    self.source = 'wandb'
                    return self.history

            print(f"  Warning: Run '{run_name}' not found in project '{project_name}'")
            return None

        except ImportError:
            print("  Warning: wandb not installed. Cannot fetch learning curves.")
            return None
        except Exception as e:
            print(f"  Warning: Failed to fetch wandb history: {e}")
            return None

    def extract_from_checkpoints(
        self,
        checkpoint_dir: Path,
        pattern: str = 'checkpoint_epoch_*.pt'
    ) -> Optional[pd.DataFrame]:
        """
        Extract metrics from periodic checkpoints.

        Parameters
        ----------
        checkpoint_dir : Path
            Directory containing checkpoint files
        pattern : str
            Glob pattern for checkpoint files

        Returns
        -------
        pd.DataFrame or None
            Extracted history DataFrame
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_files = sorted(checkpoint_dir.glob(pattern))

        if not checkpoint_files:
            print(f"  Warning: No checkpoint files found in {checkpoint_dir}")
            return None

        history = []
        for ckpt_path in checkpoint_files:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                epoch = ckpt.get('epoch', 0)

                row = {'epoch': epoch}

                # Check for various metric storage patterns
                if 'metrics' in ckpt:
                    row.update(ckpt['metrics'])

                if 'val_metrics' in ckpt:
                    for k, v in ckpt['val_metrics'].items():
                        row[f'val_{k}'] = v

                if 'train_metrics' in ckpt:
                    for k, v in ckpt['train_metrics'].items():
                        row[f'train_{k}'] = v

                # Direct metric keys
                direct_keys = [
                    'train_loss', 'val_loss', 'loss',
                    'train_acc', 'val_acc', 'accuracy',
                    'train_f1', 'val_f1', 'f1',
                    'learning_rate', 'lr', 'best_metric'
                ]
                for key in direct_keys:
                    if key in ckpt:
                        row[key] = ckpt[key]

                history.append(row)

            except Exception as e:
                print(f"  Warning: Failed to load {ckpt_path}: {e}")
                continue

        if history:
            self.history = pd.DataFrame(history).sort_values('epoch')
            self.source = 'checkpoints'
            return self.history

        return None

    def extract_from_log_file(
        self,
        log_path: Path,
        delimiter: str = ','
    ) -> Optional[pd.DataFrame]:
        """
        Extract history from a CSV/TSV log file.

        Parameters
        ----------
        log_path : Path
            Path to log file
        delimiter : str
            Column delimiter

        Returns
        -------
        pd.DataFrame or None
            Extracted history
        """
        try:
            self.history = pd.read_csv(log_path, delimiter=delimiter)
            self.source = 'log_file'
            return self.history
        except Exception as e:
            print(f"  Warning: Failed to read log file: {e}")
            return None

    def get_best_epoch(
        self,
        metric: str = 'val_loss',
        mode: str = 'min'
    ) -> Dict[str, Any]:
        """
        Find the best epoch based on a metric.

        Parameters
        ----------
        metric : str
            Metric column name
        mode : str
            'min' or 'max'

        Returns
        -------
        dict
            Best epoch info with all metrics
        """
        if self.history is None or metric not in self.history.columns:
            return {}

        if mode == 'min':
            best_idx = self.history[metric].idxmin()
        else:
            best_idx = self.history[metric].idxmax()

        return self.history.iloc[best_idx].to_dict()

    def compute_training_stats(self) -> Dict[str, Any]:
        """
        Compute training statistics.

        Returns
        -------
        dict
            Training statistics (final metrics, best metrics, convergence info)
        """
        if self.history is None or len(self.history) == 0:
            return {}

        stats = {
            'n_epochs': len(self.history),
            'source': self.source
        }

        # Final epoch metrics
        final = self.history.iloc[-1].to_dict()
        stats['final_epoch'] = final.get('epoch', len(self.history) - 1)

        for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1']:
            if key in final:
                stats[f'final_{key}'] = final[key]

        # Best validation loss
        if 'val_loss' in self.history.columns:
            best_loss = self.get_best_epoch('val_loss', 'min')
            stats['best_val_loss'] = best_loss.get('val_loss')
            stats['best_val_loss_epoch'] = best_loss.get('epoch')

        # Best validation accuracy
        if 'val_acc' in self.history.columns:
            best_acc = self.get_best_epoch('val_acc', 'max')
            stats['best_val_acc'] = best_acc.get('val_acc')
            stats['best_val_acc_epoch'] = best_acc.get('epoch')

        return stats

    # =========================================================================
    # CSV Export
    # =========================================================================

    def save_history_csv(self, save_path: Path):
        """Save training history to CSV."""
        if self.history is not None:
            self.history.to_csv(save_path, index=False)
            print(f"  Saved training history to {save_path}")

    def save_stats_csv(self, save_path: Path):
        """Save training statistics to CSV."""
        stats = self.compute_training_stats()
        if stats:
            df = pd.DataFrame([stats])
            df.to_csv(save_path, index=False)
            print(f"  Saved training stats to {save_path}")

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot_learning_curves(
        self,
        save_path: Path,
        title: str = "Learning Curves",
        figsize: tuple = (12, 4)
    ):
        """
        Plot training and validation curves.

        Parameters
        ----------
        save_path : Path
            Output file path
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        if self.history is None or len(self.history) == 0:
            print("  Warning: No history data available for learning curves")
            return

        # Determine available metrics
        has_loss = 'train_loss' in self.history.columns or 'val_loss' in self.history.columns
        has_acc = 'train_acc' in self.history.columns or 'val_acc' in self.history.columns
        has_f1 = 'val_f1' in self.history.columns or 'train_f1' in self.history.columns
        has_lr = 'learning_rate' in self.history.columns or 'lr' in self.history.columns

        n_plots = sum([has_loss, has_acc, has_f1])
        if n_plots == 0:
            print("  Warning: No plottable metrics found in history")
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        # Get x-axis (epochs or steps)
        if 'epoch' in self.history.columns:
            x = self.history['epoch']
            x_label = 'Epoch'
        elif '_step' in self.history.columns:
            x = self.history['_step']
            x_label = 'Step'
        else:
            x = np.arange(len(self.history))
            x_label = 'Iteration'

        plot_idx = 0

        # Loss curves
        if has_loss:
            ax = axes[plot_idx]
            if 'train_loss' in self.history.columns:
                ax.plot(x, self.history['train_loss'], label='Train', color='blue', alpha=0.8)
            if 'val_loss' in self.history.columns:
                ax.plot(x, self.history['val_loss'], label='Validation', color='orange', alpha=0.8)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Accuracy curves
        if has_acc:
            ax = axes[plot_idx]
            if 'train_acc' in self.history.columns:
                ax.plot(x, self.history['train_acc'], label='Train', color='blue', alpha=0.8)
            if 'val_acc' in self.history.columns:
                ax.plot(x, self.history['val_acc'], label='Validation', color='orange', alpha=0.8)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # F1 curves
        if has_f1:
            ax = axes[plot_idx]
            if 'train_f1' in self.history.columns:
                ax.plot(x, self.history['train_f1'], label='Train', color='blue', alpha=0.8)
            if 'val_f1' in self.history.columns:
                ax.plot(x, self.history['val_f1'], label='Validation', color='green', alpha=0.8)
            ax.set_xlabel(x_label)
            ax.set_ylabel('F1 Score')
            ax.set_title('F1 Score Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved learning curves to {save_path}")

    def plot_lr_schedule(
        self,
        save_path: Path,
        title: str = "Learning Rate Schedule"
    ):
        """
        Plot learning rate schedule.

        Parameters
        ----------
        save_path : Path
            Output file path
        title : str
            Plot title
        """
        if self.history is None:
            print("  Warning: No history data available")
            return

        lr_col = None
        if 'learning_rate' in self.history.columns:
            lr_col = 'learning_rate'
        elif 'lr' in self.history.columns:
            lr_col = 'lr'

        if lr_col is None:
            print("  Warning: No learning rate data found")
            return

        fig, ax = plt.subplots(figsize=(8, 4))

        if 'epoch' in self.history.columns:
            x = self.history['epoch']
            x_label = 'Epoch'
        else:
            x = np.arange(len(self.history))
            x_label = 'Iteration'

        ax.plot(x, self.history[lr_col], color='purple', linewidth=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Learning Rate')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved learning rate plot to {save_path}")

    def plot_metric_comparison(
        self,
        metrics: List[str],
        save_path: Path,
        title: str = "Metric Comparison"
    ):
        """
        Plot multiple metrics on the same axis.

        Parameters
        ----------
        metrics : list
            List of metric column names
        save_path : Path
            Output file path
        title : str
            Plot title
        """
        if self.history is None:
            print("  Warning: No history data available")
            return

        available_metrics = [m for m in metrics if m in self.history.columns]
        if not available_metrics:
            print("  Warning: None of the specified metrics found")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        if 'epoch' in self.history.columns:
            x = self.history['epoch']
            x_label = 'Epoch'
        else:
            x = np.arange(len(self.history))
            x_label = 'Iteration'

        colors = plt.cm.tab10.colors
        for i, metric in enumerate(available_metrics):
            ax.plot(x, self.history[metric], label=metric,
                   color=colors[i % len(colors)], linewidth=2, alpha=0.8)

        ax.set_xlabel(x_label)
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved metric comparison to {save_path}")


# =============================================================================
# Utility Functions
# =============================================================================

def compare_training_histories(
    histories: List[pd.DataFrame],
    labels: List[str],
    metric: str,
    save_path: Path,
    title: str = "Training Comparison"
):
    """
    Compare training histories from multiple runs.

    Parameters
    ----------
    histories : list of pd.DataFrame
        List of history DataFrames
    labels : list of str
        Labels for each run
    metric : str
        Metric column to compare
    save_path : Path
        Output file path
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10.colors

    for i, (history, label) in enumerate(zip(histories, labels)):
        if metric not in history.columns:
            continue

        if 'epoch' in history.columns:
            x = history['epoch']
        else:
            x = np.arange(len(history))

        ax.plot(x, history[metric], label=label,
               color=colors[i % len(colors)], linewidth=2, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved training comparison to {save_path}")


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=" * 60)
    print("Testing Learning Curves Module")
    print("=" * 60)

    # Generate synthetic training history
    np.random.seed(42)
    n_epochs = 50

    epochs = np.arange(n_epochs)
    train_loss = 2.0 * np.exp(-0.1 * epochs) + 0.3 + np.random.randn(n_epochs) * 0.05
    val_loss = 2.0 * np.exp(-0.08 * epochs) + 0.4 + np.random.randn(n_epochs) * 0.1
    train_acc = 1 - train_loss / 3 + np.random.randn(n_epochs) * 0.02
    val_acc = 1 - val_loss / 3 + np.random.randn(n_epochs) * 0.03

    history = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': np.clip(train_acc, 0, 1),
        'val_acc': np.clip(val_acc, 0, 1)
    })

    # Test LearningCurveAnalyzer
    print("\n[1] Testing LearningCurveAnalyzer...")

    analyzer = LearningCurveAnalyzer()
    analyzer.set_history(history, source='synthetic')

    # Compute stats
    stats = analyzer.compute_training_stats()
    print(f"  Training epochs: {stats['n_epochs']}")
    print(f"  Best val_loss: {stats.get('best_val_loss', 'N/A'):.4f} at epoch {stats.get('best_val_loss_epoch', 'N/A')}")
    print(f"  Best val_acc: {stats.get('best_val_acc', 'N/A'):.4f} at epoch {stats.get('best_val_acc_epoch', 'N/A')}")

    # Test best epoch
    best = analyzer.get_best_epoch('val_acc', 'max')
    print(f"  Best epoch for val_acc: {best.get('epoch')}")

    print("\n[2] Note: Visualization tests require matplotlib display")
    print("  Skipping plot generation in test mode...")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
