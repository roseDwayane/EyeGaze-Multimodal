"""
Visualization Utilities for Multimodal Analysis

This module provides publication-quality visualization functions for:
1. Box plots: Per-subject distribution
2. KDE plots: Group-level density estimation
3. Topomaps: EEG channel-wise visualization
4. Correlation plots: Cross-modality relationships
5. Confusion Matrix: Classification performance
6. ROC Curves: Multi-class ROC analysis
7. t-SNE plots: Feature space visualization
8. Learning Curves: Training progress
9. Bar charts: Model comparison

Author: CNElab
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from scipy.interpolate import griddata


# =============================================================================
# Style Configuration
# =============================================================================

# Color palette for conditions (colorblind-friendly)
CONDITION_COLORS = {
    'Single': '#4C72B0',       # Blue
    'Competition': '#DD8452',  # Orange
    'Cooperation': '#55A868',  # Green
}

# Alternative palette using seaborn's colorblind palette
CONDITION_PALETTE = ['#0173B2', '#DE8F05', '#029E73']  # Blue, Orange, Green


def setup_academic_style():
    """
    Configure matplotlib/seaborn for publication-quality figures.

    Sets up:
    - White grid background
    - Large, readable fonts
    - High DPI for crisp output
    - Appropriate figure sizes
    """
    # Set seaborn style
    sns.set_theme(style="whitegrid", context="paper")

    # Update matplotlib rcParams
    plt.rcParams.update({
        # Figure
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # Lines and markers
        'lines.linewidth': 1.5,
        'lines.markersize': 6,

        # Axes
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })


# =============================================================================
# Box Plot
# =============================================================================

def plot_entropy_boxplot(
    df: pd.DataFrame,
    x: str = 'subject_id',
    y: str = 'entropy',
    hue: str = 'condition',
    title: str = 'Entropy Distribution by Subject',
    xlabel: str = 'Subject ID',
    ylabel: str = 'Entropy (bits)',
    figsize: Tuple[float, float] = (12, 6),
    palette: Optional[List[str]] = None,
    show_points: bool = True,
    point_alpha: float = 0.4,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a box plot showing entropy distribution per subject.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entropy data with columns for x, y, and hue
    x : str, default='subject_id'
        Column name for x-axis (categorical)
    y : str, default='entropy'
        Column name for y-axis (numerical)
    hue : str, default='condition'
        Column name for grouping/coloring
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    palette : list, optional
        Color palette. If None, uses CONDITION_PALETTE
    show_points : bool, default=True
        Whether to overlay individual data points
    point_alpha : float, default=0.4
        Alpha (transparency) for data points
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.Figure
        The created figure object
    """
    setup_academic_style()

    if palette is None:
        palette = CONDITION_PALETTE

    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        ax=ax,
        linewidth=1.2,
        fliersize=3
    )

    # Overlay strip plot for individual points
    if show_points:
        sns.stripplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            ax=ax,
            dodge=True,
            alpha=point_alpha,
            size=3,
            legend=False
        )

    # Customize
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    n_hue = df[hue].nunique()
    ax.legend(
        handles[:n_hue],
        labels[:n_hue],
        title='Condition',
        loc='upper right',
        frameon=True
    )

    # Rotate x labels if many subjects
    if df[x].nunique() > 10:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# KDE Plot
# =============================================================================

def plot_entropy_kde(
    df: pd.DataFrame,
    value_col: str = 'entropy',
    group_col: str = 'condition',
    title: str = 'Entropy Distribution by Condition',
    xlabel: str = 'Entropy (bits)',
    ylabel: str = 'Density',
    figsize: Tuple[float, float] = (8, 6),
    palette: Optional[List[str]] = None,
    fill: bool = True,
    fill_alpha: float = 0.3,
    linewidth: float = 2.0,
    show_rug: bool = False,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a KDE plot showing entropy distribution by condition.

    This is a GROUP-LEVEL visualization that combines all subjects'
    data within each condition to show overall distribution patterns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entropy data
    value_col : str, default='entropy'
        Column name for the values to plot
    group_col : str, default='condition'
        Column name for grouping
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    palette : list, optional
        Color palette. If None, uses CONDITION_PALETTE
    fill : bool, default=True
        Whether to fill under the KDE curves
    fill_alpha : float, default=0.3
        Alpha for filled areas
    linewidth : float, default=2.0
        Line width for KDE curves
    show_rug : bool, default=False
        Whether to show rug plot (individual observations)
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.Figure
        The created figure object
    """
    setup_academic_style()

    if palette is None:
        palette = CONDITION_PALETTE

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique groups and their order
    groups = ['Single', 'Competition', 'Cooperation']
    groups = [g for g in groups if g in df[group_col].unique()]

    # Plot KDE for each group
    for i, group in enumerate(groups):
        group_data = df[df[group_col] == group][value_col]
        color = palette[i] if i < len(palette) else None

        sns.kdeplot(
            data=group_data,
            ax=ax,
            label=group,
            color=color,
            fill=fill,
            alpha=fill_alpha if fill else 1.0,
            linewidth=linewidth
        )

        if show_rug:
            sns.rugplot(
                data=group_data,
                ax=ax,
                color=color,
                alpha=0.5,
                height=0.05
            )

    # Add statistical annotations
    stats_text = []
    for i, group in enumerate(groups):
        group_data = df[df[group_col] == group][value_col]
        mean_val = group_data.mean()
        std_val = group_data.std()
        stats_text.append(f"{group}: {mean_val:.3f} +/- {std_val:.3f}")

    # Add stats box
    stats_str = '\n'.join(stats_text)
    ax.text(
        0.98, 0.98, stats_str,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Customize
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title='Condition', loc='upper left', frameon=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Topomap
# =============================================================================

# Standard 10-20 positions for 32-channel setup
CHANNEL_POSITIONS_2D = {
    'Fp1': (-0.3, 0.9),   'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.5),    'F3': (-0.35, 0.5),   'Fz': (0.0, 0.5),    'F4': (0.35, 0.5),   'F8': (0.7, 0.5),
    'FT9': (-0.9, 0.3),   'FC5': (-0.55, 0.3),  'FC1': (-0.2, 0.3),  'FC2': (0.2, 0.3),   'FC6': (0.55, 0.3),  'FT10': (0.9, 0.3),
    'T7': (-0.9, 0.0),    'C3': (-0.45, 0.0),   'Cz': (0.0, 0.0),    'C4': (0.45, 0.0),   'T8': (0.9, 0.0),
    'TP9': (-0.9, -0.3),  'CP5': (-0.55, -0.3), 'CP1': (-0.2, -0.3), 'CP2': (0.2, -0.3),  'CP6': (0.55, -0.3), 'TP10': (0.9, -0.3),
    'P7': (-0.7, -0.5),   'P3': (-0.35, -0.5),  'Pz': (0.0, -0.5),   'P4': (0.35, -0.5),  'P8': (0.7, -0.5),
    'O1': (-0.3, -0.8),   'Oz': (0.0, -0.8),    'O2': (0.3, -0.8)
}

STANDARD_32_CHANNELS = list(CHANNEL_POSITIONS_2D.keys())


def plot_entropy_topomap(
    channel_entropies: np.ndarray,
    channel_names: Optional[List[str]] = None,
    title: str = 'Spectral Entropy Topomap',
    figsize: Tuple[float, float] = (8, 8),
    cmap: str = 'RdYlBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_channels: bool = True,
    show_colorbar: bool = True,
    interpolation_points: int = 100,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a 2D topomap showing EEG channel-wise entropy.

    Parameters
    ----------
    channel_entropies : np.ndarray
        Entropy values for each channel, shape (32,) or (n_channels,)
    channel_names : list, optional
        Names of channels. If None, uses STANDARD_32_CHANNELS
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    cmap : str, default='RdYlBu_r'
        Colormap name (reversed so high entropy = red)
    vmin, vmax : float, optional
        Color scale limits. If None, uses data min/max
    show_channels : bool, default=True
        Whether to show channel markers and labels
    show_colorbar : bool, default=True
        Whether to show colorbar
    interpolation_points : int, default=100
        Number of points for interpolation grid
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.Figure
        The created figure object
    """
    setup_academic_style()

    if channel_names is None:
        channel_names = STANDARD_32_CHANNELS[:len(channel_entropies)]

    # Get positions for the channels we have
    positions = []
    values = []
    valid_names = []

    for i, name in enumerate(channel_names):
        if name in CHANNEL_POSITIONS_2D:
            positions.append(CHANNEL_POSITIONS_2D[name])
            values.append(channel_entropies[i])
            valid_names.append(name)

    positions = np.array(positions)
    values = np.array(values)

    if len(positions) == 0:
        raise ValueError("No valid channel positions found. Check channel names.")

    # Create interpolation grid
    xi = np.linspace(-1.1, 1.1, interpolation_points)
    yi = np.linspace(-1.1, 1.1, interpolation_points)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate values
    Zi = griddata(positions, values, (Xi, Yi), method='cubic')

    # Create head mask (circular)
    head_radius = 1.0
    mask = np.sqrt(Xi**2 + Yi**2) > head_radius
    Zi[mask] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot interpolated surface
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    im = ax.contourf(
        Xi, Yi, Zi,
        levels=50,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend='both'
    )

    # Draw head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta),
            'k-', linewidth=2)

    # Draw nose
    nose_x = [0.1, 0, -0.1]
    nose_y = [head_radius, head_radius + 0.15, head_radius]
    ax.plot(nose_x, nose_y, 'k-', linewidth=2)

    # Draw ears
    ear_x = np.array([0.08, 0.12, 0.12, 0.08])
    ear_y = np.array([-0.1, -0.05, 0.05, 0.1])
    ax.plot(head_radius + ear_x, ear_y, 'k-', linewidth=2)
    ax.plot(-head_radius - ear_x, ear_y, 'k-', linewidth=2)

    # Plot channel positions
    if show_channels:
        ax.scatter(positions[:, 0], positions[:, 1],
                   c='white', s=50, edgecolors='black', linewidths=1, zorder=5)
        for i, name in enumerate(valid_names):
            ax.annotate(name, positions[i],
                        fontsize=7, ha='center', va='center',
                        color='black', weight='bold')

    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Entropy (bits)', fontsize=11)

    ax.set_title(title, fontweight='bold', pad=15, fontsize=14)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.2, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Correlation Plot
# =============================================================================

def plot_entropy_correlation(
    df: pd.DataFrame,
    x_col: str = 'gaze_entropy',
    y_col: str = 'eeg_entropy',
    hue_col: str = 'condition',
    title: str = 'Gaze vs EEG Entropy Correlation',
    xlabel: str = 'Gaze Spatial Entropy (bits)',
    ylabel: str = 'EEG Spectral Entropy (bits)',
    figsize: Tuple[float, float] = (8, 6),
    palette: Optional[List[str]] = None,
    show_regression: bool = True,
    show_overall_regression: bool = False,
    annotate_r: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a scatter plot showing correlation between two entropy measures.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both entropy measures
    x_col : str
        Column name for x-axis values
    y_col : str
        Column name for y-axis values
    hue_col : str
        Column name for grouping/coloring
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    palette : list, optional
        Color palette
    show_regression : bool, default=True
        Whether to show regression lines per group
    show_overall_regression : bool, default=False
        Whether to show overall regression line
    annotate_r : bool, default=True
        Whether to annotate Pearson correlation coefficient
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.Figure
        The created figure object
    """
    setup_academic_style()

    if palette is None:
        palette = CONDITION_PALETTE

    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot with regression
    if show_regression:
        sns.regplot(
            data=df, x=x_col, y=y_col,
            scatter=False, ax=ax,
            color='gray', line_kws={'linestyle': '--', 'alpha': 0.5}
        ) if show_overall_regression else None

        # Scatter with hue
        sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col,
            palette=palette, ax=ax, alpha=0.7, s=40
        )

        # Add regression lines per group
        groups = ['Single', 'Competition', 'Cooperation']
        groups = [g for g in groups if g in df[hue_col].unique()]

        for i, group in enumerate(groups):
            group_data = df[df[hue_col] == group]
            if len(group_data) > 2:
                color = palette[i] if i < len(palette) else None
                sns.regplot(
                    data=group_data, x=x_col, y=y_col,
                    scatter=False, ax=ax,
                    color=color, line_kws={'linewidth': 2}
                )
    else:
        sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col,
            palette=palette, ax=ax, alpha=0.7, s=40
        )

    # Annotate correlation coefficients
    if annotate_r:
        from scipy.stats import pearsonr

        stats_text = []
        groups = ['Single', 'Competition', 'Cooperation']
        groups = [g for g in groups if g in df[hue_col].unique()]

        for group in groups:
            group_data = df[df[hue_col] == group]
            if len(group_data) > 2:
                r, p = pearsonr(group_data[x_col], group_data[y_col])
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                stats_text.append(f"{group}: r={r:.3f}{sig}")

        # Overall correlation
        r_all, p_all = pearsonr(df[x_col], df[y_col])
        sig_all = '***' if p_all < 0.001 else '**' if p_all < 0.01 else '*' if p_all < 0.05 else ''
        stats_text.append(f"Overall: r={r_all:.3f}{sig_all}")

        stats_str = '\n'.join(stats_text)
        ax.text(
            0.02, 0.98, stats_str,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )

    # Customize
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title='Condition', loc='lower right', frameon=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Additional Utility Plots
# =============================================================================

def plot_entropy_violin(
    df: pd.DataFrame,
    x: str = 'condition',
    y: str = 'entropy',
    title: str = 'Entropy Distribution by Condition',
    xlabel: str = 'Condition',
    ylabel: str = 'Entropy (bits)',
    figsize: Tuple[float, float] = (8, 6),
    palette: Optional[List[str]] = None,
    inner: str = 'box',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a violin plot showing entropy distribution by condition.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entropy data
    x : str
        Column for x-axis (categorical)
    y : str
        Column for y-axis (numerical)
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    palette : list, optional
        Color palette
    inner : str, default='box'
        Interior representation ('box', 'quartile', 'point', 'stick', None)
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    if palette is None:
        palette = CONDITION_PALETTE

    fig, ax = plt.subplots(figsize=figsize)

    # Order conditions
    order = ['Single', 'Competition', 'Cooperation']
    order = [o for o in order if o in df[x].unique()]

    sns.violinplot(
        data=df, x=x, y=y,
        hue=x,
        order=order,
        palette=palette,
        inner=inner,
        ax=ax,
        legend=False
    )

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_entropy_heatmap(
    df: pd.DataFrame,
    index: str = 'subject_id',
    columns: str = 'condition',
    values: str = 'mean_entropy',
    title: str = 'Mean Entropy Heatmap',
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'YlOrRd',
    annot: bool = True,
    fmt: str = '.3f',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a heatmap showing mean entropy by subject and condition.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with summary statistics
    index : str
        Column to use as rows
    columns : str
        Column to use as columns
    values : str
        Column to use as cell values
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    annot : bool
        Whether to annotate cells with values
    fmt : str
        Format string for annotations
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    # Pivot data
    pivot_df = df.pivot_table(index=index, columns=columns, values=values)

    # Reorder columns
    col_order = ['Single', 'Competition', 'Cooperation']
    col_order = [c for c in col_order if c in pivot_df.columns]
    pivot_df = pivot_df[col_order]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot_df,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Entropy (bits)'}
    )

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Subject ID')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Confusion Matrix Plot
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    title: str = 'Confusion Matrix',
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = 'Blues',
    normalize: bool = True,
    show_counts: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix heatmap with counts and percentages.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes)
    class_names : list, optional
        Class names for labels. Default: ["Single", "Competition", "Cooperation"]
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    normalize : bool
        Whether to show percentages
    show_counts : bool
        Whether to show raw counts
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    if class_names is None:
        class_names = ['Single', 'Competition', 'Cooperation']

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize for percentages if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    else:
        cm_display = cm.astype('float')

    # Plot heatmap
    sns.heatmap(
        cm_display,
        annot=False,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Percentage (%)' if normalize else 'Count'}
    )

    # Add annotations with both counts and percentages
    n_classes = len(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            if show_counts and normalize:
                text = f'{cm[i, j]}\n({cm_display[i, j]:.1f}%)'
            elif normalize:
                text = f'{cm_display[i, j]:.1f}%'
            else:
                text = f'{cm[i, j]}'

            color = 'white' if cm_display[i, j] > 50 else 'black'
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center',
                    color=color, fontsize=10)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# ROC Curves Plot
# =============================================================================

def plot_roc_curves(
    roc_data: Dict,
    class_names: List[str] = None,
    title: str = 'ROC Curves',
    figsize: Tuple[float, float] = (8, 6),
    show_micro: bool = True,
    show_macro: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot multi-class ROC curves.

    Parameters
    ----------
    roc_data : dict
        ROC data dictionary with keys for each class and 'micro'/'macro'.
        Each entry should have 'fpr', 'tpr', and 'auc' keys.
    class_names : list, optional
        Class names. Default: ["Single", "Competition", "Cooperation"]
    title : str
        Plot title
    figsize : tuple
        Figure size
    show_micro : bool
        Whether to show micro-average ROC
    show_macro : bool
        Whether to show macro-average ROC
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    if class_names is None:
        class_names = ['Single', 'Competition', 'Cooperation']

    fig, ax = plt.subplots(figsize=figsize)

    colors = CONDITION_PALETTE

    # Plot per-class ROC
    for i, name in enumerate(class_names):
        if name in roc_data:
            data = roc_data[name]
            color = colors[i] if i < len(colors) else None
            ax.plot(data['fpr'], data['tpr'],
                    color=color, lw=2,
                    label=f'{name} (AUC = {data["auc"]:.3f})')

    # Plot micro-average
    if show_micro and 'micro' in roc_data:
        micro = roc_data['micro']
        ax.plot(micro['fpr'], micro['tpr'],
                color='navy', linestyle='--', lw=2,
                label=f'Micro-avg (AUC = {micro["auc"]:.3f})')

    # Plot macro-average
    if show_macro and 'macro' in roc_data:
        macro = roc_data['macro']
        ax.plot(macro['fpr'], macro['tpr'],
                color='darkorange', linestyle=':', lw=2,
                label=f'Macro-avg (AUC = {macro["auc"]:.3f})')

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# t-SNE Plot
# =============================================================================

def plot_tsne(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None,
    predictions: Optional[np.ndarray] = None,
    title: str = 't-SNE Feature Visualization',
    figsize: Tuple[float, float] = (16, 7),
    palette: Optional[List[str]] = None,
    point_size: int = 30,
    alpha: float = 0.6,
    highlight_errors: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot t-SNE scatter plot with class coloring and error highlighting.

    Parameters
    ----------
    tsne_coords : np.ndarray
        t-SNE coordinates, shape (N, 2)
    labels : np.ndarray
        Ground truth labels, shape (N,)
    class_names : list, optional
        Class names
    predictions : np.ndarray, optional
        Predicted labels for error highlighting
    title : str
        Plot title
    figsize : tuple
        Figure size
    palette : list, optional
        Color palette
    point_size : int
        Size of scatter points
    alpha : float
        Point transparency
    highlight_errors : bool
        Whether to highlight misclassified samples (requires predictions)
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    if class_names is None:
        class_names = ['Single', 'Competition', 'Cooperation']

    if palette is None:
        palette = CONDITION_PALETTE

    # Determine number of subplots
    if predictions is not None and highlight_errors:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        axes = [ax]

    # Plot 1: Colored by true labels
    ax = axes[0]
    for i, name in enumerate(class_names):
        mask = labels == i
        color = palette[i] if i < len(palette) else None
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=color, label=name, alpha=alpha, s=point_size)

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Feature Space (True Labels)')
    ax.legend()

    # Plot 2: Highlight misclassified samples
    if predictions is not None and highlight_errors and len(axes) > 1:
        ax = axes[1]
        correct_mask = labels == predictions

        # Plot correct predictions lightly
        for i, name in enumerate(class_names):
            mask = (labels == i) & correct_mask
            color = palette[i] if i < len(palette) else None
            ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                       c=color, alpha=0.3, s=point_size//2)

        # Plot misclassified samples with markers
        incorrect_mask = ~correct_mask
        ax.scatter(tsne_coords[incorrect_mask, 0], tsne_coords[incorrect_mask, 1],
                   c='red', marker='x', s=point_size*1.5,
                   linewidths=2, label='Misclassified')

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('Misclassified Samples Highlighted')
        ax.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Learning Curves Plot
# =============================================================================

def plot_learning_curves(
    history: pd.DataFrame,
    title: str = 'Learning Curves',
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation learning curves.

    Parameters
    ----------
    history : pd.DataFrame
        Training history with columns like 'epoch', 'train_loss', 'val_loss', etc.
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    if history is None or len(history) == 0:
        print("Warning: No history data available")
        return None

    # Determine available metrics
    has_loss = 'train_loss' in history.columns or 'val_loss' in history.columns
    has_acc = 'train_acc' in history.columns or 'val_acc' in history.columns
    has_f1 = 'val_f1' in history.columns

    n_plots = sum([has_loss, has_acc, has_f1])
    if n_plots == 0:
        print("Warning: No plottable metrics found")
        return None

    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0], figsize[1]))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    epoch_col = 'epoch' if 'epoch' in history.columns else history.index

    if has_loss:
        ax = axes[plot_idx]
        if 'train_loss' in history.columns:
            ax.plot(history[epoch_col], history['train_loss'],
                    label='Train', color='blue')
        if 'val_loss' in history.columns:
            ax.plot(history[epoch_col], history['val_loss'],
                    label='Validation', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        plot_idx += 1

    if has_acc:
        ax = axes[plot_idx]
        if 'train_acc' in history.columns:
            ax.plot(history[epoch_col], history['train_acc'],
                    label='Train', color='blue')
        if 'val_acc' in history.columns:
            ax.plot(history[epoch_col], history['val_acc'],
                    label='Validation', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        plot_idx += 1

    if has_f1:
        ax = axes[plot_idx]
        ax.plot(history[epoch_col], history['val_f1'],
                label='Val F1', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Curve')
        ax.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Model Comparison Bar Chart
# =============================================================================

def plot_metrics_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = None,
    model_col: str = 'model',
    title: str = 'Model Performance Comparison',
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot bar chart comparing metrics across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with model metrics, one row per model
    metrics : list, optional
        List of metric columns to plot. Default: ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    model_col : str
        Column containing model names
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    if metrics is None:
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    available_metrics = [m for m in metrics if m in comparison_df.columns]
    if not available_metrics:
        print("Warning: No matching metrics found in DataFrame")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    model_names = comparison_df[model_col].tolist()
    x = np.arange(len(available_metrics))
    width = 0.8 / len(model_names)

    colors = plt.cm.tab10.colors

    for idx, model in enumerate(model_names):
        model_data = comparison_df[comparison_df[model_col] == model]
        values = [model_data[m].values[0] for m in available_metrics]
        offset = (idx - len(model_names)/2 + 0.5) * width

        bars = ax.bar(x + offset, values, width,
                      label=model, color=colors[idx % len(colors)], alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
    ax.legend(title='Model')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Pair Accuracy Bar Chart
# =============================================================================

def plot_pair_accuracy(
    pair_stats: pd.DataFrame,
    overall_accuracy: float,
    pair_col: str = 'pair_id',
    accuracy_col: str = 'accuracy',
    title: str = 'Per-Pair Accuracy',
    figsize: Tuple[float, float] = (14, 6),
    highlight_threshold: float = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot bar chart showing accuracy by pair ID.

    Parameters
    ----------
    pair_stats : pd.DataFrame
        DataFrame with pair statistics
    overall_accuracy : float
        Overall accuracy for reference line
    pair_col : str
        Column containing pair IDs
    accuracy_col : str
        Column containing accuracy values
    title : str
        Plot title
    figsize : tuple
        Figure size
    highlight_threshold : float, optional
        Threshold below which pairs are highlighted in red
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by accuracy
    pair_stats_sorted = pair_stats.sort_values(accuracy_col, ascending=False)

    # Determine threshold if not provided
    if highlight_threshold is None:
        highlight_threshold = pair_stats[accuracy_col].mean() - 1.5 * pair_stats[accuracy_col].std()

    # Color bars
    colors = ['#e74c3c' if acc < highlight_threshold else '#3498db'
              for acc in pair_stats_sorted[accuracy_col]]

    ax.bar(range(len(pair_stats_sorted)),
           pair_stats_sorted[accuracy_col],
           color=colors, alpha=0.8)

    # Add overall accuracy line
    ax.axhline(y=overall_accuracy, color='green', linestyle='--', linewidth=2,
               label=f'Overall Accuracy: {overall_accuracy:.2%}')

    # Add threshold line if meaningful
    if highlight_threshold > 0:
        ax.axhline(y=highlight_threshold, color='red', linestyle=':', linewidth=1,
                   label=f'Hard Pair Threshold: {highlight_threshold:.2%}')

    ax.set_xlabel('Pair (sorted by accuracy)')
    ax.set_ylabel('Accuracy')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(range(len(pair_stats_sorted)))
    ax.set_xticklabels(pair_stats_sorted[pair_col], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Mechanism Analysis Plot
# =============================================================================

def plot_mechanism_analysis(
    data: pd.DataFrame,
    analysis_type: str,
    value_col: str,
    label_col: str = 'label_name',
    stats_df: Optional[pd.DataFrame] = None,
    class_names: List[str] = None,
    title: str = 'Mechanism Analysis',
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot mechanism analysis figures (spatial or correlation).

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data
    analysis_type : str
        Type of analysis: 'spatial' or 'correlation'
    value_col : str
        Column containing values to analyze
    label_col : str
        Column containing class labels
    stats_df : pd.DataFrame, optional
        Statistical test results
    class_names : list, optional
        Class names
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    setup_academic_style()

    if class_names is None:
        class_names = ['Single', 'Competition', 'Cooperation']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if analysis_type == 'spatial':
        # Left: Accuracy vs Distance (binned)
        if 'correct' in data.columns:
            data['distance_bin'] = pd.cut(data[value_col], bins=10)
            bin_stats = data.groupby('distance_bin')['correct'].mean()

            axes[0].bar(range(len(bin_stats)), bin_stats.values, alpha=0.7, color='#3498db')
            axes[0].set_xlabel('Gaze Distance Bin')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Accuracy vs. Gaze Distance')

        # Right: Distance distribution by class
        for name in class_names:
            class_data = data[data[label_col] == name][value_col]
            if len(class_data) > 0:
                sns.kdeplot(class_data, ax=axes[1], label=name, fill=True, alpha=0.3)

        axes[1].set_xlabel('Gaze Distance')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Gaze Distance Distribution by Class')
        axes[1].legend()

    else:  # correlation
        # Left: Violin plot
        sns.violinplot(data=data, x=label_col, y=value_col,
                       hue=label_col, ax=axes[0],
                       palette=CONDITION_COLORS, legend=False)
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Feature Cosine Similarity')
        axes[0].set_title('Inter-stream Feature Correlation by Class')

        # Right: Box plot with points
        sns.boxplot(data=data, x=label_col, y=value_col,
                    hue=label_col, ax=axes[1],
                    palette=CONDITION_COLORS, showfliers=False, legend=False)
        sns.stripplot(data=data, x=label_col, y=value_col,
                      ax=axes[1], color='black', alpha=0.2, size=2)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Feature Cosine Similarity')
        axes[1].set_title('Feature Correlation Distribution')

    # Add statistical test results as text
    if stats_df is not None:
        stats_text = "Statistical Tests:\n"
        for _, row in stats_df.iterrows():
            sig = "*" if row.get('significant', False) else ""
            stats_text += f"{row['comparison']}: p={row['p_value']:.4f}{sig}\n"

        fig.text(0.02, 0.02, stats_text, fontsize=8, family='monospace',
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{title} ({analysis_type.capitalize()})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=" * 60)
    print("Testing Visualizers")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)

    n_subjects = 5
    trials_per_condition = {'Single': 40, 'Competition': 20, 'Cooperation': 20}

    data = []
    for subj in range(n_subjects):
        for condition, n_trials in trials_per_condition.items():
            # Different entropy distributions per condition
            base_entropy = {'Single': 4.0, 'Competition': 4.5, 'Cooperation': 4.2}
            entropies = np.random.normal(base_entropy[condition], 0.3, n_trials)
            for i, e in enumerate(entropies):
                data.append({
                    'subject_id': f'S{subj+1:02d}',
                    'condition': condition,
                    'trial': i + 1,
                    'entropy': e
                })

    df = pd.DataFrame(data)

    print(f"\nGenerated {len(df)} data points")
    print(df.groupby('condition')['entropy'].describe())

    # Test box plot
    print("\n[1] Testing Box Plot...")
    plot_entropy_boxplot(df, show=False)
    print("    Box plot: OK")

    # Test KDE plot
    print("\n[2] Testing KDE Plot...")
    plot_entropy_kde(df, show=False)
    print("    KDE plot: OK")

    # Test topomap
    print("\n[3] Testing Topomap...")
    channel_entropy = np.random.rand(32) * 2 + 3  # Random entropy values
    plot_entropy_topomap(channel_entropy, show=False)
    print("    Topomap: OK")

    # Test correlation plot
    print("\n[4] Testing Correlation Plot...")
    df['gaze_entropy'] = df['entropy'] + np.random.normal(0, 0.2, len(df))
    df['eeg_entropy'] = df['entropy']
    plot_entropy_correlation(df, show=False)
    print("    Correlation plot: OK")

    print("\n" + "=" * 60)
    print("All visualization tests completed!")
    print("=" * 60)
