"""
Entropy Analysis Script for Multimodal Physiological Signals

This script computes Information Entropy for Gaze heatmaps (Spatial Entropy)
and EEG signals (Spectral Entropy) to evaluate data complexity and variability
across subjects and experimental conditions.

Usage:
    # Analyze EEG data only
    python analyze_entropy.py --modality eeg

    # Analyze Gaze data only
    python analyze_entropy.py --modality gaze

    # Analyze both modalities
    python analyze_entropy.py --modality both

    # Use mock data for testing
    python analyze_entropy.py --modality both --use_mock

Author: Kong-Yi Chang
Date: 2026
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import using importlib since folder names start with numbers
import importlib.util

def import_module_from_path(module_name: str, file_path: str):
    """Import a module from a file path (handles folder names starting with numbers)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import entropy calculators
entropy_module = import_module_from_path(
    "entropy_calculators",
    str(PROJECT_ROOT / "5_Metrics" / "entropy_calculators.py")
)
SpatialEntropyCalculator = entropy_module.SpatialEntropyCalculator
SpectralEntropyCalculator = entropy_module.SpectralEntropyCalculator
STANDARD_32_CHANNELS = entropy_module.STANDARD_32_CHANNELS

# Import visualizers
viz_module = import_module_from_path(
    "visualizers",
    str(PROJECT_ROOT / "6_Utils" / "visualizers.py")
)
setup_academic_style = viz_module.setup_academic_style
plot_entropy_boxplot = viz_module.plot_entropy_boxplot
plot_entropy_kde = viz_module.plot_entropy_kde
plot_entropy_topomap = viz_module.plot_entropy_topomap
plot_entropy_correlation = viz_module.plot_entropy_correlation
plot_entropy_violin = viz_module.plot_entropy_violin
plot_entropy_heatmap = viz_module.plot_entropy_heatmap


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for entropy analysis."""

    # Data paths
    eeg_data_path: str = str(PROJECT_ROOT / "1_Data" / "datasets" / "EEGseg")
    gaze_data_path: str = r"G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\bgOn_heatmapOn_trajOn"

    # Output paths
    output_dir: str = str(PROJECT_ROOT / "7_Analysis" / "raw_result" / "entropy_analysis")
    figure_dir: str = str(PROJECT_ROOT / "7_Analysis" / "figures" / "entropy_analysis")

    # EEG parameters
    eeg_sampling_rate: float = 250.0
    eeg_filter_low: float = 0.5
    eeg_filter_high: float = 50.0
    eeg_n_channels: int = 32

    # Analysis parameters
    conditions: List[str] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = ['Single', 'Competition', 'Cooperation']


# =============================================================================
# Data Loading Utilities
# =============================================================================

def parse_eeg_filename(filename: str) -> Optional[Dict]:
    """
    Parse EEG CSV filename to extract metadata.

    Filename patterns:
    - Single: Pair-{id}-A-Single-EYE_trial{num}_player.csv
              Pair-{id}-B-Single-EYE_trial{num}_observer.csv
    - Competition: Pair-{id}-Comp-EYE_trial{num}_playerA.csv
    - Cooperation: Pair-{id}-Coop-EYE_trial{num}_playerA.csv

    Returns
    -------
    dict or None
        Dictionary with keys: pair_id, condition, trial_idx, player
        Returns None if filename doesn't match expected pattern
    """
    # Pattern for Single condition
    single_pattern = r"Pair-(\d+)-([AB])-Single-EYE_trial(\d+)_(player|observer)\.csv"
    match = re.match(single_pattern, filename)
    if match:
        pair_id, ab, trial_idx, role = match.groups()
        return {
            'pair_id': int(pair_id),
            'condition': 'Single',
            'trial_idx': int(trial_idx),
            'player': f"{ab}_{role}"  # e.g., "A_player" or "B_observer"
        }

    # Pattern for Competition condition
    comp_pattern = r"Pair-(\d+)-Comp-EYE_trial(\d+)_(playerA|playerB)\.csv"
    match = re.match(comp_pattern, filename)
    if match:
        pair_id, trial_idx, player = match.groups()
        return {
            'pair_id': int(pair_id),
            'condition': 'Competition',
            'trial_idx': int(trial_idx),
            'player': player
        }

    # Pattern for Cooperation condition
    coop_pattern = r"Pair-(\d+)-Coop-EYE_trial(\d+)_(playerA|playerB)\.csv"
    match = re.match(coop_pattern, filename)
    if match:
        pair_id, trial_idx, player = match.groups()
        return {
            'pair_id': int(pair_id),
            'condition': 'Cooperation',
            'trial_idx': int(trial_idx),
            'player': player
        }

    return None


def parse_gaze_filename(filename: str) -> Optional[Dict]:
    """
    Parse Gaze image filename to extract metadata.

    Expected patterns similar to EEG filenames but with .jpg extension.

    Returns
    -------
    dict or None
        Dictionary with keys: pair_id, condition, trial_idx, player
    """
    # Remove extension and parse similar to EEG
    base = filename.replace('.jpg', '.csv').replace('.png', '.csv')
    result = parse_eeg_filename(base)
    return result


def load_eeg_csv(filepath: str) -> np.ndarray:
    """
    Load EEG data from CSV file.

    The CSV format is:
    - Each row is one channel (32 rows)
    - Each column is one timepoint
    - Values are comma-separated floats

    Parameters
    ----------
    filepath : str
        Path to CSV file

    Returns
    -------
    np.ndarray
        EEG data of shape (n_channels, n_timepoints)
    """
    # Read CSV without header
    data = pd.read_csv(filepath, header=None)
    return data.values.astype(np.float32)


def load_gaze_image(filepath: str) -> np.ndarray:
    """
    Load Gaze heatmap image.

    Parameters
    ----------
    filepath : str
        Path to image file

    Returns
    -------
    np.ndarray
        Image data of shape (H, W, 3) or (H, W)
    """
    img = Image.open(filepath)
    return np.array(img)


def scan_eeg_files(eeg_dir: str) -> List[Dict]:
    """
    Scan EEG directory and return list of file metadata.

    Parameters
    ----------
    eeg_dir : str
        Path to EEG data directory

    Returns
    -------
    list of dict
        List of dictionaries with file info and metadata
    """
    eeg_path = Path(eeg_dir)
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")

    files = []
    for csv_file in eeg_path.glob("*.csv"):
        meta = parse_eeg_filename(csv_file.name)
        if meta is not None:
            meta['filepath'] = str(csv_file)
            meta['filename'] = csv_file.name
            files.append(meta)

    print(f"Found {len(files)} EEG CSV files")
    return files


def scan_gaze_files(gaze_dir: str) -> List[Dict]:
    """
    Scan Gaze directory and return list of file metadata.

    Parameters
    ----------
    gaze_dir : str
        Path to Gaze image directory

    Returns
    -------
    list of dict
        List of dictionaries with file info and metadata
    """
    gaze_path = Path(gaze_dir)
    if not gaze_path.exists():
        raise FileNotFoundError(f"Gaze directory not found: {gaze_dir}")

    files = []
    for img_file in gaze_path.glob("*.jpg"):
        meta = parse_gaze_filename(img_file.name)
        if meta is not None:
            meta['filepath'] = str(img_file)
            meta['filename'] = img_file.name
            files.append(meta)

    # Also check for PNG files
    for img_file in gaze_path.glob("*.png"):
        meta = parse_gaze_filename(img_file.name)
        if meta is not None:
            meta['filepath'] = str(img_file)
            meta['filename'] = img_file.name
            files.append(meta)

    print(f"Found {len(files)} Gaze image files")
    return files


# =============================================================================
# Mock Data Generation
# =============================================================================

def generate_mock_data(
    n_subjects: int = 5,
    trials_per_condition: Dict[str, int] = None,
    output_dir: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate mock entropy data for testing.

    Creates synthetic data with realistic entropy distributions for
    each condition:
    - Single: Lower entropy (more regular patterns)
    - Competition: Higher entropy (more chaotic)
    - Cooperation: Medium entropy

    Parameters
    ----------
    n_subjects : int
        Number of subjects (pairs) to simulate
    trials_per_condition : dict
        Number of trials per condition. Default: Single=40, Comp=20, Coop=20
    output_dir : str, optional
        If provided, save mock data to this directory

    Returns
    -------
    gaze_df, eeg_df : pd.DataFrame
        DataFrames with mock entropy data
    """
    if trials_per_condition is None:
        trials_per_condition = {
            'Single': 40,
            'Competition': 20,
            'Cooperation': 20
        }

    np.random.seed(42)

    # Entropy parameters per condition (mean, std)
    # Based on theoretical expectations
    gaze_params = {
        'Single': (4.5, 0.4),       # Focused attention
        'Competition': (5.2, 0.5),  # Distributed attention
        'Cooperation': (4.8, 0.3),  # Moderate focus
    }

    eeg_params = {
        'Single': (3.8, 0.3),       # More regular neural activity
        'Competition': (4.5, 0.4),  # More variable
        'Cooperation': (4.1, 0.35), # Intermediate
    }

    gaze_data = []
    eeg_data = []

    for subj_idx in range(n_subjects):
        pair_id = 12 + subj_idx  # Start from Pair-12

        for player in ['playerA', 'playerB']:
            for condition, n_trials in trials_per_condition.items():
                # Gaze entropy
                gaze_mean, gaze_std = gaze_params[condition]
                gaze_entropies = np.random.normal(gaze_mean, gaze_std, n_trials)

                # EEG entropy (per channel)
                eeg_mean, eeg_std = eeg_params[condition]

                for trial_idx in range(n_trials):
                    # Gaze data point
                    gaze_data.append({
                        'pair_id': pair_id,
                        'player': player,
                        'trial_idx': trial_idx + 1,
                        'condition': condition,
                        'spatial_entropy': gaze_entropies[trial_idx]
                    })

                    # EEG data point (with per-channel values)
                    channel_entropies = np.random.normal(
                        eeg_mean, eeg_std, 32
                    )
                    eeg_row = {
                        'pair_id': pair_id,
                        'player': player,
                        'trial_idx': trial_idx + 1,
                        'condition': condition,
                        'mean_entropy': channel_entropies.mean()
                    }
                    # Add per-channel columns
                    for ch_idx, ch_name in enumerate(STANDARD_32_CHANNELS):
                        eeg_row[ch_name] = channel_entropies[ch_idx]

                    eeg_data.append(eeg_row)

    gaze_df = pd.DataFrame(gaze_data)
    eeg_df = pd.DataFrame(eeg_data)

    print(f"Generated mock data:")
    print(f"  Gaze: {len(gaze_df)} rows")
    print(f"  EEG: {len(eeg_df)} rows")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        gaze_df.to_csv(Path(output_dir) / "mock_gaze_entropy_raw.csv", index=False)
        eeg_df.to_csv(Path(output_dir) / "mock_eeg_entropy_raw.csv", index=False)
        print(f"  Saved to: {output_dir}")

    return gaze_df, eeg_df


# =============================================================================
# Entropy Analysis Functions
# =============================================================================

def analyze_gaze_entropy(
    config: Config,
    file_list: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Compute Spatial Entropy for all Gaze heatmaps.

    Parameters
    ----------
    config : Config
        Configuration object
    file_list : list, optional
        Pre-scanned file list. If None, will scan directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        pair_id, player, trial_idx, condition, spatial_entropy
    """
    if file_list is None:
        file_list = scan_gaze_files(config.gaze_data_path)

    if len(file_list) == 0:
        print("Warning: No Gaze files found!")
        return pd.DataFrame()

    calculator = SpatialEntropyCalculator()
    results = []

    print("\nComputing Gaze Spatial Entropy...")
    for file_info in tqdm(file_list, desc="Processing Gaze"):
        try:
            # Load image
            image = load_gaze_image(file_info['filepath'])

            # Compute entropy
            entropy = calculator.compute(image)

            results.append({
                'pair_id': file_info['pair_id'],
                'player': file_info['player'],
                'trial_idx': file_info['trial_idx'],
                'condition': file_info['condition'],
                'spatial_entropy': entropy
            })
        except Exception as e:
            print(f"Error processing {file_info['filename']}: {e}")
            continue

    df = pd.DataFrame(results)
    print(f"Computed entropy for {len(df)} Gaze trials")
    return df


def analyze_eeg_entropy(
    config: Config,
    file_list: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Compute Spectral Entropy for all EEG trials.

    Parameters
    ----------
    config : Config
        Configuration object
    file_list : list, optional
        Pre-scanned file list. If None, will scan directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        pair_id, player, trial_idx, condition, mean_entropy, ch01, ch02, ..., ch32
    """
    if file_list is None:
        file_list = scan_eeg_files(config.eeg_data_path)

    if len(file_list) == 0:
        print("Warning: No EEG files found!")
        return pd.DataFrame()

    calculator = SpectralEntropyCalculator(
        sampling_rate=config.eeg_sampling_rate,
        filter_low=config.eeg_filter_low,
        filter_high=config.eeg_filter_high,
        apply_filter=True
    )

    results = []

    print("\nComputing EEG Spectral Entropy...")
    for file_info in tqdm(file_list, desc="Processing EEG"):
        try:
            # Load EEG data
            eeg_data = load_eeg_csv(file_info['filepath'])

            # Ensure correct shape (32 channels)
            if eeg_data.shape[0] != config.eeg_n_channels:
                print(f"Warning: {file_info['filename']} has {eeg_data.shape[0]} channels, expected {config.eeg_n_channels}")
                continue

            # Compute entropy (returns shape (32,))
            channel_entropies = calculator.compute(eeg_data)

            row = {
                'pair_id': file_info['pair_id'],
                'player': file_info['player'],
                'trial_idx': file_info['trial_idx'],
                'condition': file_info['condition'],
                'mean_entropy': channel_entropies.mean()
            }

            # Add per-channel values
            for ch_idx, ch_name in enumerate(STANDARD_32_CHANNELS):
                row[ch_name] = channel_entropies[ch_idx]

            results.append(row)

        except Exception as e:
            print(f"Error processing {file_info['filename']}: {e}")
            continue

    df = pd.DataFrame(results)
    print(f"Computed entropy for {len(df)} EEG trials")
    return df


def compute_summary_statistics(
    df: pd.DataFrame,
    value_col: str = 'entropy',
    group_cols: List[str] = None
) -> pd.DataFrame:
    """
    Compute summary statistics per subject and condition.

    Parameters
    ----------
    df : pd.DataFrame
        Raw entropy data
    value_col : str
        Column containing entropy values
    group_cols : list
        Columns to group by. Default: ['pair_id', 'player', 'condition']

    Returns
    -------
    pd.DataFrame
        Summary statistics with mean, std, min, max, n_trials
    """
    if group_cols is None:
        group_cols = ['pair_id', 'player', 'condition']

    summary = df.groupby(group_cols)[value_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('n_trials', 'count')
    ]).reset_index()

    return summary


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_analysis(
    modality: str = 'both',
    use_mock: bool = False,
    config: Optional[Config] = None
):
    """
    Run the full entropy analysis pipeline.

    Parameters
    ----------
    modality : str
        'gaze', 'eeg', or 'both'
    use_mock : bool
        If True, use mock data instead of real data
    config : Config, optional
        Configuration object. If None, uses defaults.
    """
    if config is None:
        config = Config()

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 60)
    print(f"Entropy Analysis Pipeline")
    print(f"Started at: {timestamp}")
    print(f"Modality: {modality}")
    print(f"Use mock data: {use_mock}")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load or Generate Data
    # =========================================================================

    if use_mock:
        print("\n[Step 1] Generating mock data...")
        gaze_df, eeg_df = generate_mock_data(
            n_subjects=5,
            output_dir=config.output_dir
        )
    else:
        gaze_df = None
        eeg_df = None

        if modality in ['gaze', 'both']:
            print("\n[Step 1a] Analyzing Gaze data...")
            gaze_df = analyze_gaze_entropy(config)

        if modality in ['eeg', 'both']:
            print("\n[Step 1b] Analyzing EEG data...")
            eeg_df = analyze_eeg_entropy(config)

    # =========================================================================
    # Step 2: Save Raw Results to CSV
    # =========================================================================

    print("\n[Step 2] Saving raw results to CSV...")

    if gaze_df is not None and len(gaze_df) > 0:
        gaze_raw_path = Path(config.output_dir) / "gaze_entropy_raw.csv"
        gaze_df.to_csv(gaze_raw_path, index=False)
        print(f"  Saved: {gaze_raw_path}")

        # Summary statistics
        gaze_summary = compute_summary_statistics(
            gaze_df, value_col='spatial_entropy'
        )
        gaze_summary_path = Path(config.output_dir) / "gaze_entropy_summary.csv"
        gaze_summary.to_csv(gaze_summary_path, index=False)
        print(f"  Saved: {gaze_summary_path}")

    if eeg_df is not None and len(eeg_df) > 0:
        eeg_raw_path = Path(config.output_dir) / "eeg_entropy_raw_corrected.csv"
        eeg_df.to_csv(eeg_raw_path, index=False)
        print(f"  Saved: {eeg_raw_path}")

        # Summary statistics (using mean_entropy)
        eeg_summary = compute_summary_statistics(
            eeg_df, value_col='mean_entropy'
        )
        eeg_summary_path = Path(config.output_dir) / "eeg_entropy_summary.csv"
        eeg_summary.to_csv(eeg_summary_path, index=False)
        print(f"  Saved: {eeg_summary_path}")

    # =========================================================================
    # Step 3: Generate Visualizations
    # =========================================================================

    print("\n[Step 3] Generating visualizations...")
    setup_academic_style()

    # --- Gaze Visualizations ---
    if gaze_df is not None and len(gaze_df) > 0:
        # Create subject_id column for plotting
        gaze_df['subject_id'] = gaze_df['pair_id'].astype(str) + '_' + gaze_df['player']

        # Box plot
        print("  Generating Gaze box plot...")
        plot_entropy_boxplot(
            gaze_df,
            x='pair_id',
            y='spatial_entropy',
            hue='condition',
            title='Gaze Spatial Entropy by Subject',
            ylabel='Spatial Entropy (bits)',
            xlabel='Pair ID',
            save_path=str(Path(config.figure_dir) / "fig_gaze_boxplot.pdf"),
            show=False
        )

        # KDE plot (group-level)
        print("  Generating Gaze KDE plot...")
        plot_entropy_kde(
            gaze_df,
            value_col='spatial_entropy',
            group_col='condition',
            title='Gaze Spatial Entropy Distribution by Condition',
            xlabel='Spatial Entropy (bits)',
            save_path=str(Path(config.figure_dir) / "fig_gaze_kde.pdf"),
            show=False
        )

        # Violin plot
        print("  Generating Gaze violin plot...")
        plot_entropy_violin(
            gaze_df,
            x='condition',
            y='spatial_entropy',
            title='Gaze Spatial Entropy by Condition',
            ylabel='Spatial Entropy (bits)',
            save_path=str(Path(config.figure_dir) / "fig_gaze_violin.pdf"),
            show=False
        )

    # --- EEG Visualizations ---
    if eeg_df is not None and len(eeg_df) > 0:
        # Box plot
        print("  Generating EEG box plot...")
        plot_entropy_boxplot(
            eeg_df,
            x='pair_id',
            y='mean_entropy',
            hue='condition',
            title='EEG Spectral Entropy by Subject',
            ylabel='Spectral Entropy (bits)',
            xlabel='Pair ID',
            save_path=str(Path(config.figure_dir) / "fig_eeg_boxplot.pdf"),
            show=False
        )

        # KDE plot (group-level)
        print("  Generating EEG KDE plot...")
        plot_entropy_kde(
            eeg_df,
            value_col='mean_entropy',
            group_col='condition',
            title='EEG Spectral Entropy Distribution by Condition',
            xlabel='Spectral Entropy (bits)',
            save_path=str(Path(config.figure_dir) / "fig_eeg_kde.pdf"),
            show=False
        )

        # Violin plot
        print("  Generating EEG violin plot...")
        plot_entropy_violin(
            eeg_df,
            x='condition',
            y='mean_entropy',
            title='EEG Spectral Entropy by Condition',
            ylabel='Spectral Entropy (bits)',
            save_path=str(Path(config.figure_dir) / "fig_eeg_violin.pdf"),
            show=False
        )

        # Topomap (average per condition)
        print("  Generating EEG topomaps...")
        for condition in config.conditions:
            cond_data = eeg_df[eeg_df['condition'] == condition]
            if len(cond_data) > 0:
                # Average entropy per channel
                channel_means = cond_data[STANDARD_32_CHANNELS].mean().values
                plot_entropy_topomap(
                    channel_means,
                    title=f'EEG Spectral Entropy - {condition}',
                    save_path=str(Path(config.figure_dir) / f"fig_eeg_topomap_{condition.lower()}.pdf"),
                    show=False
                )

    # --- Cross-modality Correlation ---
    if (gaze_df is not None and eeg_df is not None and
        len(gaze_df) > 0 and len(eeg_df) > 0):

        print("  Generating cross-modality correlation plot...")

        # Merge gaze and EEG data on matching keys
        merge_keys = ['pair_id', 'player', 'trial_idx', 'condition']

        # Ensure both dataframes have the necessary columns
        gaze_for_merge = gaze_df[merge_keys + ['spatial_entropy']].copy()
        eeg_for_merge = eeg_df[merge_keys + ['mean_entropy']].copy()

        merged_df = pd.merge(
            gaze_for_merge,
            eeg_for_merge,
            on=merge_keys,
            how='inner'
        )

        if len(merged_df) > 0:
            merged_df.rename(columns={
                'spatial_entropy': 'gaze_entropy',
                'mean_entropy': 'eeg_entropy'
            }, inplace=True)

            plot_entropy_correlation(
                merged_df,
                x_col='gaze_entropy',
                y_col='eeg_entropy',
                hue_col='condition',
                title='Gaze vs EEG Entropy Correlation',
                save_path=str(Path(config.figure_dir) / "fig_correlation.pdf"),
                show=False
            )

            # Save merged data
            merged_path = Path(config.output_dir) / "cross_modality_entropy.csv"
            merged_df.to_csv(merged_path, index=False)
            print(f"  Saved: {merged_path}")

    # =========================================================================
    # Step 4: Print Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    print(f"\nOutput directory: {config.output_dir}")
    print(f"Figure directory: {config.figure_dir}")

    if gaze_df is not None and len(gaze_df) > 0:
        print(f"\nGaze Entropy Summary:")
        print(gaze_df.groupby('condition')['spatial_entropy'].describe().round(4))

    if eeg_df is not None and len(eeg_df) > 0:
        print(f"\nEEG Entropy Summary:")
        print(eeg_df.groupby('condition')['mean_entropy'].describe().round(4))

    print("\n" + "=" * 60)


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Entropy Analysis for Multimodal Physiological Signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_entropy.py --modality eeg
  python analyze_entropy.py --modality gaze
  python analyze_entropy.py --modality both
  python analyze_entropy.py --modality both --use_mock
        """
    )

    parser.add_argument(
        '--modality',
        type=str,
        choices=['gaze', 'eeg', 'both'],
        default='both',
        help="Which modality to analyze: 'gaze', 'eeg', or 'both' (default: both)"
    )

    parser.add_argument(
        '--use_mock',
        action='store_true',
        help="Use mock data for testing instead of real data"
    )

    parser.add_argument(
        '--eeg_path',
        type=str,
        default=None,
        help="Override default EEG data path"
    )

    parser.add_argument(
        '--gaze_path',
        type=str,
        default=None,
        help="Override default Gaze data path"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Override default output directory"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Create config and apply overrides
    config = Config()

    if args.eeg_path:
        config.eeg_data_path = args.eeg_path

    if args.gaze_path:
        config.gaze_data_path = args.gaze_path

    if args.output_dir:
        config.output_dir = args.output_dir
        config.figure_dir = str(Path(args.output_dir) / "figures")

    # Run analysis
    run_analysis(
        modality=args.modality,
        use_mock=args.use_mock,
        config=config
    )
