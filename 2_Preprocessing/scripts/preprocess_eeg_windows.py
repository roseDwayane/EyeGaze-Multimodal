"""
EEG Preprocessing Script - Prepare Sliding Windows for Training

This script preprocesses raw EEG CSV files and saves them as optimized .npy files
for fast training. All preprocessing (bandpass filter, CAR, z-score) is done once
and windows are pre-computed.

Supports two split modes:
    - "pair": Split by participant pairs (original method)
    - "stratified": Window-level stratified split by class (80/20)

Output Structure:
    EEGseg_preprocessed/
    ├── train/
    │   ├── eeg1.npy          # (N, 32, 1024) Player 1 windows
    │   ├── eeg2.npy          # (N, 32, 1024) Player 2 windows
    │   ├── labels.npy        # (N,) class labels
    │   └── metadata.json     # window metadata
    └── val/
        └── ...

Usage:
    python 2_Preprocessing/scripts/preprocess_eeg_windows.py
    python 2_Preprocessing/scripts/preprocess_eeg_windows.py --config path/to/config.yaml
    python 2_Preprocessing/scripts/preprocess_eeg_windows.py --split-mode stratified
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

# scipy for bandpass filter
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Warning] scipy not installed. Bandpass filtering disabled.")

# sklearn for stratified split
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[Warning] sklearn not installed. Stratified split requires sklearn.")

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Preprocessing Functions
# =============================================================================

def load_eeg_csv(path: Path, num_channels: int = 32) -> np.ndarray:
    """
    Load EEG from CSV file.

    Args:
        path: Path to CSV file
        num_channels: Expected number of channels

    Returns:
        eeg: (C, T) numpy array
    """
    df = pd.read_csv(path, header=None)
    eeg = df.values.astype(np.float32)

    # Ensure shape is (C, T)
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)
    elif eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T

    # Adjust channels
    if eeg.shape[0] > num_channels:
        eeg = eeg[:num_channels, :]
    elif eeg.shape[0] < num_channels:
        pad = np.zeros((num_channels - eeg.shape[0], eeg.shape[1]), dtype=np.float32)
        eeg = np.vstack([eeg, pad])

    return eeg


def bandpass_filter(
    eeg: np.ndarray,
    low_freq: float = 0.5,
    high_freq: float = 50.0,
    sampling_rate: int = 250,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to EEG signal.

    Args:
        eeg: (C, T) EEG signal
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz)
        sampling_rate: Sampling rate (Hz)
        order: Filter order

    Returns:
        Filtered EEG signal
    """
    if not SCIPY_AVAILABLE:
        return eeg

    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)

    try:
        b, a = signal.butter(order, [low, high], btype='band')
        eeg_filtered = signal.filtfilt(b, a, eeg, axis=1)
        return eeg_filtered.astype(np.float32)
    except Exception as e:
        print(f"[Warning] Bandpass filter failed: {e}")
        return eeg


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    """Apply Common Average Reference (CAR)."""
    car = eeg.mean(axis=0, keepdims=True)
    return (eeg - car).astype(np.float32)


def zscore_normalize(eeg: np.ndarray) -> np.ndarray:
    """Apply z-score normalization per channel."""
    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True) + 1e-8
    return ((eeg - mean) / std).astype(np.float32)


def preprocess_eeg(
    eeg: np.ndarray,
    sampling_rate: int = 250,
    low_freq: float = 0.5,
    high_freq: float = 50.0
) -> np.ndarray:
    """
    Full preprocessing pipeline: Bandpass -> CAR -> Z-score.

    Args:
        eeg: (C, T) raw EEG signal
        sampling_rate: Sampling rate in Hz
        low_freq: Bandpass low cutoff
        high_freq: Bandpass high cutoff

    Returns:
        Preprocessed EEG signal (C, T)
    """
    # 1. Bandpass filter
    eeg = bandpass_filter(eeg, low_freq, high_freq, sampling_rate)

    # 2. Common Average Reference
    eeg = common_average_reference(eeg)

    # 3. Z-score normalization
    eeg = zscore_normalize(eeg)

    return eeg


def extract_windows(
    eeg: np.ndarray,
    window_size: int = 1024,
    stride: int = 256
) -> List[np.ndarray]:
    """
    Extract sliding windows from EEG signal.

    Args:
        eeg: (C, T) EEG signal
        window_size: Window size in timepoints
        stride: Stride between windows

    Returns:
        List of (C, window_size) windows
    """
    windows = []
    num_timepoints = eeg.shape[1]

    if num_timepoints < window_size:
        return windows

    num_windows = (num_timepoints - window_size) // stride + 1

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        if end <= num_timepoints:
            windows.append(eeg[:, start:end])

    return windows


# =============================================================================
# Sample Processing (for parallel execution)
# =============================================================================

def process_single_sample(args: Tuple) -> Optional[Dict]:
    """
    Process a single sample (pair of EEG files).

    This function is designed for parallel processing.

    Args:
        args: (sample_dict, eeg_base_path, config)

    Returns:
        Dict with windows and metadata, or None if failed
    """
    sample, eeg_base_path, config = args

    try:
        # Construct paths
        player1_path = Path(eeg_base_path) / f"{sample['player1']}.csv"
        player2_path = Path(eeg_base_path) / f"{sample['player2']}.csv"

        if not player1_path.exists() or not player2_path.exists():
            return None

        # Load EEG
        eeg1 = load_eeg_csv(player1_path, config['num_channels'])
        eeg2 = load_eeg_csv(player2_path, config['num_channels'])

        # Preprocess
        eeg1 = preprocess_eeg(
            eeg1,
            sampling_rate=config['sampling_rate'],
            low_freq=config['filter_low'],
            high_freq=config['filter_high']
        )
        eeg2 = preprocess_eeg(
            eeg2,
            sampling_rate=config['sampling_rate'],
            low_freq=config['filter_low'],
            high_freq=config['filter_high']
        )

        # Align lengths
        min_len = min(eeg1.shape[1], eeg2.shape[1])
        eeg1 = eeg1[:, :min_len]
        eeg2 = eeg2[:, :min_len]

        # Extract windows
        windows1 = extract_windows(eeg1, config['window_size'], config['stride'])
        windows2 = extract_windows(eeg2, config['window_size'], config['stride'])

        if len(windows1) == 0 or len(windows1) != len(windows2):
            return None

        # Create result
        result = {
            'eeg1_windows': windows1,
            'eeg2_windows': windows2,
            'label': sample['class'],
            'pair': sample.get('pair', -1),
            'player1': sample['player1'],
            'player2': sample['player2'],
            'num_windows': len(windows1)
        }

        return result

    except Exception as e:
        return None


# =============================================================================
# Main Preprocessing
# =============================================================================

def preprocess_dataset(
    metadata: List[Dict],
    eeg_base_path: str,
    output_dir: Path,
    config: Dict,
    label2id: Dict[str, int],
    split_name: str = "train",
    num_workers: int = 4
):
    """
    Preprocess a dataset split and save to disk.

    Args:
        metadata: List of sample metadata dicts
        eeg_base_path: Base path for EEG files
        output_dir: Output directory
        config: Preprocessing config
        label2id: Label to ID mapping
        split_name: Name of split (train/val)
        num_workers: Number of parallel workers
    """
    print(f"\n[Preprocess] Processing {split_name} split ({len(metadata)} samples)...")

    # Prepare arguments for parallel processing
    args_list = [(sample, eeg_base_path, config) for sample in metadata]

    # Process samples (parallel or sequential)
    all_results = []

    if num_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_sample, args): i
                      for i, args in enumerate(args_list)}

            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Processing {split_name}"):
                result = future.result()
                if result is not None:
                    all_results.append(result)
    else:
        # Sequential processing (for debugging)
        for args in tqdm(args_list, desc=f"Processing {split_name}"):
            result = process_single_sample(args)
            if result is not None:
                all_results.append(result)

    if len(all_results) == 0:
        print(f"[Error] No valid samples found for {split_name} split!")
        return

    # Aggregate windows
    print(f"[Preprocess] Aggregating {len(all_results)} samples...")

    all_eeg1 = []
    all_eeg2 = []
    all_labels = []
    window_metadata = []

    for result in all_results:
        for i in range(result['num_windows']):
            all_eeg1.append(result['eeg1_windows'][i])
            all_eeg2.append(result['eeg2_windows'][i])
            all_labels.append(label2id[result['label']])
            window_metadata.append({
                'pair': result['pair'],
                'player1': result['player1'],
                'player2': result['player2'],
                'window_idx': i,
                'class': result['label']
            })

    # Convert to numpy arrays
    eeg1_array = np.stack(all_eeg1, axis=0).astype(np.float32)
    eeg2_array = np.stack(all_eeg2, axis=0).astype(np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    print(f"[Preprocess] {split_name} shapes: eeg1={eeg1_array.shape}, eeg2={eeg2_array.shape}, labels={labels_array.shape}")

    # Create output directory
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    print(f"[Preprocess] Saving to {split_dir}...")
    np.save(split_dir / 'eeg1.npy', eeg1_array)
    np.save(split_dir / 'eeg2.npy', eeg2_array)
    np.save(split_dir / 'labels.npy', labels_array)

    # Save metadata
    with open(split_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({
            'num_windows': len(all_labels),
            'shape': {
                'eeg1': list(eeg1_array.shape),
                'eeg2': list(eeg2_array.shape),
                'labels': list(labels_array.shape)
            },
            'class_distribution': {k: int(v) for k, v in pd.Series(all_labels).map(
                {v: k for k, v in label2id.items()}).value_counts().items()},
            'config': config,
            'windows': window_metadata[:100]  # Save first 100 for reference
        }, f, indent=2, ensure_ascii=False)

    # Print statistics
    print(f"\n[Preprocess] {split_name} Statistics:")
    print(f"  - Total windows: {len(all_labels)}")
    print(f"  - EEG1 shape: {eeg1_array.shape}")
    print(f"  - EEG2 shape: {eeg2_array.shape}")
    print(f"  - File sizes:")
    print(f"    - eeg1.npy: {(split_dir / 'eeg1.npy').stat().st_size / 1e9:.2f} GB")
    print(f"    - eeg2.npy: {(split_dir / 'eeg2.npy').stat().st_size / 1e9:.2f} GB")

    # Class distribution
    from collections import Counter
    class_counts = Counter(all_labels)
    id2label = {v: k for k, v in label2id.items()}
    print(f"  - Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"    - {id2label[class_id]}: {count}")


def save_split(
    eeg1_list: List[np.ndarray],
    eeg2_list: List[np.ndarray],
    labels_list: List[int],
    metadata_list: List[Dict],
    output_dir: Path,
    split_name: str,
    label2id: Dict[str, int],
    config: Dict
):
    """Save a data split to disk."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays
    eeg1_array = np.stack(eeg1_list, axis=0).astype(np.float32)
    eeg2_array = np.stack(eeg2_list, axis=0).astype(np.float32)
    labels_array = np.array(labels_list, dtype=np.int64)

    # Save arrays
    print(f"[Preprocess] Saving {split_name} to {split_dir}...")
    np.save(split_dir / 'eeg1.npy', eeg1_array)
    np.save(split_dir / 'eeg2.npy', eeg2_array)
    np.save(split_dir / 'labels.npy', labels_array)

    # Save metadata
    id2label = {v: k for k, v in label2id.items()}
    with open(split_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({
            'num_windows': len(labels_list),
            'shape': {
                'eeg1': list(eeg1_array.shape),
                'eeg2': list(eeg2_array.shape),
                'labels': list(labels_array.shape)
            },
            'class_distribution': {id2label[k]: int(v) for k, v in
                pd.Series(labels_list).value_counts().items()},
            'config': config,
            'windows': metadata_list[:100]
        }, f, indent=2, ensure_ascii=False)

    # Print statistics
    from collections import Counter
    class_counts = Counter(labels_list)
    print(f"\n[Preprocess] {split_name} Statistics:")
    print(f"  - Total windows: {len(labels_list)}")
    print(f"  - EEG1 shape: {eeg1_array.shape}")
    print(f"  - EEG2 shape: {eeg2_array.shape}")
    print(f"  - File sizes:")
    print(f"    - eeg1.npy: {(split_dir / 'eeg1.npy').stat().st_size / 1e9:.2f} GB")
    print(f"    - eeg2.npy: {(split_dir / 'eeg2.npy').stat().st_size / 1e9:.2f} GB")
    print(f"  - Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        pct = count / len(labels_list) * 100
        print(f"    - {id2label[class_id]}: {count} ({pct:.1f}%)")


def preprocess_stratified(
    all_metadata: List[Dict],
    eeg_base_path: str,
    output_dir: Path,
    config: Dict,
    label2id: Dict[str, int],
    val_ratio: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 4
):
    """
    Preprocess all data and split using stratified sampling.

    Args:
        all_metadata: All sample metadata
        eeg_base_path: Path to EEG files
        output_dir: Output directory
        config: Preprocessing config
        label2id: Label to ID mapping
        val_ratio: Validation set ratio (default: 0.2 = 20%)
        random_seed: Random seed for reproducibility
        num_workers: Number of parallel workers
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for stratified split. Install with: pip install scikit-learn")

    print(f"\n[Preprocess] Processing ALL samples ({len(all_metadata)}) for stratified split...")
    print(f"[Preprocess] Val ratio: {val_ratio}, Random seed: {random_seed}")

    # Prepare arguments for parallel processing
    args_list = [(sample, eeg_base_path, config) for sample in all_metadata]

    # Process all samples
    all_results = []

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_sample, args): i
                      for i, args in enumerate(args_list)}

            for future in tqdm(as_completed(futures), total=len(futures),
                              desc="Processing all samples"):
                result = future.result()
                if result is not None:
                    all_results.append(result)
    else:
        for args in tqdm(args_list, desc="Processing all samples"):
            result = process_single_sample(args)
            if result is not None:
                all_results.append(result)

    if len(all_results) == 0:
        print("[Error] No valid samples found!")
        return

    print(f"[Preprocess] Processed {len(all_results)} samples successfully")

    # Aggregate all windows
    print("[Preprocess] Aggregating windows...")
    all_eeg1 = []
    all_eeg2 = []
    all_labels = []
    all_window_metadata = []

    for result in all_results:
        for i in range(result['num_windows']):
            all_eeg1.append(result['eeg1_windows'][i])
            all_eeg2.append(result['eeg2_windows'][i])
            all_labels.append(label2id[result['label']])
            all_window_metadata.append({
                'pair': result['pair'],
                'player1': result['player1'],
                'player2': result['player2'],
                'window_idx': i,
                'class': result['label']
            })

    total_windows = len(all_labels)
    print(f"[Preprocess] Total windows: {total_windows}")

    # Stratified split
    print(f"[Preprocess] Performing stratified split ({1-val_ratio:.0%}/{val_ratio:.0%})...")

    indices = list(range(total_windows))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=all_labels,
        random_state=random_seed
    )

    print(f"[Preprocess] Train windows: {len(train_idx)}, Val windows: {len(val_idx)}")

    # Split data
    train_eeg1 = [all_eeg1[i] for i in train_idx]
    train_eeg2 = [all_eeg2[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    train_metadata = [all_window_metadata[i] for i in train_idx]

    val_eeg1 = [all_eeg1[i] for i in val_idx]
    val_eeg2 = [all_eeg2[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]
    val_metadata = [all_window_metadata[i] for i in val_idx]

    # Save splits
    save_split(train_eeg1, train_eeg2, train_labels, train_metadata,
               output_dir, 'train', label2id, config)
    save_split(val_eeg1, val_eeg2, val_labels, val_metadata,
               output_dir, 'val', label2id, config)


def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG data for training')
    parser.add_argument('--config', type=str, default='4_Experiments/configs/eeg_hypereeg.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='1_Data/datasets/EEGseg_preprocessed',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--split-mode', type=str, default=None,
                       choices=['pair', 'stratified'],
                       help='Split mode: "pair" (by participant pairs) or "stratified" (window-level)')
    parser.add_argument('--val-ratio', type=float, default=None,
                       help='Validation ratio for stratified split (default: 0.2)')
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    print(f"[Preprocess] Loading config from: {config_path}")

    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    # Extract relevant config
    data_config = full_config['data']
    preprocess_config = {
        'window_size': data_config['window_size'],
        'stride': data_config['stride'],
        'sampling_rate': data_config['sampling_rate'],
        'num_channels': data_config['num_channels'],
        'filter_low': data_config.get('filter_low', 0.5),
        'filter_high': data_config.get('filter_high', 50.0),
    }

    label2id = data_config['label2id']

    # Determine split mode (CLI arg overrides config)
    split_mode = args.split_mode or data_config.get('split_by', 'pair')
    val_ratio = args.val_ratio or data_config.get('val_ratio', 0.2)
    random_seed = data_config.get('random_seed', 42)

    print(f"[Preprocess] Split mode: {split_mode}")

    # Use local path
    eeg_base_path = PROJECT_ROOT / "1_Data" / "datasets" / "EEGseg"
    if not eeg_base_path.exists():
        # Fallback to config path
        eeg_base_path = Path(data_config['eeg_base_path'])

    print(f"[Preprocess] EEG base path: {eeg_base_path}")
    print(f"[Preprocess] Config: {preprocess_config}")

    # Load metadata
    metadata_path = PROJECT_ROOT / data_config['metadata_path']
    print(f"[Preprocess] Loading metadata from: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)

    print(f"[Preprocess] Total samples: {len(all_metadata)}")

    # Output directory
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Preprocess] Output directory: {output_dir}")

    # =========================================================================
    # Split Mode Selection
    # =========================================================================
    if split_mode == 'stratified':
        # Window-level stratified split
        print("\n" + "=" * 60)
        print("Mode: STRATIFIED (Window-level split by class)")
        print("=" * 60)

        preprocess_stratified(
            all_metadata=all_metadata,
            eeg_base_path=str(eeg_base_path),
            output_dir=output_dir,
            config=preprocess_config,
            label2id=label2id,
            val_ratio=val_ratio,
            random_seed=random_seed,
            num_workers=args.workers
        )

    else:  # split_mode == 'pair'
        # Original pair-based split
        print("\n" + "=" * 60)
        print("Mode: PAIR (Split by participant pairs)")
        print("=" * 60)

        val_pairs = data_config.get('val_pairs', [33, 34, 35, 36, 37, 38, 39, 40])
        train_metadata = [m for m in all_metadata if m.get('pair', -1) not in val_pairs]
        val_metadata = [m for m in all_metadata if m.get('pair', -1) in val_pairs]

        print(f"[Preprocess] Train samples: {len(train_metadata)}")
        print(f"[Preprocess] Val samples: {len(val_metadata)}")
        print(f"[Preprocess] Val pairs: {val_pairs}")

        # Process train split
        preprocess_dataset(
            metadata=train_metadata,
            eeg_base_path=str(eeg_base_path),
            output_dir=output_dir,
            config=preprocess_config,
            label2id=label2id,
            split_name='train',
            num_workers=args.workers
        )

        # Process val split
        preprocess_dataset(
            metadata=val_metadata,
            eeg_base_path=str(eeg_base_path),
            output_dir=output_dir,
            config=preprocess_config,
            label2id=label2id,
            split_name='val',
            num_workers=args.workers
        )

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print(f"Output saved to: {output_dir}")
    print(f"Split mode: {split_mode}")
    print("=" * 60)

    # Print usage instructions
    print("\nNext steps:")
    print("1. Set 'use_preprocessed: true' in config")
    print("2. Run training: python 4_Experiments/scripts/train_eeg_hypereeg.py")


if __name__ == '__main__':
    # Windows multiprocessing fix
    multiprocessing.freeze_support()
    main()
