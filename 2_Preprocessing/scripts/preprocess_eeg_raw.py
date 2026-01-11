"""
EEG Raw Data Preprocessing Script - No Preprocessing, No Windowing

This script converts raw EEG CSV files to .npy format WITHOUT any preprocessing.
Each trial is kept as a complete sample (no windowing).

Split Strategy: By participant pairs
- Train: Pairs not in val_pairs
- Val: Pairs in val_pairs (default: [33, 34, 35, 36, 37, 38, 39, 40])

Output Structure:
    EEGseg_raw/
    ├── train/
    │   ├── eeg1.npy          # (N, 32, 3250) Player 1 trials
    │   ├── eeg2.npy          # (N, 32, 3250) Player 2 trials
    │   ├── labels.npy        # (N,) class labels
    │   └── metadata.json     # trial metadata
    └── val/
        └── ...

Usage:
    python 2_Preprocessing/scripts/preprocess_eeg_raw.py
    python 2_Preprocessing/scripts/preprocess_eeg_raw.py --output 1_Data/datasets/EEGseg_raw
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    'num_channels': 32,
    'expected_timepoints': 3250,  # 13 seconds @ 250 Hz
    'sampling_rate': 250,
    'val_pairs': [33, 34, 35, 36, 37, 38, 39, 40],
    'label2id': {
        'Single': 0,
        'Competition': 1,
        'Cooperation': 2
    }
}


# =============================================================================
# Data Loading (No Preprocessing)
# =============================================================================

def load_eeg_csv_raw(path: Path, num_channels: int = 32) -> Optional[np.ndarray]:
    """
    Load EEG from CSV file WITHOUT any preprocessing.

    Args:
        path: Path to CSV file
        num_channels: Expected number of channels

    Returns:
        eeg: (C, T) numpy array, or None if loading fails
    """
    try:
        df = pd.read_csv(path, header=None)
        eeg = df.values.astype(np.float32)

        # Ensure shape is (C, T) where C=32
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)
        elif eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T

        # Validate channel count
        if eeg.shape[0] != num_channels:
            print(f"[Warning] {path.name}: Expected {num_channels} channels, got {eeg.shape[0]}")
            if eeg.shape[0] > num_channels:
                eeg = eeg[:num_channels, :]
            else:
                # Pad with zeros if fewer channels
                pad = np.zeros((num_channels - eeg.shape[0], eeg.shape[1]), dtype=np.float32)
                eeg = np.vstack([eeg, pad])

        return eeg

    except Exception as e:
        print(f"[Error] Failed to load {path}: {e}")
        return None


# =============================================================================
# Sample Processing
# =============================================================================

def process_single_sample(args: Tuple) -> Optional[Dict]:
    """
    Process a single sample (pair of EEG files).

    Args:
        args: (sample_dict, eeg_base_path, config)

    Returns:
        Dict with trial data and metadata, or None if failed
    """
    sample, eeg_base_path, config = args

    try:
        # Construct paths
        player1_path = Path(eeg_base_path) / f"{sample['player1']}.csv"
        player2_path = Path(eeg_base_path) / f"{sample['player2']}.csv"

        if not player1_path.exists():
            return None
        if not player2_path.exists():
            return None

        # Load EEG (raw, no preprocessing)
        eeg1 = load_eeg_csv_raw(player1_path, config['num_channels'])
        eeg2 = load_eeg_csv_raw(player2_path, config['num_channels'])

        if eeg1 is None or eeg2 is None:
            return None

        # Align lengths (in case of minor differences)
        min_len = min(eeg1.shape[1], eeg2.shape[1])
        eeg1 = eeg1[:, :min_len]
        eeg2 = eeg2[:, :min_len]

        # Validate timepoints
        expected_t = config['expected_timepoints']
        if min_len != expected_t:
            # Pad or truncate to expected length
            if min_len < expected_t:
                # Pad with zeros
                pad1 = np.zeros((config['num_channels'], expected_t - min_len), dtype=np.float32)
                pad2 = np.zeros((config['num_channels'], expected_t - min_len), dtype=np.float32)
                eeg1 = np.hstack([eeg1, pad1])
                eeg2 = np.hstack([eeg2, pad2])
            else:
                # Truncate
                eeg1 = eeg1[:, :expected_t]
                eeg2 = eeg2[:, :expected_t]

        # Create result
        result = {
            'eeg1': eeg1,  # (32, 3250)
            'eeg2': eeg2,  # (32, 3250)
            'label': sample['class'],
            'pair': sample.get('pair', -1),
            'player1': sample['player1'],
            'player2': sample['player2']
        }

        return result

    except Exception as e:
        print(f"[Error] Processing {sample.get('player1', 'unknown')}: {e}")
        return None


# =============================================================================
# Main Processing
# =============================================================================

def process_and_save_split(
    metadata: List[Dict],
    eeg_base_path: str,
    output_dir: Path,
    config: Dict,
    split_name: str = "train",
    num_workers: int = 4
):
    """
    Process a dataset split and save to disk.

    Args:
        metadata: List of sample metadata dicts
        eeg_base_path: Base path for EEG files
        output_dir: Output directory
        config: Processing config
        split_name: Name of split (train/val)
        num_workers: Number of parallel workers
    """
    print(f"\n[Process] Processing {split_name} split ({len(metadata)} samples)...")

    # Prepare arguments for parallel processing
    args_list = [(sample, eeg_base_path, config) for sample in metadata]

    # Process samples
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
        # Sequential processing
        for args in tqdm(args_list, desc=f"Processing {split_name}"):
            result = process_single_sample(args)
            if result is not None:
                all_results.append(result)

    if len(all_results) == 0:
        print(f"[Error] No valid samples found for {split_name} split!")
        return

    # Aggregate data
    print(f"[Process] Aggregating {len(all_results)} trials...")

    all_eeg1 = []
    all_eeg2 = []
    all_labels = []
    trial_metadata = []

    label2id = config['label2id']

    for result in all_results:
        all_eeg1.append(result['eeg1'])
        all_eeg2.append(result['eeg2'])
        all_labels.append(label2id[result['label']])
        trial_metadata.append({
            'pair': result['pair'],
            'player1': result['player1'],
            'player2': result['player2'],
            'class': result['label']
        })

    # Convert to numpy arrays
    eeg1_array = np.stack(all_eeg1, axis=0).astype(np.float32)
    eeg2_array = np.stack(all_eeg2, axis=0).astype(np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    print(f"[Process] {split_name} shapes: eeg1={eeg1_array.shape}, eeg2={eeg2_array.shape}")

    # Create output directory
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    print(f"[Process] Saving to {split_dir}...")
    np.save(split_dir / 'eeg1.npy', eeg1_array)
    np.save(split_dir / 'eeg2.npy', eeg2_array)
    np.save(split_dir / 'labels.npy', labels_array)

    # Create class distribution
    id2label = {v: k for k, v in label2id.items()}
    class_counts = Counter(all_labels)
    class_distribution = {id2label[k]: int(v) for k, v in class_counts.items()}

    # Save metadata
    with open(split_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({
            'num_trials': len(all_labels),
            'shape': {
                'eeg1': list(eeg1_array.shape),
                'eeg2': list(eeg2_array.shape),
                'labels': list(labels_array.shape)
            },
            'class_distribution': class_distribution,
            'config': {
                'num_channels': config['num_channels'],
                'timepoints': config['expected_timepoints'],
                'sampling_rate': config['sampling_rate'],
                'preprocessing': 'none',
                'windowing': 'none'
            },
            'pairs': sorted(list(set(m['pair'] for m in trial_metadata))),
            'trials': trial_metadata
        }, f, indent=2, ensure_ascii=False)

    # Print statistics
    print(f"\n[Process] {split_name} Statistics:")
    print(f"  - Total trials: {len(all_labels)}")
    print(f"  - EEG1 shape: {eeg1_array.shape}")
    print(f"  - EEG2 shape: {eeg2_array.shape}")
    print(f"  - File sizes:")
    print(f"    - eeg1.npy: {(split_dir / 'eeg1.npy').stat().st_size / 1e6:.2f} MB")
    print(f"    - eeg2.npy: {(split_dir / 'eeg2.npy').stat().st_size / 1e6:.2f} MB")
    print(f"  - Pairs: {sorted(list(set(m['pair'] for m in trial_metadata)))}")
    print(f"  - Class distribution:")
    for class_name, count in sorted(class_distribution.items()):
        pct = count / len(all_labels) * 100
        print(f"    - {class_name}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Convert EEG CSV to NPY (raw, no preprocessing)')
    parser.add_argument('--metadata', type=str,
                       default='1_Data/metadata/complete_metadata.json',
                       help='Path to metadata JSON file')
    parser.add_argument('--eeg-path', type=str,
                       default='1_Data/datasets/EEGseg',
                       help='Path to EEG CSV files')
    parser.add_argument('--output', type=str,
                       default='1_Data/datasets/EEGseg_raw',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--val-pairs', type=int, nargs='+',
                       default=None,
                       help='Validation pairs (default: 33-40)')
    args = parser.parse_args()

    # Configuration
    config = DEFAULT_CONFIG.copy()
    if args.val_pairs:
        config['val_pairs'] = args.val_pairs

    # Paths
    metadata_path = PROJECT_ROOT / args.metadata
    eeg_base_path = PROJECT_ROOT / args.eeg_path
    output_dir = PROJECT_ROOT / args.output

    print("=" * 60)
    print("EEG Raw Data Conversion (No Preprocessing, No Windowing)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - EEG path: {eeg_base_path}")
    print(f"  - Output: {output_dir}")
    print(f"  - Workers: {args.workers}")
    print(f"  - Val pairs: {config['val_pairs']}")
    print(f"  - Expected shape: ({config['num_channels']}, {config['expected_timepoints']})")
    print(f"  - Preprocessing: None")
    print(f"  - Windowing: None (full trial)")

    # Validate paths
    if not metadata_path.exists():
        print(f"\n[Error] Metadata file not found: {metadata_path}")
        return 1

    if not eeg_base_path.exists():
        print(f"\n[Error] EEG path not found: {eeg_base_path}")
        return 1

    # Load metadata
    print(f"\n[Load] Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)

    print(f"[Load] Total samples in metadata: {len(all_metadata)}")

    # Split by pairs
    val_pairs = config['val_pairs']
    train_metadata = [m for m in all_metadata if m.get('pair', -1) not in val_pairs]
    val_metadata = [m for m in all_metadata if m.get('pair', -1) in val_pairs]

    print(f"\n[Split] Split by pairs:")
    print(f"  - Train samples: {len(train_metadata)}")
    print(f"  - Val samples: {len(val_metadata)}")
    print(f"  - Val pairs: {val_pairs}")

    # Analyze pairs in each split
    train_pairs = sorted(list(set(m.get('pair', -1) for m in train_metadata)))
    val_pairs_actual = sorted(list(set(m.get('pair', -1) for m in val_metadata)))
    print(f"  - Train pairs: {train_pairs}")
    print(f"  - Val pairs: {val_pairs_actual}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process train split
    process_and_save_split(
        metadata=train_metadata,
        eeg_base_path=str(eeg_base_path),
        output_dir=output_dir,
        config=config,
        split_name='train',
        num_workers=args.workers
    )

    # Process val split
    process_and_save_split(
        metadata=val_metadata,
        eeg_base_path=str(eeg_base_path),
        output_dir=output_dir,
        config=config,
        split_name='val',
        num_workers=args.workers
    )

    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)

    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── eeg1.npy    # (N_train, 32, 3250)")
    print(f"  │   ├── eeg2.npy    # (N_train, 32, 3250)")
    print(f"  │   ├── labels.npy  # (N_train,)")
    print(f"  │   └── metadata.json")
    print(f"  └── val/")
    print(f"      ├── eeg1.npy    # (N_val, 32, 3250)")
    print(f"      ├── eeg2.npy    # (N_val, 32, 3250)")
    print(f"      ├── labels.npy  # (N_val,)")
    print(f"      └── metadata.json")

    print("\nNext steps:")
    print("1. Update config: set preprocessed_dir to 'EEGseg_raw'")
    print("2. Update config: set in_timepoints to 3250")
    print("3. Run training: python 4_Experiments/scripts/train_eeg_hypereeg.py")

    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
