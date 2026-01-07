"""
Run All HyperEEG Ablation Experiments

This script runs all ablation experiments for the HyperEEG Encoder model.
Each experiment trains the model with a different component configuration.

Ablation Configurations:
    - full: All components enabled (M1+M2+M3+M4)
    - baseline: All components disabled
    - no_sinc: Full model without SincConv (M1)
    - no_graph: Full model without Graph Attention (M2)
    - no_cross: Full model without Cross-Attention (M3)
    - no_uncertainty: Full model without Uncertainty Fusion (M4)

Usage:
    python run_experiments.py
    python run_experiments.py --experiments full baseline
    python run_experiments.py --skip-preprocessing
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

# All ablation experiment configurations
ABLATION_EXPERIMENTS = [
    "full",
    "baseline",
    "no_sinc",
    "no_graph",
    "no_cross",
    "no_uncertainty",
]

# Project paths
PROJECT_ROOT = Path(__file__).parent
TRAIN_SCRIPT = PROJECT_ROOT / "4_Experiments" / "scripts" / "train_eeg_hypereeg.py"
PREPROCESS_SCRIPT = PROJECT_ROOT / "2_Preprocessing" / "scripts" / "preprocess_eeg_windows.py"
PREPROCESSED_DIR = PROJECT_ROOT / "1_Data" / "datasets" / "EEGseg_preprocessed"


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[RUN] {description}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {description}")
        print(f"[ERROR] Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {description}")
        return False


def check_preprocessed_data() -> bool:
    """Check if preprocessed data exists."""
    train_dir = PREPROCESSED_DIR / "train"
    val_dir = PREPROCESSED_DIR / "val"

    required_files = [
        train_dir / "eeg1.npy",
        train_dir / "eeg2.npy",
        train_dir / "labels.npy",
        val_dir / "eeg1.npy",
        val_dir / "eeg2.npy",
        val_dir / "labels.npy",
    ]

    return all(f.exists() for f in required_files)


def run_preprocessing() -> bool:
    """Run the preprocessing script."""
    cmd = [sys.executable, str(PREPROCESS_SCRIPT)]
    return run_command(cmd, "Preprocessing EEG data")


def run_experiment(ablation_mode: str) -> bool:
    """Run a single ablation experiment."""
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--ablation", ablation_mode
    ]
    return run_command(cmd, f"Training HyperEEG ({ablation_mode})")


def main():
    parser = argparse.ArgumentParser(description="Run HyperEEG ablation experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=ABLATION_EXPERIMENTS,
        default=ABLATION_EXPERIMENTS,
        help="Specific experiments to run (default: all)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step (if data already preprocessed)"
    )
    args = parser.parse_args()

    print("="*60)
    print("HyperEEG Ablation Experiments Runner")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments to run: {args.experiments}")
    print(f"Total experiments: {len(args.experiments)}")
    print("="*60)

    # Step 1: Check/Run preprocessing
    if not args.skip_preprocessing:
        if check_preprocessed_data():
            print("\n[INFO] Preprocessed data found. Skipping preprocessing.")
        else:
            print("\n[INFO] Preprocessed data not found. Running preprocessing...")
            if not run_preprocessing():
                print("\n[ERROR] Preprocessing failed. Exiting.")
                sys.exit(1)
    else:
        print("\n[INFO] Skipping preprocessing (--skip-preprocessing flag)")

    # Step 2: Run experiments
    results = {}

    for i, ablation_mode in enumerate(args.experiments, 1):
        print(f"\n{'#'*60}")
        print(f"# Experiment {i}/{len(args.experiments)}: {ablation_mode}")
        print(f"{'#'*60}")

        success = run_experiment(ablation_mode)
        results[ablation_mode] = "SUCCESS" if success else "FAILED"

        if not success:
            print(f"\n[WARNING] Experiment {ablation_mode} failed. Continuing with next...")

    # Step 3: Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'Experiment':<20} {'Status':<10}")
    print("-"*30)

    success_count = 0
    for exp, status in results.items():
        status_icon = "[OK]" if status == "SUCCESS" else "[X]"
        print(f"{exp:<20} {status_icon} {status}")
        if status == "SUCCESS":
            success_count += 1

    print("-"*30)
    print(f"Total: {success_count}/{len(results)} experiments completed successfully")
    print("="*60)

    # Exit with error code if any experiment failed
    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
