"""
Ablation Study Runner for Dual EEG Transformer

This script runs all ablation experiments defined in the research plan:
- A. Feature Contribution (5 experiments)
- B. IBS Tokenizer Design (4 experiments)
- C. Interaction & Loss (4 experiments)

Usage:
    python run_experiments/run_ablation_studies.py [--dry-run] [--experiments A,B,C] [--gpu 0]

Examples:
    # Run all experiments
    python run_experiments/run_ablation_studies.py

    # Dry run (only print commands)
    python run_experiments/run_ablation_studies.py --dry-run

    # Run only Feature Contribution experiments
    python run_experiments/run_ablation_studies.py --experiments A

    # Run specific experiments by name
    python run_experiments/run_ablation_studies.py --names baseline,full_model
"""

import os
import sys
import yaml
import copy
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "4_Experiments" / "configs" / "dual_eeg_transformer.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "4_Experiments" / "scripts" / "train_art.py"
OUTPUT_DIR = PROJECT_ROOT / "4_Experiments" / "runs" / "ablation_studies"


# ============================================================================
# Ablation Experiment Definitions
# ============================================================================

EXPERIMENTS = {
    # ===== A. Feature Contribution (輸入特徵的貢獻) =====
    "A1_baseline_temporal_only": {
        "description": "Baseline: Temporal Conv Only (no Spectrogram, no IBS)",
        "category": "A",
        "ablation": {
            "use_spectrogram": False,
            "use_ibs": False,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {},  # Use default training settings
    },
    "A2_plus_spectrogram": {
        "description": "+ Spectrogram (no IBS)",
        "category": "A",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": False,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {},
    },
    "A3_plus_ibs_scalar": {
        "description": "+ IBS (Old/Scalar, 1 token)",
        "category": "A",
        "ablation": {
            "use_spectrogram": False,
            "use_ibs": True,
            "ibs_mode": "scalar",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {},
    },
    "A4_plus_ibs_robust": {
        "description": "+ IBS (New/Robust Matrix, 42 tokens)",
        "category": "A",
        "ablation": {
            "use_spectrogram": False,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {},
    },
    "A5_full_model": {
        "description": "Full Model (Spectrogram + Robust IBS)",
        "category": "A",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {},
    },

    # ===== B. IBS Tokenizer Design (IBS Tokenizer 的設計驗證) =====
    "B1_no_instance_norm": {
        "description": "No Instance Normalization in RobustIBSTokenizer",
        "category": "B",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": False,  # Key change
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {},
    },
    "B2_phase_only": {
        "description": "Phase-based Features Only (PLV, PLI, wPLI, Phase_Diff) - 24 tokens",
        "category": "B",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "phase",  # Key change
            "use_cross_attention": True,
        },
        "training": {},
    },
    "B3_amplitude_only": {
        "description": "Amplitude-based Features Only (Coherence, Power_Corr, Time_Corr) - 18 tokens",
        "category": "B",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "amplitude",  # Key change
            "use_cross_attention": True,
        },
        "training": {},
    },
    "B4_full_ibs_baseline": {
        "description": "Full IBS (All 7 features) - Baseline for B experiments",
        "category": "B",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {},
    },

    # ===== C. Interaction & Loss (交互機制與 Loss) =====
    "C1_no_cross_attention": {
        "description": "No Cross-Brain Attention (replace with identity)",
        "category": "C",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": False,  # Key change
        },
        "training": {},
    },
    "C2_no_contrastive_loss": {
        "description": "No IBS Contrastive Loss (lambda_ibs_contrastive = 0)",
        "category": "C",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {
            "use_ibs_contrastive": False,  # Key change
            "lambda_ibs_contrastive": 0.0,
        },
    },
    "C3_no_ibs_cls_loss": {
        "description": "No IBS Classification Head Loss",
        "category": "C",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {
            "use_ibs_cls_loss": False,  # Key change
            "lambda_ibs_cls": 0.0,
        },
    },
    "C4_full_losses_baseline": {
        "description": "Full Model with All Losses - Baseline for C experiments",
        "category": "C",
        "ablation": {
            "use_spectrogram": True,
            "use_ibs": True,
            "ibs_mode": "robust",
            "ibs_instance_norm": True,
            "ibs_feature_type": "all",
            "use_cross_attention": True,
        },
        "training": {
            "use_ibs_contrastive": True,
            "use_ibs_cls_loss": True,
            "lambda_ibs_contrastive": 0.3,
            "lambda_ibs_cls": 1.0,
        },
    },
}


def load_base_config() -> Dict[str, Any]:
    """Load the base configuration from YAML file."""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_experiment_config(
    base_config: Dict[str, Any],
    experiment: Dict[str, Any],
    experiment_name: str
) -> Dict[str, Any]:
    """Create experiment-specific configuration."""
    config = copy.deepcopy(base_config)

    # Update ablation settings
    if "ablation" in experiment:
        if "ablation" not in config:
            config["ablation"] = {}
        config["ablation"].update(experiment["ablation"])

    # Update training settings
    if "training" in experiment:
        if "training" not in config:
            config["training"] = {}
        config["training"].update(experiment["training"])

    # Update output directory and wandb run name
    config["training"]["output_dir"] = str(OUTPUT_DIR / experiment_name)
    config["wandb"]["run_name"] = experiment_name

    # Add experiment tags
    category = experiment.get("category", "unknown")
    config["wandb"]["tags"] = [
        "ablation-study",
        f"category-{category}",
        experiment_name,
    ]
    config["wandb"]["notes"] = experiment.get("description", "")

    return config


def save_experiment_config(config: Dict[str, Any], experiment_name: str) -> Path:
    """Save experiment configuration to a temporary YAML file."""
    config_dir = OUTPUT_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"{experiment_name}.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return config_path


def run_experiment(
    experiment_name: str,
    experiment: Dict[str, Any],
    base_config: Dict[str, Any],
    dry_run: bool = False,
    gpu: Optional[int] = None
) -> bool:
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Description: {experiment.get('description', 'N/A')}")
    print(f"Category: {experiment.get('category', 'N/A')}")
    print(f"{'='*70}")

    # Create experiment config
    config = create_experiment_config(base_config, experiment, experiment_name)

    # Print ablation settings
    print("\nAblation Settings:")
    for key, value in config.get("ablation", {}).items():
        print(f"  - {key}: {value}")

    if experiment.get("training"):
        print("\nTraining Overrides:")
        for key, value in experiment["training"].items():
            print(f"  - {key}: {value}")

    # Save config
    config_path = save_experiment_config(config, experiment_name)
    print(f"\nConfig saved to: {config_path}")

    # Build command
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config", str(config_path)
    ]

    # Set GPU if specified
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"\nCommand: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return True

    # Run training
    print("\nStarting training...")
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            # capture_output=True,
            # text=True
        )

        if result.returncode == 0:
            print(f"\n✓ Experiment {experiment_name} completed successfully!")
            return True
        else:
            print(f"\n✗ Experiment {experiment_name} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n✗ Experiment {experiment_name} failed with error: {e}")
        return False


def filter_experiments(
    experiments: Dict[str, Dict],
    categories: Optional[List[str]] = None,
    names: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Filter experiments by category or name."""
    if categories:
        experiments = {
            name: exp for name, exp in experiments.items()
            if exp.get("category") in categories
        }

    if names:
        experiments = {
            name: exp for name, exp in experiments.items()
            if name in names or any(n in name for n in names)
        }

    return experiments


def print_experiment_summary(experiments: Dict[str, Dict]):
    """Print summary of experiments to run."""
    print("\n" + "="*70)
    print("ABLATION STUDY EXPERIMENTS")
    print("="*70)

    # Group by category
    by_category = {}
    for name, exp in experiments.items():
        cat = exp.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((name, exp))

    for cat in sorted(by_category.keys()):
        print(f"\n[Category {cat}]")
        for name, exp in by_category[cat]:
            desc = exp.get("description", "")
            print(f"  - {name}: {desc}")

    print(f"\nTotal: {len(experiments)} experiments")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run Dual EEG Transformer Ablation Studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--experiments", "-e",
        type=str,
        default=None,
        help="Comma-separated list of experiment categories (A,B,C)"
    )
    parser.add_argument(
        "--names", "-n",
        type=str,
        default=None,
        help="Comma-separated list of experiment names to run"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to use"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments and exit"
    )

    args = parser.parse_args()

    # Filter experiments
    experiments = EXPERIMENTS.copy()

    if args.experiments:
        categories = [c.strip().upper() for c in args.experiments.split(",")]
        experiments = filter_experiments(experiments, categories=categories)

    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        experiments = filter_experiments(experiments, names=names)

    if not experiments:
        print("No experiments match the specified filters!")
        return 1

    # Print summary
    print_experiment_summary(experiments)

    if args.list:
        return 0

    # Confirm before running
    if not args.dry_run:
        response = input(f"\nRun {len(experiments)} experiments? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    # Load base config
    base_config = load_base_config()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = {}
    start_time = datetime.now()

    for name, experiment in experiments.items():
        success = run_experiment(
            name,
            experiment,
            base_config,
            dry_run=args.dry_run,
            gpu=args.gpu
        )
        results[name] = success

    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"Total time: {duration}")
    print(f"Experiments run: {len(results)}")
    print(f"Successful: {sum(results.values())}")
    print(f"Failed: {len(results) - sum(results.values())}")

    if not args.dry_run:
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")

    print("="*70)

    # Return non-zero if any experiment failed
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
