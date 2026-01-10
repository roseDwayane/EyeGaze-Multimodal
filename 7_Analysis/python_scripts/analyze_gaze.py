"""
Comprehensive Gaze Model Analysis Script

This script performs quantitative and qualitative analysis of trained
Early Fusion and Late Fusion ViT models for gaze-based interaction classification.

Uses refactored modular components from:
- 5_Metrics: ClassificationMetrics, FeatureExtractor
- 6_Utils: Visualizers, AttentionAnalyzer, ErrorAnalyzer, etc.

Output locations:
- Raw data (CSV): 7_Analysis/raw_result/{exp_name}/
- Publication figures (PDF/PNG): 7_Analysis/figures/{exp_name}/
- Tables: 7_Analysis/tables/

Author: Kong-Yi Chang
Date: 2026
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # 7_Analysis/python_scripts -> EyeGaze-Multimodal_new
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Dynamic Module Loading (for numeric folder names)
# =============================================================================

def load_module_from_path(module_path: Path, module_name: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load project modules dynamically
metrics_module = load_module_from_path(
    PROJECT_ROOT / "5_Metrics" / "classification_metrics.py",
    "classification_metrics"
)
feature_module = load_module_from_path(
    PROJECT_ROOT / "5_Metrics" / "feature_extractors.py",
    "feature_extractors"
)
visualizers_module = load_module_from_path(
    PROJECT_ROOT / "6_Utils" / "visualizers.py",
    "visualizers"
)
attention_module = load_module_from_path(
    PROJECT_ROOT / "6_Utils" / "attention_utils.py",
    "attention_utils"
)
error_module = load_module_from_path(
    PROJECT_ROOT / "6_Utils" / "error_analysis.py",
    "error_analysis"
)
learning_module = load_module_from_path(
    PROJECT_ROOT / "6_Utils" / "learning_curves.py",
    "learning_curves"
)
comparison_module = load_module_from_path(
    PROJECT_ROOT / "6_Utils" / "model_comparison.py",
    "model_comparison"
)

# Load model and dataset modules
early_fusion_module = load_module_from_path(
    PROJECT_ROOT / "3_Models" / "backbones" / "early_fusion_vit.py",
    "early_fusion_vit"
)
late_fusion_module = load_module_from_path(
    PROJECT_ROOT / "3_Models" / "backbones" / "late_fusion_vit.py",
    "late_fusion_vit"
)
dataset_module = load_module_from_path(
    PROJECT_ROOT / "1_Data" / "datasets" / "gaze_pair_dataset.py",
    "gaze_pair_dataset"
)

# Import classes
ClassificationMetrics = metrics_module.ClassificationMetrics
FeatureExtractor = feature_module.FeatureExtractor
AttentionAnalyzer = attention_module.AttentionAnalyzer
ErrorAnalyzer = error_module.ErrorAnalyzer
MechanismAnalyzer = error_module.MechanismAnalyzer
LearningCurveAnalyzer = learning_module.LearningCurveAnalyzer
ModelResults = comparison_module.ModelResults
MultiModelComparator = comparison_module.MultiModelComparator

EarlyFusionViT = early_fusion_module.EarlyFusionViT
LateFusionViT = late_fusion_module.LateFusionViT
GazePairDataset = dataset_module.GazePairDataset


# =============================================================================
# Constants
# =============================================================================

CLASS_NAMES = ["Single", "Competition", "Cooperation"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Utility Functions
# =============================================================================

def create_output_dirs(project_root: Path, exp_name: str) -> Dict[str, Path]:
    """Create output directories for analysis results."""
    paths = {
        'raw_result': project_root / "7_Analysis" / "raw_result" / exp_name,
        'figures': project_root / "7_Analysis" / "figures" / exp_name,
        'tables': project_root / "7_Analysis" / "tables",
        'comparison': project_root / "7_Analysis" / "figures" / "comparison"
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


# =============================================================================
# Main Analyzer Class
# =============================================================================

class GazeAnalyzer:
    """
    Comprehensive analyzer for Gaze classification models.

    Encapsulates all analysis functionality using modular components:
    - Quantitative: metrics, confusion matrix, ROC curves
    - Qualitative: attention visualization, t-SNE
    - Error analysis: pair-wise, mechanism validation
    - Learning curves
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        model_type: str,
        fusion_mode: Optional[str] = None,
        device: Optional[str] = None,
        exp_name: Optional[str] = None
    ):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_type = model_type
        self.fusion_mode = fusion_mode or self.config['model'].get('fusion_mode', 'concat')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.exp_name = exp_name or f"{model_type}fusion_{self.fusion_mode}"

        # Setup output directories
        self.paths = create_output_dirs(PROJECT_ROOT, self.exp_name)

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Load data
        print("Loading validation dataset...")
        self.val_loader = self._load_data()

        # Initialize analyzers (using refactored modules)
        self.metrics_calc = ClassificationMetrics(CLASS_NAMES)
        self.feature_extractor = FeatureExtractor(self.model, model_type, self.device, CLASS_NAMES)
        self.attention_analyzer = AttentionAnalyzer(self.model, model_type, self.device)
        self.error_analyzer = ErrorAnalyzer(CLASS_NAMES)
        self.mechanism_analyzer = MechanismAnalyzer(model_type, CLASS_NAMES)
        self.learning_analyzer = LearningCurveAnalyzer()

        # Storage for results
        self.predictions = None
        self.labels = None
        self.probabilities = None
        self.metadata = None

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_config = checkpoint.get('config', self.config).get('model', self.config['model'])

        if self.model_type == 'early':
            model = EarlyFusionViT(
                model_name=model_config.get('name', 'vit_base_patch16_224'),
                num_classes=model_config.get('num_classes', 3),
                pretrained=False,
                fusion_mode=model_config.get('fusion_mode', self.fusion_mode),
                weight_init_strategy=model_config.get('weight_init_strategy', 'duplicate')
            )
        else:
            model = LateFusionViT(
                model_name=model_config.get('name', 'vit_base_patch16_224'),
                num_classes=model_config.get('num_classes', 3),
                pretrained=False,
                fusion_mode=model_config.get('fusion_mode', self.fusion_mode),
                dropout=model_config.get('dropout', 0.1)
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _load_data(self) -> DataLoader:
        """Load validation dataset with metadata."""
        data_config = self.config['data']

        val_transform = transforms.Compose([
            transforms.Resize((data_config.get('image_size', 224),
                              data_config.get('image_size', 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Load metadata
        metadata_path = PROJECT_ROOT / data_config['metadata_path']
        with open(metadata_path, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)

        val_pairs = data_config.get('val_pairs', [33, 34, 35, 36, 37, 38, 39, 40])
        val_metadata = [m for m in all_metadata if m['pair'] in val_pairs]

        label2id = data_config.get('label2id', {"Single": 0, "Competition": 1, "Cooperation": 2})

        val_dataset = GazePairDataset(
            metadata=val_metadata,
            image_base_path=data_config['image_base_path'],
            image_extension=data_config.get('image_extension', '.jpg'),
            label2id=label2id,
            transform=val_transform,
            return_metadata=True
        )

        def collate_fn(batch):
            img_a = torch.stack([item[0] for item in batch])
            img_b = torch.stack([item[1] for item in batch])
            labels = torch.tensor([item[2] for item in batch])
            metadata = [item[3] for item in batch]
            return img_a, img_b, labels, metadata

        return DataLoader(
            val_dataset,
            batch_size=self.config['training'].get('batch_size', 16),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

    def run_inference(self):
        """Run model inference on validation set."""
        print("\n[Step 1] Running inference...")

        self.model.eval()
        all_predictions, all_labels, all_probs, all_metadata = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Inference"):
                img_a, img_b, labels, metadata = batch
                img_a, img_b = img_a.to(self.device), img_b.to(self.device)

                logits = self.model(img_a, img_b)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.append(probs.cpu().numpy())
                all_metadata.extend(metadata)

        self.predictions = np.array(all_predictions)
        self.labels = np.array(all_labels)
        self.probabilities = np.concatenate(all_probs, axis=0)
        self.metadata = all_metadata

        # Save predictions CSV
        self.metrics_calc.save_predictions_csv(
            self.labels, self.predictions, self.probabilities,
            self.paths['raw_result'] / 'predictions.csv',
            metadata=[{'pair_id': m['pair']} for m in self.metadata]
        )

    def run_quantitative_analysis(self):
        """Run quantitative analysis: metrics, confusion matrix, ROC."""
        print("\n[Step 2] Quantitative Analysis...")

        # Compute metrics
        metrics = self.metrics_calc.compute_metrics(self.labels, self.predictions)
        self.metrics_calc.save_metrics_csv(metrics, self.paths['raw_result'] / 'metrics.csv')

        # Confusion matrix
        cm = self.metrics_calc.compute_confusion_matrix(self.labels, self.predictions)
        self.metrics_calc.save_confusion_matrix_csv(cm, self.paths['raw_result'] / 'conf_mat.csv')

        visualizers_module.plot_confusion_matrix(
            cm, CLASS_NAMES, self.paths['figures'] / 'fig_conf_mat.pdf',
            title=f'Confusion Matrix - {self.exp_name}'
        )

        # ROC curves
        roc_data = self.metrics_calc.compute_roc_data(self.labels, self.probabilities)
        self.metrics_calc.save_roc_data_csv(roc_data, self.paths['raw_result'] / 'roc_data.csv')

        visualizers_module.plot_roc_curves(
            roc_data, CLASS_NAMES, self.paths['figures'] / 'fig_roc_curves.pdf',
            title=f'ROC Curves - {self.exp_name}'
        )

        print(f"\n  Performance Summary:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 (Macro): {metrics['f1_macro']:.4f}")

    def run_qualitative_analysis(self):
        """Run qualitative analysis: features and t-SNE."""
        print("\n[Step 3] Qualitative Analysis...")

        # Extract features using refactored module
        features, labels, _ = self.feature_extractor.extract_features(
            self.val_loader, return_metadata=False
        )

        # t-SNE
        tsne_coords = self.feature_extractor.compute_tsne(features)
        self.feature_extractor.save_tsne_csv(
            tsne_coords, self.labels,
            self.paths['raw_result'] / 'tsne_coords.csv',
            predictions=self.predictions
        )

        visualizers_module.plot_tsne(
            tsne_coords, self.labels, self.predictions, CLASS_NAMES,
            self.paths['figures'] / 'fig_tsne.pdf',
            title=f't-SNE - {self.exp_name}'
        )

    def run_error_analysis(self):
        """Run error and mechanism analysis."""
        print("\n[Step 4] Error & Mechanism Analysis...")

        # Pair-wise analysis
        predictions_df = pd.DataFrame({
            'pair_id': [m['pair'] for m in self.metadata],
            'true_label': self.labels,
            'pred_label': self.predictions,
            'correct': self.labels == self.predictions
        })

        pair_stats = self.error_analyzer.analyze_pair_performance(predictions_df)
        self.error_analyzer.save_pair_stats_csv(
            pair_stats, self.paths['raw_result'] / 'pair_stats.csv'
        )

        overall_accuracy = (self.predictions == self.labels).mean()
        visualizers_module.plot_pair_accuracy(
            pair_stats, overall_accuracy,
            self.paths['figures'] / 'fig_pair_accuracy.png',
            title=f'Per-Pair Accuracy - {self.exp_name}'
        )

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print(f"\n{'='*60}")
        print(f"Starting Full Analysis: {self.exp_name}")
        print(f"{'='*60}")

        self.run_inference()
        self.run_quantitative_analysis()
        self.run_qualitative_analysis()
        self.run_error_analysis()

        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}")
        print(f"\nResults saved to:")
        print(f"  Raw data: {self.paths['raw_result']}")
        print(f"  Figures: {self.paths['figures']}")


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Gaze Model Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model analysis
  python analyze_gaze.py --config configs/gaze_earlyfusion.yaml \\
                         --checkpoint runs/best_model.pt \\
                         --model_type early

  # Multi-model comparison
  python analyze_gaze.py --compare \\
                         --checkpoints model1.pt model2.pt \\
                         --configs config1.yaml config2.yaml \\
                         --model_types early late \\
                         --labels "Early-Concat" "Late-Full"
        """
    )

    # Single model arguments
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['early', 'late'],
                       help='Model type: early or late fusion')
    parser.add_argument('--fusion_mode', type=str, default=None,
                       help='Override fusion mode from config')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name for output folders')

    # Multi-model comparison
    parser.add_argument('--compare', action='store_true',
                       help='Enable multi-model comparison mode')
    parser.add_argument('--checkpoints', nargs='+',
                       help='List of checkpoint paths')
    parser.add_argument('--configs', nargs='+',
                       help='List of config paths')
    parser.add_argument('--model_types', nargs='+',
                       help='Model types for each checkpoint')
    parser.add_argument('--labels', nargs='+',
                       help='Display labels for each model')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.compare:
        # Multi-model comparison mode
        if not all([args.checkpoints, args.configs, args.model_types, args.labels]):
            print("Error: --compare requires --checkpoints, --configs, --model_types, --labels")
            return

        print(f"\n{'='*60}")
        print("Multi-Model Comparison Mode")
        print(f"{'='*60}")

        results_list = []
        for ckpt, cfg, mtype, label in zip(args.checkpoints, args.configs,
                                           args.model_types, args.labels):
            print(f"\nAnalyzing: {label}")
            analyzer = GazeAnalyzer(
                config_path=cfg,
                checkpoint_path=ckpt,
                model_type=mtype,
                exp_name=label.replace(' ', '_').lower()
            )
            analyzer.run_inference()
            analyzer.run_quantitative_analysis()

            # Create ModelResults for comparison
            metrics = analyzer.metrics_calc.compute_metrics(
                analyzer.labels, analyzer.predictions
            )
            result = ModelResults(
                label, analyzer.labels, analyzer.predictions,
                analyzer.probabilities, metrics
            )
            results_list.append(result)

        # Run comparison
        comparator = MultiModelComparator(results_list, CLASS_NAMES)
        comparison_dir = PROJECT_ROOT / "7_Analysis" / "figures" / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        comparator.save_comparison_csv(PROJECT_ROOT / "7_Analysis" / "tables" / "table_comparison.csv")
        comparator.plot_metrics_comparison(comparison_dir / "fig_compare_metrics.pdf")
        comparator.plot_confusion_matrices(comparison_dir / "fig_compare_conf_mat.pdf")
        comparator.plot_roc_comparison(comparison_dir / "fig_compare_roc.pdf")

        print(f"\n{'='*60}")
        print("Comparison Complete!")
        print(f"Results saved to: {comparison_dir}")

    else:
        # Single model analysis
        if not all([args.config, args.checkpoint, args.model_type]):
            print("Error: Requires --config, --checkpoint, and --model_type")
            return

        analyzer = GazeAnalyzer(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            fusion_mode=args.fusion_mode,
            device=args.device,
            exp_name=args.exp_name
        )
        analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
