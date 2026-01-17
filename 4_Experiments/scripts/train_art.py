"""
Training script for Dual EEG Transformer
Following HuggingFace patterns with custom training loop for EEG signals
"""

import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from tqdm import tqdm
import logging
import wandb
from typing import Dict, Any

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3_Models" / "backbones"))  # Add backbones to path for imports

# Import using importlib for folders starting with numbers
import importlib.util

def import_module_from_path(module_name: str, file_path: str):
    """Import a module from a file path and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import DualEEGTransformer model
_model_module = import_module_from_path(
    "dual_eeg_transformer",
    str(PROJECT_ROOT / "3_Models" / "backbones" / "dual_eeg_transformer.py")
)
DualEEGTransformer = _model_module.DualEEGTransformer

# Import DualEEGDataset
_dataset_module = import_module_from_path(
    "dual_eeg_dataset",
    str(PROJECT_ROOT / "1_Data" / "processed" / "dual_eeg_dataset.py")
)
DualEEGDataset = _dataset_module.DualEEGDataset
collate_fn = _dataset_module.collate_fn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_datasets(config: Dict[str, Any]):
    """
    Prepare train and test datasets

    Args:
        config: Configuration dictionary

    Returns:
        train_dataset, test_dataset
    """
    logger.info("Loading dataset from JSON...")

    # Load dataset using HuggingFace datasets
    metadata_path = config['data']['metadata_path']
    datasets = load_dataset("json", data_files=metadata_path, split="train")

    # Limit samples if specified
    max_samples = config['data'].get('max_samples', None)
    if max_samples is not None and max_samples > 0:
        datasets = datasets.select(range(min(max_samples, len(datasets))))
        logger.info(f"Limited to {len(datasets)} samples for quick testing")

    logger.info(f"Total samples: {len(datasets)}")

    # Train/test split
    test_size = config['data']['train_test_split']
    seed = config['data']['random_seed']

    # Try stratified split, fall back to regular split if it fails
    try:
        split_datasets = datasets.train_test_split(
            test_size=test_size,
            seed=seed,
            stratify_by_column='class'
        )
    except (ValueError, KeyError):
        logger.warning("Stratified split not available, using regular split")
        split_datasets = datasets.train_test_split(
            test_size=test_size,
            seed=seed
        )

    logger.info(f"Train samples: {len(split_datasets['train'])}")
    logger.info(f"Test samples: {len(split_datasets['test'])}")

    # Create DualEEGDataset
    eeg_base_path = config['data']['eeg_base_path']
    label2id = config['data']['label2id']
    window_size = config['data']['window_size']
    stride = config['data']['stride']
    enable_preprocessing = config['data'].get('enable_preprocessing', True)

    train_dataset = DualEEGDataset(
        split_datasets['train'],
        eeg_base_path,
        label2id,
        window_size=window_size,
        stride=stride,
        enable_preprocessing=enable_preprocessing
    )

    test_dataset = DualEEGDataset(
        split_datasets['test'],
        eeg_base_path,
        label2id,
        window_size=window_size,
        stride=stride,
        enable_preprocessing=enable_preprocessing
    )

    return train_dataset, test_dataset


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_sym_loss: bool = False,
    use_ibs_loss: bool = False,
    use_ibs_cls_loss: bool = True,
    use_ibs_contrastive: bool = True,
    lambda_sym: float = 0.1,
    lambda_ibs: float = 0.1,
    lambda_ibs_cls: float = 0.5,
    lambda_ibs_contrastive: float = 0.3,
    model_has_ibs: bool = True,  # NEW: Check if model has IBS enabled
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_sym = 0.0
    total_loss_ibs = 0.0
    total_loss_ibs_cls = 0.0
    total_loss_ibs_contrastive = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        eeg1 = batch['eeg1'].to(device)  # (B, C, T)
        eeg2 = batch['eeg2'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(eeg1, eeg2, labels)

        loss_ce = outputs['loss_ce']

        # Optional losses
        loss_sym = torch.tensor(0.0, device=device)
        loss_ibs = torch.tensor(0.0, device=device)
        loss_ibs_cls = torch.tensor(0.0, device=device)
        loss_ibs_contrastive = torch.tensor(0.0, device=device)

        if use_sym_loss:
            # Symmetry loss: encourages similar CLS representations
            cls1 = outputs['cls1']
            cls2 = outputs['cls2']
            loss_sym = model.module.compute_symmetry_loss(cls1, cls2) if hasattr(model, 'module') else model.compute_symmetry_loss(cls1, cls2)

        # IBS-related losses only if model has IBS enabled
        if model_has_ibs and 'ibs_token' in outputs:
            if use_ibs_loss:
                # IBS alignment loss: aligns IBS token with CLS tokens
                ibs_token = outputs['ibs_token']
                cls1 = outputs['cls1']
                cls2 = outputs['cls2']
                loss_ibs = model.module.compute_ibs_alignment_loss(ibs_token, cls1, cls2) if hasattr(model, 'module') else model.compute_ibs_alignment_loss(ibs_token, cls1, cls2)

            if use_ibs_cls_loss and 'loss_ibs_cls' in outputs:
                # IBS 直接分類 loss
                loss_ibs_cls = outputs['loss_ibs_cls']

            if use_ibs_contrastive:
                # IBS 對比學習 loss
                ibs_token = outputs['ibs_token']
                loss_ibs_contrastive = model.module.compute_ibs_contrastive_loss(ibs_token, labels) if hasattr(model, 'module') else model.compute_ibs_contrastive_loss(ibs_token, labels)

        # Total loss
        loss = (loss_ce +
                lambda_sym * loss_sym +
                lambda_ibs * loss_ibs +
                lambda_ibs_cls * loss_ibs_cls +
                lambda_ibs_contrastive * loss_ibs_contrastive)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_ce += loss_ce.item()
        total_loss_sym += loss_sym.item()
        total_loss_ibs += loss_ibs.item()
        total_loss_ibs_cls += loss_ibs_cls.item()
        total_loss_ibs_contrastive += loss_ibs_contrastive.item()
        num_batches += 1

        postfix_dict = {
            'loss': loss.item(),
            'loss_ce': loss_ce.item()
        }
        if use_sym_loss:
            postfix_dict['loss_sym'] = loss_sym.item()
        if model_has_ibs:
            if use_ibs_loss:
                postfix_dict['loss_ibs'] = loss_ibs.item()
            if use_ibs_cls_loss:
                postfix_dict['loss_ibs_cls'] = loss_ibs_cls.item()
            if use_ibs_contrastive:
                postfix_dict['loss_ibs_contra'] = loss_ibs_contrastive.item()

        pbar.set_postfix(postfix_dict)

    return {
        'train/loss': total_loss / num_batches,
        'train/loss_ce': total_loss_ce / num_batches,
        'train/loss_sym': total_loss_sym / num_batches,
        'train/loss_ibs': total_loss_ibs / num_batches,
        'train/loss_ibs_cls': total_loss_ibs_cls / num_batches,
        'train/loss_ibs_contrastive': total_loss_ibs_contrastive / num_batches
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch in pbar:
            eeg1 = batch['eeg1'].to(device)
            eeg2 = batch['eeg2'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(eeg1, eeg2, labels)

            loss = outputs['loss']
            logits = outputs['logits']

            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Use custom compute_metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    metrics = {
        'eval/loss': total_loss / num_batches,
        'eval/accuracy': accuracy,
        'eval/precision': precision,
        'eval/recall': recall,
        'eval/f1': f1
    }

    return metrics


def main(args):
    # Load config
    config = load_config(args.config)

    # Set seed
    seed = config['system']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(config)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['per_device_train_batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    # ===== Read ablation config =====
    ablation_config = config.get('ablation', {})

    # Determine IBS mode from ablation config
    ibs_mode = ablation_config.get('ibs_mode', 'robust')
    use_robust_ibs = (ibs_mode == 'robust')

    # Create model
    model = DualEEGTransformer(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_labels'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        dropout=config['training']['dropout'],
        max_len=config['data']['window_size'] // 4,  # After conv downsampling
        conv_kernel_size=config['model']['conv_kernel_size'],
        conv_stride=config['model']['conv_stride'],
        conv_layers=config['model']['conv_layers'],
        sampling_rate=config['data']['sampling_rate'],
        # Spectrogram parameters
        use_spectrogram=ablation_config.get('use_spectrogram', True),
        spec_n_fft=config['model'].get('spec_n_fft', 128),
        spec_hop_length=config['model'].get('spec_hop_length', 64),
        spec_freq_bins=config['model'].get('spec_freq_bins', 64),
        # IBS tokenization mode
        use_robust_ibs=use_robust_ibs,
        # ===== Ablation Study Parameters =====
        use_ibs=ablation_config.get('use_ibs', True),
        use_cross_attention=ablation_config.get('use_cross_attention', True),
        ibs_instance_norm=ablation_config.get('ibs_instance_norm', True),
        ibs_feature_type=ablation_config.get('ibs_feature_type', 'all'),
    )

    # Log ablation settings
    logger.info(f"Ablation Settings:")
    logger.info(f"  - use_spectrogram: {ablation_config.get('use_spectrogram', True)}")
    logger.info(f"  - use_ibs: {ablation_config.get('use_ibs', True)}")
    logger.info(f"  - ibs_mode: {ibs_mode}")
    logger.info(f"  - ibs_instance_norm: {ablation_config.get('ibs_instance_norm', True)}")
    logger.info(f"  - ibs_feature_type: {ablation_config.get('ibs_feature_type', 'all')}")
    logger.info(f"  - use_cross_attention: {ablation_config.get('use_cross_attention', True)}")

    model = model.to(device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    num_epochs = config['training']['num_train_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Wandb
    if config['training']['report_to'] == ['wandb']:
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['run_name'],
            tags=config['wandb']['tags'],
            notes=config['wandb']['notes'],
            config=config
        )

    # Training loop
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_sym_loss=config['training'].get('use_sym_loss', False),
            use_ibs_loss=config['training'].get('use_ibs_loss', False),
            use_ibs_cls_loss=config['training'].get('use_ibs_cls_loss', True),
            use_ibs_contrastive=config['training'].get('use_ibs_contrastive', True),
            lambda_sym=config['training'].get('lambda_sym', 0.1),
            lambda_ibs=config['training'].get('lambda_ibs', 0.1),
            lambda_ibs_cls=config['training'].get('lambda_ibs_cls', 0.5),
            lambda_ibs_contrastive=config['training'].get('lambda_ibs_contrastive', 0.3),
            model_has_ibs=ablation_config.get('use_ibs', True),  # Pass ablation setting
        )

        # Evaluate
        eval_metrics = evaluate(model, test_loader, device)

        # Log metrics
        all_metrics = {**train_metrics, **eval_metrics, 'epoch': epoch+1}

        for key, value in all_metrics.items():
            if key != 'epoch':
                logger.info(f"{key}: {value:.4f}")

        if config['training']['report_to'] == ['wandb']:
            wandb.log(all_metrics)

        # Save best model
        if eval_metrics['eval/f1'] > best_f1:
            best_f1 = eval_metrics['eval/f1']
            best_epoch = epoch + 1

            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config
            }, best_model_path)

            logger.info(f"✓ Saved best model (F1: {best_f1:.4f}) to {best_model_path}")

        # Save checkpoint
        if (epoch + 1) % config['training']['save_every_n_epochs'] == 0:
            checkpoint_path = output_dir / f'checkpoint-epoch-{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': all_metrics,
                'config': config
            }, checkpoint_path)

            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Step scheduler
        scheduler.step()

    logger.info(f"\n{'='*50}")
    logger.info(f"Training completed!")
    logger.info(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    logger.info(f"{'='*50}")

    # Final evaluation on best model
    logger.info("\nLoading best model for final evaluation...")
    best_checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])

    final_metrics = evaluate(model, test_loader, device)

    logger.info("\nFinal Test Metrics:")
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value:.4f}")

    if config['training']['report_to'] == ['wandb']:
        wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual EEG Transformer")
    parser.add_argument(
        '--config',
        type=str,
        default='Experiments/configs/dual_eeg_transformer.yaml',
        help='Path to config file'
    )
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path to resume from')

    args = parser.parse_args()

    main(args)
