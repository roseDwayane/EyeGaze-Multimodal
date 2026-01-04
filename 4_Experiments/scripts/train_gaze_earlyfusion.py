"""
Training Script for Early Fusion ViT - Gaze Heatmap Classification

This script trains the EarlyFusionViT model on dual gaze heatmap data
for social interaction classification (Single/Competition/Cooperation).

Features:
    - Weights & Biases logging
    - Checkpoint saving and resume training
    - Weighted loss for class imbalance
    - Cosine LR scheduler with warmup
    - Mixed precision training (FP16)

Usage:
    python 4_Experiments/scripts/train_gaze_earlyfusion.py
    python 4_Experiments/scripts/train_gaze_earlyfusion.py --config path/to/config.yaml
    python 4_Experiments/scripts/train_gaze_earlyfusion.py --resume path/to/checkpoint.pt
"""

import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import using importlib for folders starting with numbers
import importlib.util

def import_module_from_path(module_name: str, file_path: str):
    """Import a module from a file path and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so pickle can find it in worker processes
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import dataset module
_dataset_module = import_module_from_path(
    "gaze_pair_dataset",
    str(PROJECT_ROOT / "1_Data" / "datasets" / "gaze_pair_dataset.py")
)
create_train_val_datasets = _dataset_module.create_train_val_datasets
custom_collate_fn = _dataset_module.custom_collate_fn

# Import model module
_model_module = import_module_from_path(
    "early_fusion_vit",
    str(PROJECT_ROOT / "3_Models" / "backbones" / "early_fusion_vit.py")
)
EarlyFusionViT = _model_module.EarlyFusionViT

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Logging disabled.")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_linear_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    """
    Create a scheduler with linear warmup followed by cosine annealing.

    Args:
        optimizer: The optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch

    Returns:
        LambdaLR scheduler
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, class_names: list) -> Dict:
    """
    Compute classification metrics.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and confusion matrix
    """
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    conf_matrix = confusion_matrix(labels, preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_metric: float,
    config: Dict,
    save_path: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_metric': best_metric,
        'config': config
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = Path(save_path).parent / 'best_model.pt'
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0.0)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Dict,
    use_amp: bool = True
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')

    for batch_idx, (img_a, img_b, labels) in enumerate(pbar):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if use_amp:
            with autocast('cuda'):
                logits = model(img_a, img_b)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            # Gradient clipping
            if config['training'].get('max_grad_norm'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['max_grad_norm']
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img_a, img_b)
            loss = criterion(logits, labels)
            loss.backward()

            if config['training'].get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['max_grad_norm']
                )

            optimizer.step()

        # Update scheduler (step-wise)
        if scheduler:
            scheduler.step()

        # Collect predictions
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    # Compute epoch metrics
    metrics = compute_metrics(
        np.array(all_preds),
        np.array(all_labels),
        config['data']['class_names']
    )
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict,
    use_amp: bool = True
) -> Dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Validation')

    for img_a, img_b, labels in pbar:
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        labels = labels.to(device)

        if use_amp:
            with autocast('cuda'):
                logits = model(img_a, img_b)
                loss = criterion(logits, labels)
        else:
            logits = model(img_a, img_b)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

    # Compute metrics
    metrics = compute_metrics(
        np.array(all_preds),
        np.array(all_labels),
        config['data']['class_names']
    )
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def train(config: Dict, resume_path: Optional[str] = None):
    """Main training function."""

    # Set seed
    set_seed(config['system']['seed'])

    # Device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")

    # Create output directory
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Data
    # =========================================================================
    print("\n[Train] Loading datasets...")
    train_dataset, val_dataset = create_train_val_datasets(
        metadata_path=str(PROJECT_ROOT / config['data']['metadata_path']),
        image_base_path=config['data']['image_base_path'],
        val_pairs=config['data']['val_pairs'],
        image_size=config['data']['image_size'],
        image_extension=config['data'].get('image_extension', '.jpg'),
        label2id=config['data']['label2id'],
        augmentation_config=config.get('augmentation', {}).get('train', {})
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    # =========================================================================
    # Model
    # =========================================================================
    print("\n[Train] Creating model...")
    fusion_mode = config['model'].get('fusion_mode', 'concat')
    model = EarlyFusionViT(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        fusion_mode=fusion_mode,
        weight_init_strategy=config['model'].get('weight_init_strategy', 'duplicate')
    )
    model = model.to(device)
    print(f"[Train] Fusion mode: {fusion_mode}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Total parameters: {total_params:,}")
    print(f"[Train] Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # Loss (with class weights)
    # =========================================================================
    if config['training'].get('use_weighted_loss', False):
        class_weights = train_dataset.get_class_weights().to(device)
        print(f"[Train] Using weighted loss: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # =========================================================================
    # Optimizer
    # =========================================================================
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # =========================================================================
    # Scheduler (Warmup + Cosine)
    # =========================================================================
    steps_per_epoch = len(train_loader)
    scheduler = get_linear_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config['training']['warmup_epochs'],
        total_epochs=config['training']['epochs'],
        steps_per_epoch=steps_per_epoch
    )

    # =========================================================================
    # Mixed Precision
    # =========================================================================
    use_amp = config['training'].get('fp16', False) and torch.cuda.is_available()
    scaler = GradScaler('cuda') if use_amp else None
    print(f"[Train] Mixed precision (FP16): {use_amp}")

    # =========================================================================
    # Resume from checkpoint
    # =========================================================================
    start_epoch = 0
    best_metric = 0.0

    if resume_path or config['resume'].get('enabled', False):
        ckpt_path = resume_path or config['resume']['checkpoint_path']
        if ckpt_path and Path(ckpt_path).exists():
            print(f"\n[Train] Resuming from checkpoint: {ckpt_path}")
            start_epoch, best_metric = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler
            )
            start_epoch += 1  # Start from next epoch
            print(f"[Train] Resuming from epoch {start_epoch}, best_metric: {best_metric:.4f}")

    # =========================================================================
    # Wandb
    # =========================================================================
    if WANDB_AVAILABLE and config['wandb'].get('enabled', False):
        run_name = f"{config['wandb']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config['wandb']['project'],
            name=run_name,
            config=config,
            tags=config['wandb'].get('tags', []),
            notes=config['wandb'].get('notes', ''),
            resume='allow' if resume_path else None
        )
        wandb.watch(model, log='all', log_freq=100)
        print(f"[Train] Wandb initialized: {config['wandb']['project']}/{run_name}")

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    metric_for_best = config['checkpoint'].get('metric_for_best', 'val_f1')
    greater_is_better = config['checkpoint'].get('greater_is_better', True)

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, config, use_amp
        )

        print(f"\n[Train] Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config, use_amp)

        print(f"[Val]   Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")

        # Print confusion matrix
        print(f"\n[Val] Confusion Matrix:")
        print(val_metrics['confusion_matrix'])

        # Check if best model
        current_metric = val_metrics[metric_for_best.replace('val_', '')]
        if greater_is_better:
            is_best = current_metric > best_metric
        else:
            is_best = current_metric < best_metric

        if is_best:
            best_metric = current_metric
            print(f"[Train] New best {metric_for_best}: {best_metric:.4f}")

        # Save checkpoint
        if config['checkpoint'].get('save_best', True) and is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_metric, config,
                save_dir / 'best_model.pt', is_best=True
            )
            print(f"[Train] Saved best model checkpoint")

        if (epoch + 1) % config['checkpoint'].get('save_every_epochs', 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_metric, config,
                save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            )
            print(f"[Train] Saved checkpoint at epoch {epoch + 1}")

        # Wandb logging
        if WANDB_AVAILABLE and config['wandb'].get('enabled', False):
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/f1': train_metrics['f1'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict)

            # Log confusion matrix as table
            if (epoch + 1) % 5 == 0:  # Every 5 epochs
                class_names = config['data']['class_names']
                wandb.log({
                    'val/confusion_matrix': wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=list(range(len(class_names))),
                        preds=list(range(len(class_names))),
                        class_names=class_names
                    )
                })

    # =========================================================================
    # Final
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best {metric_for_best}: {best_metric:.4f}")
    print("=" * 60)

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        config['training']['epochs'] - 1, best_metric, config,
        save_dir / 'final_model.pt'
    )

    if WANDB_AVAILABLE and config['wandb'].get('enabled', False):
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Early Fusion ViT')
    parser.add_argument(
        '--config',
        type=str,
        default='4_Experiments/configs/gaze_earlyfusion.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    print(f"[Train] Loading config from: {config_path}")
    config = load_config(config_path)

    # Train
    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()
