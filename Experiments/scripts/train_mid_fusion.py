"""
Training script for Mid Fusion Model (Four-Tower Architecture)
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from datasets import load_dataset
from tqdm import tqdm
import logging
import wandb
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Models.fusion.mid_fusion import MidFusionModel
from Data.processed.multimodal_dataset import MultimodalDataset, collate_fn

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
    """Prepare train and test datasets"""
    logger.info("Loading dataset from JSON...")

    # Load dataset
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

    # Create multimodal datasets
    image_base_path = config['data']['image_base_path']
    eeg_base_path = config['data']['eeg_base_path']
    label2id = config['data']['label2id']
    image_size = config['data']['image_size']
    eeg_window_size = config['data']['eeg_window_size']
    eeg_stride = config['data']['eeg_stride']
    enable_eeg_preprocessing = config['data'].get('enable_eeg_preprocessing', False)

    train_dataset = MultimodalDataset(
        split_datasets['train'],
        image_base_path,
        eeg_base_path,
        label2id,
        image_size=image_size,
        eeg_window_size=eeg_window_size,
        eeg_stride=eeg_stride,
        enable_eeg_preprocessing=enable_eeg_preprocessing,
        mode='train'
    )

    test_dataset = MultimodalDataset(
        split_datasets['test'],
        image_base_path,
        eeg_base_path,
        label2id,
        image_size=image_size,
        eeg_window_size=eeg_window_size,
        eeg_stride=eeg_stride,
        enable_eeg_preprocessing=enable_eeg_preprocessing,
        mode='test'
    )

    return train_dataset, test_dataset


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        eeg1 = batch['eeg1'].to(device)
        eeg2 = batch['eeg2'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(img1, img2, eeg1, eeg2, labels)

        loss = outputs['loss']

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    return {
        'train/loss': total_loss / num_batches
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
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            eeg1 = batch['eeg1'].to(device)
            eeg2 = batch['eeg2'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(img1, img2, eeg1, eeg2, labels)

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

    # Create model
    model = MidFusionModel(
        # Image encoder
        image_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        image_d_model=config['model']['image_d_model'],
        image_num_layers=config['model']['image_num_layers'],
        image_num_heads=config['model']['image_num_heads'],
        image_shared_weights=config['model']['image_shared_weights'],

        # EEG encoder
        eeg_in_channels=config['model']['eeg_in_channels'],
        eeg_d_model=config['model']['eeg_d_model'],
        eeg_num_layers=config['model']['eeg_num_layers'],
        eeg_num_heads=config['model']['eeg_num_heads'],
        eeg_d_ff=config['model']['eeg_d_ff'],
        eeg_window_size=config['model']['eeg_window_size'],
        eeg_conv_kernel=config['model']['eeg_conv_kernel'],
        eeg_conv_stride=config['model']['eeg_conv_stride'],
        eeg_conv_layers=config['model']['eeg_conv_layers'],
        eeg_shared_weights=config['model']['eeg_shared_weights'],

        # Fusion
        num_classes=config['model']['num_classes'],
        fusion_mode=config['model']['fusion_mode'],
        use_ibs_token=config['model']['use_ibs_token'],
        use_cross_attention=config['model']['use_cross_attention'],
        cross_attn_num_heads=config['model']['cross_attn_num_heads'],
        fusion_dropout=config['model']['fusion_dropout'],
        fusion_hidden_dim=config['model']['fusion_hidden_dim'],

        # Pre-trained
        image_model_path=config['model'].get('image_model_path'),
        eeg_model_path=config['model'].get('eeg_model_path'),
        freeze_image=config['model']['freeze_image'],
        freeze_eeg=config['model']['freeze_eeg']
    )

    model = model.to(device)

    logger.info(f"Model created with {model.get_num_params()} trainable parameters")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    num_epochs = config['training']['num_train_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 5)

    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

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
        train_metrics = train_epoch(model, train_loader, optimizer, device)

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

    # Final evaluation
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
    parser = argparse.ArgumentParser(description="Train Mid Fusion Model")
    parser.add_argument(
        '--config',
        type=str,
        default='Experiments/configs/mid_fusion.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()
    main(args)
