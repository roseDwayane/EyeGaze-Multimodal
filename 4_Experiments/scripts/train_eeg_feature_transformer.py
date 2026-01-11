"""
Training Script for Multi-Stream Feature Transformer - Hyperscanning EEG Classification

This script trains the Multi-Stream Feature Transformer (MSFT) model on pre-extracted
EEG features for social interaction classification (Single/Competition/Cooperation).

The model processes features from extract_eeg_features.py:
    - bands_energy: (2, 32, 5) - Band power per player
    - intra_con: (2, 7, 5, 32, 32) - Intra-brain connectivity per player
    - inter_con: (7, 5, 32, 32) - Inter-brain connectivity

Features:
    - Support for ablation studies (use_spectral, use_intra, use_inter, etc.)
    - Weights & Biases logging
    - Checkpoint saving and resume training
    - Weighted loss for class imbalance
    - Cosine LR scheduler with warmup
    - Mixed precision training (FP16)
    - Train/Val split by participant pairs

Usage:
    python 4_Experiments/scripts/train_eeg_feature_transformer.py
    python 4_Experiments/scripts/train_eeg_feature_transformer.py --config path/to/config.yaml
    python 4_Experiments/scripts/train_eeg_feature_transformer.py --ablation full
    python 4_Experiments/scripts/train_eeg_feature_transformer.py --resume path/to/checkpoint.pt
"""

import sys
import argparse
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import Multi-Stream Feature Transformer model
_model_module = import_module_from_path(
    "multi_stream_feature_transformer",
    str(PROJECT_ROOT / "3_Models" / "backbones" / "multi_stream_feature_transformer.py")
)
MultiStreamFeatureTransformer = _model_module.MultiStreamFeatureTransformer
create_msft_model = _model_module.create_msft_model
ABLATION_CONFIGS = _model_module.ABLATION_CONFIGS

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Logging disabled.")


# =============================================================================
# EEG Feature Dataset
# =============================================================================

CLASS_MAPPING = {
    'Single': 0,
    'Competition': 1,
    'Comp': 1,
    'Cooperation': 2,
    'Coop': 2,
}

ID_TO_CLASS = {0: 'Single', 1: 'Competition', 2: 'Cooperation'}


class EEGFeatureDataset(Dataset):
    """
    Dataset for pre-extracted EEG features (.npy files).

    Loads features from individual trial .npy files created by extract_eeg_features.py.

    Args:
        feature_dir: Directory containing .npy feature files
        split: 'train' or 'val'
        val_pairs: List of pair IDs for validation set
        load_to_memory: If True, load all features into RAM (recommended)
        augmentation_config: Data augmentation settings

    Each .npy file contains:
        {
            'bands_energy': (2, 32, 5),
            'intra_con': (2, 7, 5, 32, 32),
            'inter_con': (7, 5, 32, 32),
            'metadata': {'class_idx': int, 'pair': int, ...}
        }

    Returns:
        Dict with 'bands_energy', 'intra_con', 'inter_con', 'label'
    """

    def __init__(
        self,
        feature_dir: str,
        split: str = 'train',
        val_pairs: Optional[List[int]] = None,
        load_to_memory: bool = True,
        augmentation_config: Optional[Dict] = None
    ):
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.val_pairs = val_pairs or [33, 34, 35, 36, 37, 38, 39, 40]
        self.load_to_memory = load_to_memory
        self.augmentation_config = augmentation_config or {}

        # Get all .npy files
        all_files = sorted(self.feature_dir.glob("*.npy"))

        if len(all_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {feature_dir}")

        # Filter by split (based on pair ID)
        self.feature_files = []
        for f in all_files:
            # Skip summary.json.npy if exists
            if 'summary' in f.name:
                continue

            # Load metadata to check pair
            try:
                data = np.load(f, allow_pickle=True).item()
                pair_id = data.get('metadata', {}).get('pair', -1)

                if split == 'val' and pair_id in self.val_pairs:
                    self.feature_files.append(f)
                elif split == 'train' and pair_id not in self.val_pairs:
                    self.feature_files.append(f)

            except Exception as e:
                print(f"[Dataset] Warning: Could not load {f.name}: {e}")
                continue

        print(f"[Dataset] Found {len(self.feature_files)} {split} samples")

        # Load all data into memory if requested
        if load_to_memory:
            print(f"[Dataset] Loading {split} data into memory...")
            self.data = []
            for f in tqdm(self.feature_files, desc=f"Loading {split}"):
                try:
                    item = np.load(f, allow_pickle=True).item()
                    self.data.append(item)
                except Exception as e:
                    print(f"[Dataset] Error loading {f}: {e}")

            print(f"[Dataset] Loaded {len(self.data)} samples into memory")

    def _apply_augmentation(self, features: Dict) -> Dict:
        """Apply data augmentation to features."""
        if not self.augmentation_config:
            return features

        # Make copies to avoid modifying original
        features = {k: v.copy() if isinstance(v, np.ndarray) else v
                   for k, v in features.items()}

        # Connectivity noise
        conn_noise_cfg = self.augmentation_config.get('connectivity_noise', {})
        if conn_noise_cfg.get('enabled', False):
            std = conn_noise_cfg.get('std', 0.05)

            if 'intra_con' in features:
                noise = np.random.randn(*features['intra_con'].shape).astype(np.float32) * std
                features['intra_con'] = features['intra_con'] + noise

            if 'inter_con' in features:
                noise = np.random.randn(*features['inter_con'].shape).astype(np.float32) * std
                features['inter_con'] = features['inter_con'] + noise

        # Band energy noise
        band_noise_cfg = self.augmentation_config.get('band_noise', {})
        if band_noise_cfg.get('enabled', False):
            std = band_noise_cfg.get('std', 0.1)
            if 'bands_energy' in features:
                noise = np.random.randn(*features['bands_energy'].shape).astype(np.float32) * std
                features['bands_energy'] = features['bands_energy'] + noise

        # Channel dropout (set random channels to zero)
        ch_drop_cfg = self.augmentation_config.get('channel_dropout', {})
        if ch_drop_cfg.get('enabled', False):
            drop_prob = ch_drop_cfg.get('dropout_prob', 0.1)

            if 'bands_energy' in features:
                # bands_energy: (2, 32, 5)
                mask = np.random.random(32) > drop_prob
                features['bands_energy'] = features['bands_energy'] * mask.reshape(1, 32, 1)

        return features

    def __len__(self) -> int:
        if self.load_to_memory:
            return len(self.data)
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        if self.load_to_memory:
            item = self.data[idx]
        else:
            item = np.load(self.feature_files[idx], allow_pickle=True).item()

        # Extract features
        features = {
            'bands_energy': item['bands_energy'].astype(np.float32),
            'intra_con': item['intra_con'].astype(np.float32),
            'inter_con': item['inter_con'].astype(np.float32),
        }

        # Apply augmentation (training only)
        if self.split == 'train' and self.augmentation_config:
            features = self._apply_augmentation(features)

        # Get label
        metadata = item.get('metadata', {})
        label = metadata.get('class_idx', -1)

        # Handle string class labels
        if label == -1:
            class_name = metadata.get('class', 'Single')
            label = CLASS_MAPPING.get(class_name, 0)

        # Convert to tensors
        return {
            'bands_energy': torch.from_numpy(features['bands_energy']),
            'intra_con': torch.from_numpy(features['intra_con']),
            'inter_con': torch.from_numpy(features['inter_con']),
            'label': label
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        labels = []
        for idx in range(len(self)):
            if self.load_to_memory:
                item = self.data[idx]
            else:
                item = np.load(self.feature_files[idx], allow_pickle=True).item()

            metadata = item.get('metadata', {})
            label = metadata.get('class_idx', -1)
            if label == -1:
                class_name = metadata.get('class', 'Single')
                label = CLASS_MAPPING.get(class_name, 0)
            labels.append(label)

        class_counts = Counter(labels)
        num_classes = len(class_counts)
        total = len(labels)

        weights = torch.zeros(num_classes)
        for class_id, count in class_counts.items():
            weights[class_id] = total / (num_classes * count)

        return weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        labels = []
        for idx in range(len(self)):
            if self.load_to_memory:
                item = self.data[idx]
            else:
                item = np.load(self.feature_files[idx], allow_pickle=True).item()

            metadata = item.get('metadata', {})
            label = metadata.get('class_idx', -1)
            if label == -1:
                class_name = metadata.get('class', 'Single')
                label = CLASS_MAPPING.get(class_name, 0)
            labels.append(label)

        counts = Counter(labels)
        return {ID_TO_CLASS.get(k, str(k)): v for k, v in counts.items()}


def feature_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for feature-based DataLoader."""
    return {
        'bands_energy': torch.stack([item['bands_energy'] for item in batch]),
        'intra_con': torch.stack([item['intra_con'] for item in batch]),
        'inter_con': torch.stack([item['inter_con'] for item in batch]),
        'labels': torch.tensor([item['label'] for item in batch], dtype=torch.long)
    }


def create_feature_datasets(
    feature_dir: str,
    val_pairs: Optional[List[int]] = None,
    augmentation_config: Optional[Dict] = None,
    load_to_memory: bool = True
) -> Tuple[EEGFeatureDataset, EEGFeatureDataset]:
    """
    Create train and validation datasets from pre-extracted features.

    Args:
        feature_dir: Directory containing .npy feature files
        val_pairs: List of pair IDs for validation
        augmentation_config: Augmentation settings for training
        load_to_memory: If True, load all data into RAM

    Returns:
        train_dataset, val_dataset
    """
    train_dataset = EEGFeatureDataset(
        feature_dir=feature_dir,
        split='train',
        val_pairs=val_pairs,
        load_to_memory=load_to_memory,
        augmentation_config=augmentation_config
    )

    val_dataset = EEGFeatureDataset(
        feature_dir=feature_dir,
        split='val',
        val_pairs=val_pairs,
        load_to_memory=load_to_memory,
        augmentation_config=None  # No augmentation for validation
    )

    print(f"\n[Dataset] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"[Dataset] Train distribution: {train_dataset.get_class_distribution()}")
    print(f"[Dataset] Val distribution: {val_dataset.get_class_distribution()}")

    return train_dataset, val_dataset


# =============================================================================
# Training Utilities
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_linear_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int
):
    """Create scheduler with linear warmup followed by cosine annealing."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, class_names: List[str]) -> Dict:
    """Compute classification metrics."""
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
    scaler: Optional[GradScaler],
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


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None
):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0.0)


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
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

    for batch_idx, batch in enumerate(pbar):
        # Unpack batch (dict format)
        features = {
            'bands_energy': batch['bands_energy'].to(device),
            'intra_con': batch['intra_con'].to(device),
            'inter_con': batch['inter_con'].to(device),
        }
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_amp and scaler is not None:
            with autocast('cuda'):
                logits = model(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if config['training'].get('max_grad_norm'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['max_grad_norm']
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
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

    for batch in pbar:
        features = {
            'bands_energy': batch['bands_energy'].to(device),
            'intra_con': batch['intra_con'].to(device),
            'inter_con': batch['inter_con'].to(device),
        }
        labels = batch['labels'].to(device)

        if use_amp:
            with autocast('cuda'):
                logits = model(features)
                loss = criterion(logits, labels)
        else:
            logits = model(features)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

    metrics = compute_metrics(
        np.array(all_preds),
        np.array(all_labels),
        config['data']['class_names']
    )
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


# =============================================================================
# Main Training Function
# =============================================================================

def train(config: Dict, resume_path: Optional[str] = None, ablation_mode: Optional[str] = None):
    """Main training function."""

    # Set seed
    set_seed(config['system']['seed'])

    # Device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")

    # Get ablation mode
    ablation_mode = ablation_mode or config['model'].get('ablation_mode', 'full')
    print(f"[Train] Ablation mode: {ablation_mode}")

    # Create output directory
    save_dir = Path(config['checkpoint']['save_dir']) / ablation_mode
    save_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Data
    # =========================================================================
    print("\n[Train] Loading datasets...")

    feature_dir = PROJECT_ROOT / config['data']['feature_dir']

    train_dataset, val_dataset = create_feature_datasets(
        feature_dir=str(feature_dir),
        val_pairs=config['data'].get('val_pairs', [33, 34, 35, 36, 37, 38, 39, 40]),
        augmentation_config=config.get('augmentation', {}).get('train', {}),
        load_to_memory=config['data'].get('load_to_memory', True)
    )

    if len(train_dataset) == 0:
        print("[Error] No training data found. Check data paths.")
        return

    # DataLoader settings
    num_workers = config['training'].get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=feature_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=feature_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    # =========================================================================
    # Model
    # =========================================================================
    print("\n[Train] Creating model...")

    # Get ablation configuration
    if ablation_mode in ABLATION_CONFIGS:
        ablation_flags = ABLATION_CONFIGS[ablation_mode]
    else:
        ablation_flags = {
            'use_spectral': config['model'].get('use_spectral', True),
            'use_intra': config['model'].get('use_intra', True),
            'use_inter': config['model'].get('use_inter', True),
            'use_transformer_fusion': config['model'].get('use_transformer_fusion', True),
            'use_uncertainty': config['model'].get('use_uncertainty', True)
        }

    model = MultiStreamFeatureTransformer(
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        num_channels=config['model'].get('num_channels', 32),
        num_bands=config['model'].get('num_bands', 5),
        num_metrics=config['model'].get('num_metrics', 7),
        **ablation_flags
    )
    model = model.to(device)

    # Print model info
    print(f"[Train] Model configuration:")
    print(f"  - use_spectral: {ablation_flags.get('use_spectral', True)}")
    print(f"  - use_intra: {ablation_flags.get('use_intra', True)}")
    print(f"  - use_inter: {ablation_flags.get('use_inter', True)}")
    print(f"  - use_transformer_fusion: {ablation_flags.get('use_transformer_fusion', True)}")
    print(f"  - use_uncertainty: {ablation_flags.get('use_uncertainty', True)}")
    print(f"  - num_tokens: {model.num_tokens}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Total parameters: {total_params:,}")
    print(f"[Train] Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # Loss
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
    # Scheduler
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
    # Resume
    # =========================================================================
    start_epoch = 0
    best_metric = 0.0

    if resume_path or config.get('resume', {}).get('enabled', False):
        ckpt_path = resume_path or config['resume']['checkpoint_path']
        if ckpt_path and Path(ckpt_path).exists():
            print(f"\n[Train] Resuming from checkpoint: {ckpt_path}")
            start_epoch, best_metric = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler
            )
            start_epoch += 1
            print(f"[Train] Resuming from epoch {start_epoch}, best_metric: {best_metric:.4f}")

    # =========================================================================
    # Wandb
    # =========================================================================
    if WANDB_AVAILABLE and config.get('wandb', {}).get('enabled', False):
        run_name = f"{config['wandb']['run_name']}_{ablation_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config['wandb']['project'],
            name=run_name,
            config=config,
            tags=config['wandb'].get('tags', []) + [ablation_mode],
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

    # Early stopping
    early_stop_cfg = config['training'].get('early_stopping', {})
    patience = early_stop_cfg.get('patience', 15) if early_stop_cfg.get('enabled', False) else float('inf')
    min_delta = early_stop_cfg.get('min_delta', 0.001)
    epochs_without_improvement = 0

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

        print(f"\n[Val] Confusion Matrix:")
        print(val_metrics['confusion_matrix'])

        # Check if best model
        current_metric = val_metrics[metric_for_best.replace('val_', '')]
        if greater_is_better:
            is_best = current_metric > best_metric + min_delta
        else:
            is_best = current_metric < best_metric - min_delta

        if is_best:
            best_metric = current_metric
            epochs_without_improvement = 0
            print(f"[Train] New best {metric_for_best}: {best_metric:.4f}")
        else:
            epochs_without_improvement += 1

        # Save checkpoint
        if config['checkpoint'].get('save_best', True) and is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_metric, config,
                str(save_dir / 'best_model.pt'), is_best=True
            )
            print(f"[Train] Saved best model checkpoint")

        if (epoch + 1) % config['checkpoint'].get('save_every_epochs', 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_metric, config,
                str(save_dir / f'checkpoint_epoch_{epoch+1}.pt')
            )
            print(f"[Train] Saved checkpoint at epoch {epoch + 1}")

        # Wandb logging
        if WANDB_AVAILABLE and config.get('wandb', {}).get('enabled', False):
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

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n[Train] Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {patience} epochs)")
            break

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
        str(save_dir / 'final_model.pt')
    )

    if WANDB_AVAILABLE and config.get('wandb', {}).get('enabled', False):
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Stream Feature Transformer')
    parser.add_argument(
        '--config',
        type=str,
        default='4_Experiments/configs/eeg_feature_transformer.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--ablation',
        type=str,
        default=None,
        choices=list(ABLATION_CONFIGS.keys()),
        help='Ablation mode (overrides config)'
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
    train(config, resume_path=args.resume, ablation_mode=args.ablation)


if __name__ == '__main__':
    main()
