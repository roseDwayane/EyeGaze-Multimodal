"""
Training Script for HyperEEG Encoder - Hyperscanning EEG Classification

This script trains the HyperEEG model on dual EEG data for social interaction
classification (Single/Competition/Cooperation).

Features:
    - Support for ablation studies (use_sinc, use_graph, use_cross_attn, use_uncertainty)
    - Weights & Biases logging
    - Checkpoint saving and resume training
    - Weighted loss for class imbalance
    - Cosine LR scheduler with warmup
    - Mixed precision training (FP16)
    - EEG-specific data augmentation
    - Early stopping

Usage:
    python 4_Experiments/scripts/train_eeg_hypereeg.py
    python 4_Experiments/scripts/train_eeg_hypereeg.py --config path/to/config.yaml
    python 4_Experiments/scripts/train_eeg_hypereeg.py --ablation full
    python 4_Experiments/scripts/train_eeg_hypereeg.py --resume path/to/checkpoint.pt
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


# Import HyperEEG model
_model_module = import_module_from_path(
    "hypereeg",
    str(PROJECT_ROOT / "3_Models" / "backbones" / "hypereeg.py")
)
HyperEEG_Encoder = _model_module.HyperEEG_Encoder
create_hypereeg_model = _model_module.create_hypereeg_model
ABLATION_CONFIGS = _model_module.ABLATION_CONFIGS

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Logging disabled.")

# Optional: scipy for bandpass filter
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Warning] scipy not installed. Bandpass filtering disabled.")


# =============================================================================
# EEG Dataset
# =============================================================================

class HyperEEGDataset(Dataset):
    """
    Dataset for dual-EEG hyperscanning classification.

    Loads paired EEG signals from two participants and applies preprocessing.

    Args:
        metadata: List of sample metadata dicts
        eeg_base_path: Base directory containing EEG files
        label2id: Mapping from class name to label
        window_size: Timepoints per window
        stride: Sliding window stride
        sampling_rate: EEG sampling rate (Hz)
        num_channels: Number of EEG channels to use
        enable_preprocessing: Whether to apply CAR and normalization
        augmentation_config: Data augmentation settings
        return_metadata: Whether to return metadata dict

    Returns:
        eeg1: (C, T) tensor for Person 1
        eeg2: (C, T) tensor for Person 2
        label: Class label
    """

    def __init__(
        self,
        metadata: List[Dict],
        eeg_base_path: str,
        label2id: Dict[str, int],
        window_size: int = 1024,
        stride: int = 512,
        sampling_rate: int = 500,
        num_channels: int = 32,
        enable_preprocessing: bool = True,
        filter_low: float = 1.0,
        filter_high: float = 45.0,
        augmentation_config: Optional[Dict] = None,
        return_metadata: bool = False
    ):
        self.metadata = metadata
        self.eeg_base_path = Path(eeg_base_path)
        self.label2id = label2id
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.enable_preprocessing = enable_preprocessing
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.augmentation_config = augmentation_config or {}
        self.return_metadata = return_metadata

        # Build valid windows
        self.valid_windows = []
        self._prepare_windows()

        print(f"[Dataset] Created {len(self.valid_windows)} windows from {len(metadata)} samples")

    def _prepare_windows(self):
        """Precompute valid sliding windows from all samples."""
        for idx, item in enumerate(self.metadata):
            # Construct EEG file paths
            player1_path = self.eeg_base_path / f"{item['player1']}.csv"
            player2_path = self.eeg_base_path / f"{item['player2']}.csv"

            # Also try .npy format
            if not player1_path.exists():
                player1_path = self.eeg_base_path / f"{item['player1']}.npy"
            if not player2_path.exists():
                player2_path = self.eeg_base_path / f"{item['player2']}.npy"

            if not player1_path.exists() or not player2_path.exists():
                continue

            try:
                # Get data length (without loading full data if possible)
                if player1_path.suffix == '.npy':
                    # For npy, we can get shape from header
                    eeg1 = np.load(player1_path, mmap_mode='r')
                    eeg2 = np.load(player2_path, mmap_mode='r')
                    len1 = eeg1.shape[-1] if eeg1.ndim > 1 else eeg1.shape[0]
                    len2 = eeg2.shape[-1] if eeg2.ndim > 1 else eeg2.shape[0]
                else:
                    # For CSV, read header only
                    import pandas as pd
                    df1 = pd.read_csv(player1_path, header=None, nrows=1)
                    df2 = pd.read_csv(player2_path, header=None, nrows=1)
                    len1 = df1.shape[1]
                    len2 = df2.shape[1]

                min_len = min(len1, len2)

                if min_len < self.window_size:
                    continue

                # Create sliding windows
                num_windows = (min_len - self.window_size) // self.stride + 1

                for win_idx in range(num_windows):
                    start = win_idx * self.stride
                    end = start + self.window_size

                    if end <= min_len:
                        self.valid_windows.append({
                            'metadata_idx': idx,
                            'start': start,
                            'end': end,
                            'player1_path': str(player1_path),
                            'player2_path': str(player2_path),
                            'pair': item.get('pair', -1),
                            'class': item['class']
                        })

            except Exception as e:
                if idx < 10:
                    print(f"[Dataset] Warning: Error processing {idx}: {e}")
                continue

    def _load_eeg(self, path: str) -> np.ndarray:
        """Load EEG from file (supports .csv and .npy)."""
        path = Path(path)

        if path.suffix == '.npy':
            eeg = np.load(path)
        else:
            import pandas as pd
            df = pd.read_csv(path, header=None)
            eeg = df.values

        # Ensure shape is (C, T)
        if eeg.ndim == 1:
            eeg = eeg.reshape(1, -1)
        elif eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T

        return eeg.astype(np.float32)

    def _preprocess_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG signal:
        1. Bandpass filter (if scipy available)
        2. Common Average Reference (CAR)
        3. Z-score normalization
        """
        # 1. Bandpass filter
        if SCIPY_AVAILABLE and self.enable_preprocessing:
            try:
                nyquist = self.sampling_rate / 2
                low = self.filter_low / nyquist
                high = min(self.filter_high / nyquist, 0.99)
                b, a = signal.butter(4, [low, high], btype='band')
                eeg = signal.filtfilt(b, a, eeg, axis=1)
            except Exception:
                pass  # Skip filter on error

        # 2. Common Average Reference (CAR)
        car = eeg.mean(axis=0, keepdims=True)
        eeg = eeg - car

        # 3. Z-score normalization per channel
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True) + 1e-8
        eeg = (eeg - mean) / std

        return eeg.astype(np.float32)

    def _apply_augmentation(self, eeg: np.ndarray) -> np.ndarray:
        """Apply EEG-specific data augmentation."""
        if not self.augmentation_config:
            return eeg

        # Time masking
        time_mask_cfg = self.augmentation_config.get('time_mask', {})
        if time_mask_cfg.get('enabled', False):
            max_len = time_mask_cfg.get('max_mask_length', 100)
            num_masks = time_mask_cfg.get('num_masks', 2)
            for _ in range(num_masks):
                if random.random() < 0.5:
                    mask_len = random.randint(1, max_len)
                    start = random.randint(0, max(0, eeg.shape[1] - mask_len))
                    eeg[:, start:start + mask_len] = 0

        # Channel dropout
        ch_drop_cfg = self.augmentation_config.get('channel_dropout', {})
        if ch_drop_cfg.get('enabled', False):
            drop_prob = ch_drop_cfg.get('dropout_prob', 0.1)
            mask = np.random.random(eeg.shape[0]) > drop_prob
            if mask.sum() > 0:  # Ensure at least one channel remains
                eeg = eeg * mask.reshape(-1, 1)

        # Gaussian noise
        noise_cfg = self.augmentation_config.get('gaussian_noise', {})
        if noise_cfg.get('enabled', False):
            std = noise_cfg.get('std', 0.1)
            noise = np.random.randn(*eeg.shape) * std
            eeg = eeg + noise

        return eeg.astype(np.float32)

    def __len__(self) -> int:
        return len(self.valid_windows)

    def __getitem__(self, idx: int):
        """Get a windowed sample."""
        window = self.valid_windows[idx]

        try:
            # Load EEG
            eeg1 = self._load_eeg(window['player1_path'])
            eeg2 = self._load_eeg(window['player2_path'])

            # Extract window
            start, end = window['start'], window['end']
            eeg1 = eeg1[:, start:end]
            eeg2 = eeg2[:, start:end]

            # Ensure correct number of channels
            if eeg1.shape[0] > self.num_channels:
                eeg1 = eeg1[:self.num_channels, :]
            if eeg2.shape[0] > self.num_channels:
                eeg2 = eeg2[:self.num_channels, :]

            # Pad if fewer channels
            if eeg1.shape[0] < self.num_channels:
                pad = np.zeros((self.num_channels - eeg1.shape[0], eeg1.shape[1]))
                eeg1 = np.vstack([eeg1, pad])
            if eeg2.shape[0] < self.num_channels:
                pad = np.zeros((self.num_channels - eeg2.shape[0], eeg2.shape[1]))
                eeg2 = np.vstack([eeg2, pad])

            # Preprocess
            if self.enable_preprocessing:
                eeg1 = self._preprocess_eeg(eeg1)
                eeg2 = self._preprocess_eeg(eeg2)

            # Augmentation (training only)
            if self.augmentation_config:
                eeg1 = self._apply_augmentation(eeg1)
                eeg2 = self._apply_augmentation(eeg2)

            # Convert to tensors
            eeg1_tensor = torch.from_numpy(eeg1).float()
            eeg2_tensor = torch.from_numpy(eeg2).float()
            label = self.label2id[window['class']]

            if self.return_metadata:
                return eeg1_tensor, eeg2_tensor, label, window

            return eeg1_tensor, eeg2_tensor, label

        except Exception as e:
            print(f"[Dataset] Error loading window {idx}: {e}")
            # Return dummy data
            dummy = torch.zeros(self.num_channels, self.window_size)
            if self.return_metadata:
                return dummy, dummy, 0, window
            return dummy, dummy, 0

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        class_counts = Counter(w['class'] for w in self.valid_windows)
        num_classes = len(self.label2id)
        total = len(self.valid_windows)

        weights = torch.zeros(num_classes)
        for class_name, class_id in self.label2id.items():
            count = class_counts.get(class_name, 1)
            weights[class_id] = total / (num_classes * count)

        return weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        return dict(Counter(w['class'] for w in self.valid_windows))


# =============================================================================
# Preprocessed EEG Dataset (Optimized for Speed)
# =============================================================================

class PreprocessedEEGDataset(Dataset):
    """
    Dataset for preprocessed EEG data stored as .npy files.

    This dataset loads pre-computed windows from disk, optionally loading
    everything into memory for maximum speed.

    Args:
        data_dir: Directory containing eeg1.npy, eeg2.npy, labels.npy
        load_to_memory: If True, load all data into RAM (recommended)
        augmentation_config: Data augmentation settings (applied on-the-fly)

    Expected files in data_dir:
        - eeg1.npy: (N, C, T) Player 1 EEG windows
        - eeg2.npy: (N, C, T) Player 2 EEG windows
        - labels.npy: (N,) Class labels
    """

    def __init__(
        self,
        data_dir: str,
        load_to_memory: bool = True,
        augmentation_config: Optional[Dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.load_to_memory = load_to_memory
        self.augmentation_config = augmentation_config or {}

        # Check files exist
        self.eeg1_path = self.data_dir / 'eeg1.npy'
        self.eeg2_path = self.data_dir / 'eeg2.npy'
        self.labels_path = self.data_dir / 'labels.npy'

        if not all(p.exists() for p in [self.eeg1_path, self.eeg2_path, self.labels_path]):
            raise FileNotFoundError(f"Missing preprocessed files in {data_dir}")

        if load_to_memory:
            # Load everything into memory
            print(f"[Dataset] Loading preprocessed data into memory from {data_dir}...")
            self.eeg1 = np.load(self.eeg1_path)
            self.eeg2 = np.load(self.eeg2_path)
            self.labels = np.load(self.labels_path)
            print(f"[Dataset] Loaded: eeg1={self.eeg1.shape}, eeg2={self.eeg2.shape}, labels={self.labels.shape}")
        else:
            # Memory-mapped for large datasets
            self.eeg1 = np.load(self.eeg1_path, mmap_mode='r')
            self.eeg2 = np.load(self.eeg2_path, mmap_mode='r')
            self.labels = np.load(self.labels_path)

        self.num_samples = len(self.labels)

    def _apply_augmentation(self, eeg: np.ndarray) -> np.ndarray:
        """Apply EEG-specific data augmentation."""
        if not self.augmentation_config:
            return eeg

        eeg = eeg.copy()  # Don't modify original data

        # Time masking
        time_mask_cfg = self.augmentation_config.get('time_mask', {})
        if time_mask_cfg.get('enabled', False):
            max_len = time_mask_cfg.get('max_mask_length', 100)
            num_masks = time_mask_cfg.get('num_masks', 2)
            for _ in range(num_masks):
                if random.random() < 0.5:
                    mask_len = random.randint(1, max_len)
                    start = random.randint(0, max(0, eeg.shape[1] - mask_len))
                    eeg[:, start:start + mask_len] = 0

        # Channel dropout
        ch_drop_cfg = self.augmentation_config.get('channel_dropout', {})
        if ch_drop_cfg.get('enabled', False):
            drop_prob = ch_drop_cfg.get('dropout_prob', 0.1)
            mask = np.random.random(eeg.shape[0]) > drop_prob
            if mask.sum() > 0:
                eeg = eeg * mask.reshape(-1, 1)

        # Gaussian noise
        noise_cfg = self.augmentation_config.get('gaussian_noise', {})
        if noise_cfg.get('enabled', False):
            std = noise_cfg.get('std', 0.1)
            noise = np.random.randn(*eeg.shape).astype(np.float32) * std
            eeg = eeg + noise

        return eeg

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        """Get a preprocessed sample."""
        eeg1 = self.eeg1[idx]
        eeg2 = self.eeg2[idx]
        label = self.labels[idx]

        # Apply augmentation if configured
        if self.augmentation_config:
            eeg1 = self._apply_augmentation(eeg1)
            eeg2 = self._apply_augmentation(eeg2)

        return (
            torch.from_numpy(eeg1).float(),
            torch.from_numpy(eeg2).float(),
            label
        )

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        class_counts = Counter(self.labels)
        num_classes = len(class_counts)
        total = len(self.labels)

        weights = torch.zeros(num_classes)
        for class_id, count in class_counts.items():
            weights[class_id] = total / (num_classes * count)

        return weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        id2label = {0: 'Single', 1: 'Competition', 2: 'Cooperation'}
        counts = Counter(self.labels)
        return {id2label.get(k, str(k)): v for k, v in counts.items()}


def create_preprocessed_datasets(
    preprocessed_dir: str,
    augmentation_config: Optional[Dict] = None,
    load_to_memory: bool = True
) -> Tuple[PreprocessedEEGDataset, PreprocessedEEGDataset]:
    """
    Create train and validation datasets from preprocessed .npy files.

    Args:
        preprocessed_dir: Directory containing train/ and val/ subdirectories
        augmentation_config: Augmentation settings for training
        load_to_memory: If True, load all data into RAM

    Returns:
        train_dataset, val_dataset
    """
    preprocessed_dir = Path(preprocessed_dir)

    train_dataset = PreprocessedEEGDataset(
        data_dir=preprocessed_dir / 'train',
        load_to_memory=load_to_memory,
        augmentation_config=augmentation_config
    )

    val_dataset = PreprocessedEEGDataset(
        data_dir=preprocessed_dir / 'val',
        load_to_memory=load_to_memory,
        augmentation_config=None  # No augmentation for validation
    )

    print(f"[Dataset] Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")
    print(f"[Dataset] Train distribution: {train_dataset.get_class_distribution()}")
    print(f"[Dataset] Val distribution: {val_dataset.get_class_distribution()}")

    return train_dataset, val_dataset


def create_eeg_datasets(
    metadata_path: str,
    eeg_base_path: str,
    val_pairs: List[int],
    window_size: int = 1024,
    stride: int = 512,
    sampling_rate: int = 500,
    num_channels: int = 32,
    label2id: Optional[Dict[str, int]] = None,
    augmentation_config: Optional[Dict] = None,
    enable_preprocessing: bool = True
) -> Tuple[HyperEEGDataset, HyperEEGDataset]:
    """
    Create train and validation EEG datasets.

    Args:
        metadata_path: Path to metadata JSON file
        eeg_base_path: Base directory for EEG files
        val_pairs: List of pair IDs for validation
        window_size: Timepoints per window
        stride: Sliding window stride
        sampling_rate: EEG sampling rate
        num_channels: Number of EEG channels
        label2id: Label mapping
        augmentation_config: Augmentation settings for training
        enable_preprocessing: Whether to preprocess EEG

    Returns:
        train_dataset, val_dataset
    """
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)

    # Default label mapping
    label2id = label2id or {"Single": 0, "Competition": 1, "Cooperation": 2}

    # Split by pair
    train_metadata = [m for m in all_metadata if m.get('pair', -1) not in val_pairs]
    val_metadata = [m for m in all_metadata if m.get('pair', -1) in val_pairs]

    print(f"[Dataset] Train samples: {len(train_metadata)}, Val samples: {len(val_metadata)}")

    # Create datasets
    train_dataset = HyperEEGDataset(
        metadata=train_metadata,
        eeg_base_path=eeg_base_path,
        label2id=label2id,
        window_size=window_size,
        stride=stride,
        sampling_rate=sampling_rate,
        num_channels=num_channels,
        enable_preprocessing=enable_preprocessing,
        augmentation_config=augmentation_config
    )

    val_dataset = HyperEEGDataset(
        metadata=val_metadata,
        eeg_base_path=eeg_base_path,
        label2id=label2id,
        window_size=window_size,
        stride=stride,
        sampling_rate=sampling_rate,
        num_channels=num_channels,
        enable_preprocessing=enable_preprocessing,
        augmentation_config=None  # No augmentation for validation
    )

    print(f"[Dataset] Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")
    print(f"[Dataset] Train distribution: {train_dataset.get_class_distribution()}")
    print(f"[Dataset] Val distribution: {val_dataset.get_class_distribution()}")

    return train_dataset, val_dataset


def eeg_collate_fn(batch):
    """Collate function for EEG DataLoader."""
    if len(batch[0]) == 4:  # With metadata
        eeg1 = torch.stack([item[0] for item in batch])
        eeg2 = torch.stack([item[1] for item in batch])
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        metadata = [item[3] for item in batch]
        return eeg1, eeg2, labels, metadata
    else:
        eeg1 = torch.stack([item[0] for item in batch])
        eeg2 = torch.stack([item[1] for item in batch])
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        return eeg1, eeg2, labels


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

    for batch_idx, batch in enumerate(pbar):
        # Unpack batch
        eeg1, eeg2, labels = batch[0], batch[1], batch[2]
        eeg1 = eeg1.to(device)
        eeg2 = eeg2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        if use_amp:
            with autocast('cuda'):
                logits = model(eeg1, eeg2)
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
            logits = model(eeg1, eeg2)
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
        eeg1, eeg2, labels = batch[0], batch[1], batch[2]
        eeg1 = eeg1.to(device)
        eeg2 = eeg2.to(device)
        labels = labels.to(device)

        if use_amp:
            with autocast('cuda'):
                logits = model(eeg1, eeg2)
                loss = criterion(logits, labels)
        else:
            logits = model(eeg1, eeg2)
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

    # Check if using preprocessed data
    use_preprocessed = config['data'].get('use_preprocessed', False)
    preprocessed_dir = config['data'].get('preprocessed_dir', '1_Data/datasets/EEGseg_preprocessed')

    if use_preprocessed:
        # Use fast preprocessed datasets
        print("[Train] Using preprocessed .npy data for fast loading...")
        preprocessed_path = PROJECT_ROOT / preprocessed_dir
        train_dataset, val_dataset = create_preprocessed_datasets(
            preprocessed_dir=str(preprocessed_path),
            augmentation_config=config.get('augmentation', {}).get('train', {}),
            load_to_memory=config['data'].get('load_to_memory', True)
        )
    else:
        # Use original CSV-based datasets (slower)
        print("[Train] Using raw CSV data (consider running preprocessing script)...")
        train_dataset, val_dataset = create_eeg_datasets(
            metadata_path=str(PROJECT_ROOT / config['data']['metadata_path']),
            eeg_base_path=config['data']['eeg_base_path'],
            val_pairs=config['data']['val_pairs'],
            window_size=config['data']['window_size'],
            stride=config['data']['stride'],
            sampling_rate=config['data']['sampling_rate'],
            num_channels=config['data']['num_channels'],
            label2id=config['data']['label2id'],
            augmentation_config=config.get('augmentation', {}).get('train', {}),
            enable_preprocessing=config['data'].get('enable_preprocessing', True)
        )

    if len(train_dataset) == 0:
        print("[Error] No training data found. Check data paths.")
        return

    # DataLoader settings - optimized for speed
    num_workers = config['training'].get('num_workers', 4)
    prefetch_factor = config['training'].get('prefetch_factor', 2)
    persistent_workers = config['training'].get('persistent_workers', False) and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=eeg_collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eeg_collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers
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
            'use_sinc': config['model'].get('use_sinc', True),
            'use_graph': config['model'].get('use_graph', True),
            'use_cross_attn': config['model'].get('use_cross_attn', True),
            'use_uncertainty': config['model'].get('use_uncertainty', True)
        }

    model = HyperEEG_Encoder(
        in_channels=config['model']['in_channels'],
        in_timepoints=config['model']['in_timepoints'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        sample_rate=config['model']['sample_rate'],
        sinc_kernel_size=config['model']['sinc_kernel_size'],
        **ablation_flags
    )
    model = model.to(device)

    # Print model info
    print(f"[Train] Model configuration:")
    print(f"  - use_sinc (M1): {ablation_flags['use_sinc']}")
    print(f"  - use_graph (M2): {ablation_flags['use_graph']}")
    print(f"  - use_cross_attn (M3): {ablation_flags['use_cross_attn']}")
    print(f"  - use_uncertainty (M4): {ablation_flags['use_uncertainty']}")

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

    if resume_path or config['resume'].get('enabled', False):
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
    if WANDB_AVAILABLE and config['wandb'].get('enabled', False):
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
        save_dir / 'final_model.pt'
    )

    if WANDB_AVAILABLE and config['wandb'].get('enabled', False):
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train HyperEEG Encoder')
    parser.add_argument(
        '--config',
        type=str,
        default='4_Experiments/configs/eeg_hypereeg.yaml',
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
