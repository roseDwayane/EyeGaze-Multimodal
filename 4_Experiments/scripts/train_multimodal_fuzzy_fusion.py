"""
Training Script for Multimodal Fuzzy Gating Fusion

This script trains a multimodal fusion model that combines:
- Gaze Encoder (EarlyFusionViT): Processes dual eye gaze heatmaps with early fusion
- EEG Encoder (DualEEGTransformer): Processes dual EEG signals
- FuzzyGatingFusion: Uncertainty-aware fusion of modality logits

The fusion module uses differentiable fuzzy logic to dynamically adjust
modality weights based on prediction entropy/uncertainty.

Features:
    - Pretrained encoder loading with optional freezing
    - Multi-task loss (CE + auxiliary losses + temperature regularization)
    - Weights & Biases logging with fusion weight tracking
    - Ablation study support (full, no_temperature, no_fuzzification, fixed_weights)
    - Mixed precision training (FP16)

Usage:
    python 4_Experiments/scripts/train_multimodal_fuzzy_fusion.py
    python 4_Experiments/scripts/train_multimodal_fuzzy_fusion.py --config path/to/config.yaml
    python 4_Experiments/scripts/train_multimodal_fuzzy_fusion.py --resume path/to/checkpoint.pt
"""

import sys
import argparse
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3_Models" / "backbones"))  # For art module imports

# Import using importlib for folders starting with numbers
import importlib.util


def import_module_from_path(module_name: str, file_path: str):
    """Import a module from a file path and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import modules
_gaze_model_module = import_module_from_path(
    "early_fusion_vit",
    str(PROJECT_ROOT / "3_Models" / "backbones" / "early_fusion_vit.py")
)
EarlyFusionViT = _gaze_model_module.EarlyFusionViT

_eeg_model_module = import_module_from_path(
    "dual_eeg_transformer",
    str(PROJECT_ROOT / "3_Models" / "backbones" / "dual_eeg_transformer.py")
)
DualEEGTransformer = _eeg_model_module.DualEEGTransformer

_fusion_module = import_module_from_path(
    "fuzzy_gating_fusion",
    str(PROJECT_ROOT / "3_Models" / "fusion" / "fuzzy_gating_fusion.py")
)
FuzzyGatingFusion = _fusion_module.FuzzyGatingFusion

_dataset_module = import_module_from_path(
    "multimodal_dataset",
    str(PROJECT_ROOT / "1_Data" / "processed" / "multimodal_dataset.py")
)
MultimodalDataset = _dataset_module.MultimodalDataset
multimodal_collate_fn = _dataset_module.collate_fn

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Logging disabled.")


class MultimodalFusionModel(nn.Module):
    """
    Multimodal Fusion Model combining Gaze and EEG encoders with Fuzzy Gating.

    Architecture:
        img1, img2 -> GazeEncoder -> img_logits
        eeg1, eeg2 -> EEGEncoder  -> eeg_logits
        img_logits, eeg_logits -> FuzzyGatingFusion -> fused_logits
    """

    def __init__(
        self,
        gaze_encoder: nn.Module,
        eeg_encoder: nn.Module,
        fusion_module: nn.Module,
        freeze_gaze: bool = False,
        freeze_eeg: bool = False,
    ):
        super().__init__()
        self.gaze_encoder = gaze_encoder
        self.eeg_encoder = eeg_encoder
        self.fusion = fusion_module

        # Optionally freeze encoders
        if freeze_gaze:
            for param in self.gaze_encoder.parameters():
                param.requires_grad = False
            logger.info("Gaze encoder frozen")

        if freeze_eeg:
            for param in self.eeg_encoder.parameters():
                param.requires_grad = False
            logger.info("EEG encoder frozen")

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            img1: (B, 3, H, W) - Player 1 gaze heatmap
            img2: (B, 3, H, W) - Player 2 gaze heatmap
            eeg1: (B, C, T) - Player 1 EEG signal
            eeg2: (B, C, T) - Player 2 EEG signal
            labels: (B,) - Ground truth labels (optional)

        Returns:
            Dict containing logits, fusion info, and losses
        """
        # Get gaze logits
        img_logits = self.gaze_encoder(img1, img2)  # (B, num_classes)

        # Get EEG logits
        eeg_outputs = self.eeg_encoder(eeg1, eeg2, labels)
        eeg_logits = eeg_outputs['logits']  # (B, num_classes)

        # Fuse logits
        fused_logits, alpha, aux_info = self.fusion(img_logits, eeg_logits)

        outputs = {
            'fused_logits': fused_logits,
            'img_logits': img_logits,
            'eeg_logits': eeg_logits,
            'alpha': alpha,
            'aux_info': aux_info,
        }

        return outputs


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


def get_linear_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int
):
    """Create a scheduler with linear warmup followed by cosine annealing."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, class_names: list) -> Dict:
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
) -> Tuple[int, float]:
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


def load_pretrained_encoder(
    encoder: nn.Module,
    checkpoint_path: str,
    encoder_type: str = 'gaze'
) -> nn.Module:
    """Load pretrained encoder weights."""
    if not Path(checkpoint_path).exists():
        logger.warning(f"Pretrained {encoder_type} checkpoint not found: {checkpoint_path}")
        return encoder

    logger.info(f"Loading pretrained {encoder_type} encoder from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load state dict (allow partial loading)
    encoder_state = encoder.state_dict()
    pretrained_state = {k: v for k, v in state_dict.items() if k in encoder_state}

    if len(pretrained_state) == 0:
        logger.warning(f"No matching keys found in {encoder_type} checkpoint")
    else:
        encoder_state.update(pretrained_state)
        encoder.load_state_dict(encoder_state)
        logger.info(f"Loaded {len(pretrained_state)}/{len(encoder_state)} parameters for {encoder_type} encoder")

    return encoder


def prepare_datasets(config: Dict) -> Tuple[MultimodalDataset, MultimodalDataset]:
    """
    Prepare train and validation datasets.

    EEG data loading follows the same pattern as train_art.py:
    - Uses window_size, stride, enable_preprocessing (same parameter names)
    - Stratified split by class label
    """
    logger.info("Loading multimodal dataset...")

    # Load metadata (same as train_art.py)
    metadata_path = config['data']['metadata_path']
    dataset = load_dataset("json", data_files=metadata_path, split="train")

    # Limit samples if specified (same as train_art.py)
    max_samples = config['data'].get('max_samples', None)
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples for quick testing")

    logger.info(f"Total samples: {len(dataset)}")

    # Train/val split (same as train_art.py)
    test_size = config['data']['train_test_split']
    seed = config['data']['random_seed']

    # Try stratified split, fall back to regular split if it fails (same as train_art.py)
    try:
        split_datasets = dataset.train_test_split(
            test_size=test_size,
            seed=seed,
            stratify_by_column='class'
        )
    except (ValueError, KeyError):
        logger.warning("Stratified split not available, using regular split")
        split_datasets = dataset.train_test_split(test_size=test_size, seed=seed)

    logger.info(f"Train samples: {len(split_datasets['train'])}")
    logger.info(f"Test samples: {len(split_datasets['test'])}")

    # EEG parameters (same names as train_art.py / dual_eeg_transformer.yaml)
    window_size = config['data']['window_size']
    stride = config['data']['stride']
    enable_preprocessing = config['data'].get('enable_preprocessing', False)

    logger.info(f"EEG windowing: window_size={window_size}, stride={stride}, preprocessing={enable_preprocessing}")

    # Create datasets
    train_dataset = MultimodalDataset(
        dataset=split_datasets['train'],
        image_base_path=config['data']['image_base_path'],
        eeg_base_path=config['data']['eeg_base_path'],
        label2id=config['data']['label2id'],
        image_size=config['data']['image_size'],
        eeg_window_size=window_size,          # Map to MultimodalDataset parameter
        eeg_stride=stride,                     # Map to MultimodalDataset parameter
        enable_eeg_preprocessing=enable_preprocessing,  # Map to MultimodalDataset parameter
        mode='train'
    )

    val_dataset = MultimodalDataset(
        dataset=split_datasets['test'],
        image_base_path=config['data']['image_base_path'],
        eeg_base_path=config['data']['eeg_base_path'],
        label2id=config['data']['label2id'],
        image_size=config['data']['image_size'],
        eeg_window_size=window_size,          # Map to MultimodalDataset parameter
        eeg_stride=stride,                     # Map to MultimodalDataset parameter
        enable_eeg_preprocessing=enable_preprocessing,  # Map to MultimodalDataset parameter
        mode='val'
    )

    return train_dataset, val_dataset


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
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
    total_loss_ce = 0.0
    total_loss_aux_img = 0.0
    total_loss_aux_eeg = 0.0
    total_loss_reg = 0.0

    all_preds = []
    all_labels = []
    all_alphas = []

    # Loss weights
    lambda_aux_img = config['training'].get('lambda_aux_img', 0.3)
    lambda_aux_eeg = config['training'].get('lambda_aux_eeg', 0.3)
    lambda_reg = config['training'].get('lambda_reg', 0.1)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')

    for batch_idx, batch in enumerate(pbar):
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        eeg1 = batch['eeg1'].to(device)
        eeg2 = batch['eeg2'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                outputs = model(img1, img2, eeg1, eeg2, labels)

                # Main CE loss on fused logits
                loss_ce = F.cross_entropy(outputs['fused_logits'], labels)

                # Auxiliary losses on individual encoder logits
                # Use calibrated logits (divided by temperature)
                T_img = outputs['aux_info']['temperatures']['img']
                T_eeg = outputs['aux_info']['temperatures']['eeg']

                loss_aux_img = F.cross_entropy(outputs['img_logits'] / T_img, labels)
                loss_aux_eeg = F.cross_entropy(outputs['eeg_logits'] / T_eeg, labels)

                # Temperature regularization
                loss_reg = model.fusion.compute_temperature_regularization(
                    t_min=config['fusion'].get('temp_reg_min', 0.5),
                    t_max=config['fusion'].get('temp_reg_max', 5.0)
                )

                # Total loss
                loss = (loss_ce +
                        lambda_aux_img * loss_aux_img +
                        lambda_aux_eeg * loss_aux_eeg +
                        lambda_reg * loss_reg)

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
            outputs = model(img1, img2, eeg1, eeg2, labels)

            loss_ce = F.cross_entropy(outputs['fused_logits'], labels)

            T_img = outputs['aux_info']['temperatures']['img']
            T_eeg = outputs['aux_info']['temperatures']['eeg']

            loss_aux_img = F.cross_entropy(outputs['img_logits'] / T_img, labels)
            loss_aux_eeg = F.cross_entropy(outputs['eeg_logits'] / T_eeg, labels)

            loss_reg = model.fusion.compute_temperature_regularization(
                t_min=config['fusion'].get('temp_reg_min', 0.5),
                t_max=config['fusion'].get('temp_reg_max', 5.0)
            )

            loss = (loss_ce +
                    lambda_aux_img * loss_aux_img +
                    lambda_aux_eeg * loss_aux_eeg +
                    lambda_reg * loss_reg)

            loss.backward()

            if config['training'].get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['max_grad_norm']
                )

            optimizer.step()

        if scheduler:
            scheduler.step()

        # Collect predictions and alpha values
        preds = outputs['fused_logits'].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_alphas.extend(outputs['alpha'].detach().cpu().numpy())

        total_loss += loss.item()
        total_loss_ce += loss_ce.item()
        total_loss_aux_img += loss_aux_img.item()
        total_loss_aux_eeg += loss_aux_eeg.item()
        total_loss_reg += loss_reg.item()

        avg_loss = total_loss / (batch_idx + 1)
        avg_alpha = np.mean(all_alphas[-len(preds):])

        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'alpha': f'{avg_alpha:.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    num_batches = len(dataloader)
    metrics = compute_metrics(
        np.array(all_preds),
        np.array(all_labels),
        config['data']['class_names']
    )

    metrics['loss'] = total_loss / num_batches
    metrics['loss_ce'] = total_loss_ce / num_batches
    metrics['loss_aux_img'] = total_loss_aux_img / num_batches
    metrics['loss_aux_eeg'] = total_loss_aux_eeg / num_batches
    metrics['loss_reg'] = total_loss_reg / num_batches
    metrics['alpha_mean'] = np.mean(all_alphas)
    metrics['alpha_std'] = np.std(all_alphas)

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict,
    use_amp: bool = True
) -> Dict:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_alphas = []
    all_entropies_img = []
    all_entropies_eeg = []

    pbar = tqdm(dataloader, desc='Validation')

    for batch in pbar:
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        eeg1 = batch['eeg1'].to(device)
        eeg2 = batch['eeg2'].to(device)
        labels = batch['labels'].to(device)

        if use_amp:
            with autocast('cuda'):
                outputs = model(img1, img2, eeg1, eeg2, labels)
                loss = F.cross_entropy(outputs['fused_logits'], labels)
        else:
            outputs = model(img1, img2, eeg1, eeg2, labels)
            loss = F.cross_entropy(outputs['fused_logits'], labels)

        preds = outputs['fused_logits'].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_alphas.extend(outputs['alpha'].cpu().numpy())

        # Collect entropies for analysis
        all_entropies_img.extend(outputs['aux_info']['entropies']['img'].cpu().numpy())
        all_entropies_eeg.extend(outputs['aux_info']['entropies']['eeg'].cpu().numpy())

        total_loss += loss.item()

    metrics = compute_metrics(
        np.array(all_preds),
        np.array(all_labels),
        config['data']['class_names']
    )

    metrics['loss'] = total_loss / len(dataloader)
    metrics['alpha_mean'] = np.mean(all_alphas)
    metrics['alpha_std'] = np.std(all_alphas)
    metrics['entropy_img_mean'] = np.mean(all_entropies_img)
    metrics['entropy_eeg_mean'] = np.mean(all_entropies_eeg)

    return metrics


def train(config: Dict, resume_path: Optional[str] = None):
    """Main training function."""
    set_seed(config['system']['seed'])

    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Get fusion mode
    fusion_mode = config['fusion'].get('mode', 'full')

    # Create output directory
    save_dir = Path(config['checkpoint']['save_dir']) / fusion_mode
    save_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Data
    # =========================================================================
    logger.info("Loading datasets...")
    train_dataset, val_dataset = prepare_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=multimodal_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=multimodal_collate_fn,
        pin_memory=True
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # =========================================================================
    # Models
    # =========================================================================
    logger.info("Creating models...")

    # Gaze Encoder (EarlyFusionViT with channel concatenation)
    gaze_encoder = EarlyFusionViT(
        model_name=config['gaze_encoder']['model_name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['gaze_encoder'].get('pretrained', True),
        img_size=config['data'].get('image_size', 224),
        fusion_mode=config['gaze_encoder'].get('fusion_mode', 'concat'),
        weight_init_strategy=config['gaze_encoder'].get('weight_init_strategy', 'duplicate')
    )

    if config['gaze_encoder'].get('checkpoint_path'):
        gaze_encoder = load_pretrained_encoder(
            gaze_encoder,
            config['gaze_encoder']['checkpoint_path'],
            encoder_type='gaze'
        )

    # EEG Encoder (same initialization pattern as train_art.py)
    eeg_encoder = DualEEGTransformer(
        in_channels=config['eeg_encoder']['in_channels'],
        num_classes=config['model']['num_classes'],
        d_model=config['eeg_encoder']['d_model'],
        num_layers=config['eeg_encoder']['num_layers'],
        num_heads=config['eeg_encoder']['num_heads'],
        d_ff=config['eeg_encoder']['d_ff'],
        dropout=config['eeg_encoder'].get('dropout', 0.1),
        max_len=config['data']['window_size'] // 4,  # After conv downsampling (same as train_art.py)
        conv_kernel_size=config['eeg_encoder'].get('conv_kernel_size', 25),
        conv_stride=config['eeg_encoder'].get('conv_stride', 4),
        conv_layers=config['eeg_encoder'].get('conv_layers', 2),
        sampling_rate=config['data'].get('sampling_rate', 256),
        use_spectrogram=config['eeg_encoder'].get('use_spectrogram', True),
        use_robust_ibs=config['eeg_encoder'].get('use_robust_ibs', True),
        use_ibs=config['eeg_encoder'].get('use_ibs', True),
        use_cross_attention=config['eeg_encoder'].get('use_cross_attention', True),
    )

    if config['eeg_encoder'].get('checkpoint_path'):
        eeg_encoder = load_pretrained_encoder(
            eeg_encoder,
            config['eeg_encoder']['checkpoint_path'],
            encoder_type='eeg'
        )

    # Fusion Module
    fusion_module = FuzzyGatingFusion(
        num_classes=config['model']['num_classes'],
        mode=fusion_mode,
        eps_temp=config['fusion'].get('eps_temp', 0.1),
    )

    # Combined Model
    model = MultimodalFusionModel(
        gaze_encoder=gaze_encoder,
        eeg_encoder=eeg_encoder,
        fusion_module=fusion_module,
        freeze_gaze=config['gaze_encoder'].get('freeze', False),
        freeze_eeg=config['eeg_encoder'].get('freeze', False),
    )
    model = model.to(device)

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Fusion mode: {fusion_mode}")

    # =========================================================================
    # Optimizer
    # =========================================================================
    # Different learning rates for different components
    encoder_lr = config['training']['encoder_learning_rate']
    fusion_lr = config['training']['fusion_learning_rate']

    param_groups = [
        {'params': model.gaze_encoder.parameters(), 'lr': encoder_lr},
        {'params': model.eeg_encoder.parameters(), 'lr': encoder_lr},
        {'params': model.fusion.parameters(), 'lr': fusion_lr},
    ]

    optimizer = AdamW(
        param_groups,
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
    logger.info(f"Mixed precision (FP16): {use_amp}")

    # =========================================================================
    # Resume
    # =========================================================================
    start_epoch = 0
    best_metric = 0.0

    if resume_path or config['resume'].get('enabled', False):
        ckpt_path = resume_path or config['resume']['checkpoint_path']
        if ckpt_path and Path(ckpt_path).exists():
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            start_epoch, best_metric = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler
            )
            start_epoch += 1
            logger.info(f"Resuming from epoch {start_epoch}, best_metric: {best_metric:.4f}")

    # =========================================================================
    # Wandb
    # =========================================================================
    if WANDB_AVAILABLE and config['wandb'].get('enabled', False):
        run_name = f"{config['wandb']['run_name']}_{fusion_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config['wandb']['project'],
            name=run_name,
            config=config,
            tags=config['wandb'].get('tags', []),
            notes=config['wandb'].get('notes', ''),
            resume='allow' if resume_path else None
        )
        wandb.watch(model, log='all', log_freq=100)
        logger.info(f"Wandb initialized: {config['wandb']['project']}/{run_name}")

    # =========================================================================
    # Training Loop
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Starting Multimodal Fuzzy Fusion Training")
    logger.info("=" * 60)

    metric_for_best = config['checkpoint'].get('metric_for_best', 'val_f1')
    greater_is_better = config['checkpoint'].get('greater_is_better', True)

    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        logger.info(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, device, epoch, config, use_amp
        )

        logger.info(f"\n[Train] Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.4f}, "
                    f"F1: {train_metrics['f1']:.4f}, "
                    f"Alpha: {train_metrics['alpha_mean']:.3f}±{train_metrics['alpha_std']:.3f}")

        # Validate
        val_metrics = validate(model, val_loader, device, config, use_amp)

        logger.info(f"[Val]   Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}, "
                    f"Alpha: {val_metrics['alpha_mean']:.3f}±{val_metrics['alpha_std']:.3f}")

        logger.info(f"\n[Val] Confusion Matrix:")
        logger.info(f"\n{val_metrics['confusion_matrix']}")

        # Log fusion parameters
        logger.info(f"[Fusion] T_img: {model.fusion.temp_img.item():.4f}, "
                    f"T_eeg: {model.fusion.temp_eeg.item():.4f}")
        logger.info(f"[Fusion] H_img: {val_metrics['entropy_img_mean']:.4f}, "
                    f"H_eeg: {val_metrics['entropy_eeg_mean']:.4f}")

        # Check best
        current_metric = val_metrics[metric_for_best.replace('val_', '')]
        if greater_is_better:
            is_best = current_metric > best_metric
        else:
            is_best = current_metric < best_metric

        if is_best:
            best_metric = current_metric
            logger.info(f"New best {metric_for_best}: {best_metric:.4f}")

        # Save checkpoints
        if config['checkpoint'].get('save_best', True) and is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_metric, config,
                save_dir / 'best_model.pt', is_best=True
            )
            logger.info("Saved best model checkpoint")

        if (epoch + 1) % config['checkpoint'].get('save_every_epochs', 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_metric, config,
                save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            )
            logger.info(f"Saved checkpoint at epoch {epoch + 1}")

        # Wandb logging
        if WANDB_AVAILABLE and config['wandb'].get('enabled', False):
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/loss_ce': train_metrics['loss_ce'],
                'train/loss_aux_img': train_metrics['loss_aux_img'],
                'train/loss_aux_eeg': train_metrics['loss_aux_eeg'],
                'train/loss_reg': train_metrics['loss_reg'],
                'train/accuracy': train_metrics['accuracy'],
                'train/f1': train_metrics['f1'],
                'train/alpha_mean': train_metrics['alpha_mean'],
                'train/alpha_std': train_metrics['alpha_std'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'val/alpha_mean': val_metrics['alpha_mean'],
                'val/alpha_std': val_metrics['alpha_std'],
                'val/entropy_img_mean': val_metrics['entropy_img_mean'],
                'val/entropy_eeg_mean': val_metrics['entropy_eeg_mean'],
                'fusion/temp_img': model.fusion.temp_img.item(),
                'fusion/temp_eeg': model.fusion.temp_eeg.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
            }
            wandb.log(log_dict)

    # =========================================================================
    # Final
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best {metric_for_best}: {best_metric:.4f}")
    logger.info("=" * 60)

    save_checkpoint(
        model, optimizer, scheduler, scaler,
        config['training']['epochs'] - 1, best_metric, config,
        save_dir / 'final_model.pt'
    )

    if WANDB_AVAILABLE and config['wandb'].get('enabled', False):
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Fuzzy Fusion')
    parser.add_argument(
        '--config',
        type=str,
        default='4_Experiments/configs/multimodal_fuzzy_fusion.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()
