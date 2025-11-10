"""
prompt:
以Data/metadata/complete_metadata.json為原始資料，用ViT跑圖片分類，將player1,2 concate起來，訓練過程請完全依照Hugging Face的模型規則，詳細步驟如下：
1. 把原始json拆成train/test
	1. dataset = load_dataset("json", data_files="./image_gt.json", split="train")#[:5%]
	2. datasets = dataset.train_test_split(test_size=0.02)
2. models/backbones/vit.py：載入 ViTForImageClassification
3. experiments/configs/vit_single_vs_competition.yaml
4. experiments/scripts/train_vit.py
5. metrics/classification.py
===================================================
Training script for ViT image classification
Following Hugging Face Transformers best practices
"""

import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Any
import logging
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from metrics.classification import compute_metrics
from Data.processed.two_image_fusion import DualImageDataset

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
    Prepare train and test datasets following HuggingFace pattern

    Args:
        config: Configuration dictionary

    Returns:
        DatasetDict with train and test splits
    """
    logger.info("Loading dataset from JSON...")

    # Load dataset using HuggingFace datasets
    metadata_path = config['data']['metadata_path']
    datasets = load_dataset("json", data_files=metadata_path, split="train")

    logger.info(f"Total samples: {len(datasets)}")

    # Train/test split
    test_size = config['data']['train_test_split']
    seed = config['data']['random_seed']

    split_datasets = datasets.train_test_split(
        test_size=test_size,
        seed=seed
    )

    logger.info(f"Train samples: {len(split_datasets['train'])}")
    logger.info(f"Test samples: {len(split_datasets['test'])}")

    return split_datasets


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


def get_last_checkpoint(output_dir: str):
    """
    Get the last checkpoint from the output directory

    Args:
        output_dir: Directory to search for checkpoints

    Returns:
        Path to last checkpoint or None if not found
    """
    if not os.path.isdir(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, d))
    ]

    if not checkpoints:
        return None

    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1]


def get_wandb_run_id(checkpoint_path: str):
    """
    Extract wandb run_id from checkpoint directory

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        wandb run_id or None
    """
    trainer_state_file = os.path.join(checkpoint_path, 'trainer_state.json')
    if os.path.exists(trainer_state_file):
        try:
            import json
            with open(trainer_state_file, 'r') as f:
                state = json.load(f)
            # Try to get wandb run_id from log_history
            if 'log_history' in state and len(state['log_history']) > 0:
                # Check if wandb run info is stored
                return None  # Will be handled by wandb auto-resume
        except:
            pass
    return None


def main(config_path: str, resume: bool = False, checkpoint_path: str = None):
    """Main training function

    Args:
        config_path: Path to config YAML file
        resume: Whether to resume from last checkpoint
        checkpoint_path: Specific checkpoint path to resume from (overrides resume)
    """

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Set random seed
    seed = config['system']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Check for existing checkpoint
    resume_from_checkpoint = None
    output_dir = config['training']['output_dir']

    if checkpoint_path:
        # Use specified checkpoint
        if os.path.isdir(checkpoint_path):
            resume_from_checkpoint = checkpoint_path
            logger.info(f"Resuming from specified checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"Specified checkpoint not found: {checkpoint_path}")
    elif resume:
        # Auto-detect last checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            resume_from_checkpoint = last_checkpoint
            logger.info(f"Auto-detected checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("No checkpoint found, starting from scratch")

    # Initialize wandb
    wandb_config = config.get('wandb', {})

    # Try to resume wandb run if resuming training
    wandb_resume = "allow"  # allow resuming if run_id matches
    wandb_id = None

    if resume_from_checkpoint:
        # Try to load wandb run_id from checkpoint
        wandb_run_path = os.path.join(os.path.dirname(resume_from_checkpoint), 'wandb')
        if os.path.exists(wandb_run_path):
            # Find latest run directory
            run_dirs = [d for d in os.listdir(wandb_run_path) if d.startswith('run-')]
            if run_dirs:
                run_dirs.sort()
                latest_run = run_dirs[-1]
                wandb_id = latest_run.split('-')[-1]  # Extract run ID
                wandb_resume = "must"
                logger.info(f"Resuming wandb run: {wandb_id}")

    wandb.init(
        project=wandb_config.get('project', 'eyegaze-vit-classification'),
        name=wandb_config.get('run_name', None),
        id=wandb_id,
        resume=wandb_resume,
        config={
            'model': config['model'],
            'training': config['training'],
            'data': {k: v for k, v in config['data'].items() if k != 'image_base_path'},  # Exclude path
        },
        tags=wandb_config.get('tags', ['vit', 'dual-image', 'eyegaze']),
        notes=wandb_config.get('notes', 'ViT training for dual-image eye-gaze classification'),
    )
    logger.info(f"Wandb run initialized: {wandb.run.name}")

    if resume_from_checkpoint:
        logger.info(f"Training will resume from checkpoint: {resume_from_checkpoint}")

    # Prepare datasets
    split_datasets = prepare_datasets(config)

    # Initialize image processor
    model_name = config['model']['model_name']
    image_processor = ViTImageProcessor.from_pretrained(model_name)

    # Get label mappings
    label2id = config['data']['label2id']
    id2label = config['data']['id2label']
    id2label = {int(k): v for k, v in id2label.items()}  # Convert keys to int

    # Create PyTorch datasets
    concat_mode = config['model']['concat_mode']
    image_base_path = config['data']['image_base_path']

    train_dataset = DualImageDataset(
        split_datasets['train'],
        image_processor,
        image_base_path,
        label2id,
        concat_mode
    )

    test_dataset = DualImageDa taset(
        split_datasets['test'],
        image_processor,
        image_base_path,
        label2id,
        concat_mode
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"concat_mode: {concat_mode}")

    # Load model
    logger.info("Loading ViT model...")
    num_labels = config['model']['num_labels']

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Freeze backbone if specified
    if config['model'].get('freeze_backbone', False):
        logger.info("Freezing backbone parameters...")
        for param in model.vit.parameters():
            param.requires_grad = False

    # Training arguments (Following HuggingFace Trainer)
    training_config = config['training']

    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_ratio=training_config['warmup_ratio'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        eval_strategy=training_config['evaluation_strategy'],  # Changed from evaluation_strategy
        save_strategy=training_config['save_strategy'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        greater_is_better=training_config['greater_is_better'],
        logging_dir=training_config['logging_dir'],
        logging_strategy=training_config['logging_strategy'],
        logging_steps=training_config['logging_steps'],
        report_to=training_config['report_to'],
        fp16=training_config.get('fp16', False),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
        remove_unused_columns=training_config.get('remove_unused_columns', False),
        seed=seed,
        data_seed=seed,
    )

    # Callbacks (Early stopping removed as per user request)
    callbacks = []
    # Note: Early stopping has been disabled to allow full training epochs

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=lambda pred: compute_metrics(pred, id2label),
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    if resume_from_checkpoint:
        logger.info(f"Resuming training from: {resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        logger.info("Starting training from scratch")
        train_result = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    logger.info("Test Results:")
    for key, value in test_results.items():
        logger.info(f"  {key}: {value}")

    # Log final test results to wandb
    wandb.log({f"test/{k}": v for k, v in test_results.items()})

    # Save results
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'test_results.txt', 'w') as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")

    logger.info("Training completed!")

    # Finish wandb run
    wandb.finish()
    logger.info("Wandb run finished")

    return trainer, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT for image classification")
    parser.add_argument(
        "--config",
        type=str,
        default="Experiments/configs/vit_single_vs_competition.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint (auto-detect)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume from (overrides --resume)"
    )

    args = parser.parse_args()

    main(args.config, resume=args.resume, checkpoint_path=args.checkpoint)
