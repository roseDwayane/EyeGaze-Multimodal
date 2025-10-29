"""
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
    Trainer,
    EarlyStoppingCallback
)
from typing import Dict, List, Any
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from metrics.classification import compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DualImageDataset(Dataset):
    """
    Dataset for dual-image classification (player1 + player2)
    Concatenates two images horizontally or vertically
    """

    def __init__(
        self,
        dataset,
        image_processor: ViTImageProcessor,
        image_base_path: str,
        label2id: Dict[str, int],
        concat_mode: str = "horizontal"
    ):
        """
        Initialize dataset

        Args:
            dataset: HuggingFace dataset object
            image_processor: ViT image processor
            image_base_path: Base path for images
            label2id: Label to ID mapping
            concat_mode: "horizontal" or "vertical" concatenation
        """
        self.dataset = dataset
        self.image_processor = image_processor
        self.image_base_path = Path(image_base_path)
        self.label2id = label2id
        self.concat_mode = concat_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get item with concatenated images"""
        item = self.dataset[idx]

        # Load player1 and player2 images
        player1_path = self.image_base_path / f"{item['player1']}.jpg"
        player2_path = self.image_base_path / f"{item['player2']}.jpg"

        try:
            img1 = Image.open(player1_path).convert('RGB')
            img2 = Image.open(player2_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading images for idx {idx}: {e}")
            # Return a blank image as fallback
            img1 = Image.new('RGB', (224, 224), color='white')
            img2 = Image.new('RGB', (224, 224), color='white')

        # Concatenate images
        if self.concat_mode == "horizontal":
            # Concatenate horizontally (side by side)
            total_width = img1.width + img2.width
            max_height = max(img1.height, img2.height)
            concatenated = Image.new('RGB', (total_width, max_height))
            concatenated.paste(img1, (0, 0))
            concatenated.paste(img2, (img1.width, 0))
        elif self.concat_mode == "vertical":
            # Concatenate vertically (top to bottom)
            max_width = max(img1.width, img2.width)
            total_height = img1.height + img2.height
            concatenated = Image.new('RGB', (max_width, total_height))
            concatenated.paste(img1, (0, 0))
            concatenated.paste(img2, (0, img1.height))
        else:
            raise ValueError(f"Invalid concat_mode: {self.concat_mode}")

        # Process image using ViT processor
        inputs = self.image_processor(concatenated, return_tensors="pt")

        # Get label
        label = self.label2id[item['class']]

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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
        seed=seed,
        stratify_by_column='class'  # Stratified split by class
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


def main(config_path: str):
    """Main training function"""

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Set random seed
    seed = config['system']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    concat_mode = config['model'].get('concat_mode', 'horizontal')
    image_base_path = config['data']['image_base_path']

    train_dataset = DualImageDataset(
        split_datasets['train'],
        image_processor,
        image_base_path,
        label2id,
        concat_mode
    )

    test_dataset = DualImageDataset(
        split_datasets['test'],
        image_processor,
        image_base_path,
        label2id,
        concat_mode
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

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
        evaluation_strategy=training_config['evaluation_strategy'],
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

    # Callbacks
    callbacks = []
    if 'early_stopping_patience' in training_config:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_config['early_stopping_patience'],
                early_stopping_threshold=training_config.get('early_stopping_threshold', 0.0)
            )
        )

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

    # Save results
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'test_results.txt', 'w') as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")

    logger.info("Training completed!")

    return trainer, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT for image classification")
    parser.add_argument(
        "--config",
        type=str,
        default="Experiments/configs/vit_single_vs_competition.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    main(args.config)
