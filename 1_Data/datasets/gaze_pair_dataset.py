"""
Gaze Pair Dataset for Dual-stream Gaze Heatmap Classification

This module provides a PyTorch Dataset class for loading paired gaze heatmaps
from two players for social interaction classification.

Classes:
    - Single (0): Single-player mode
    - Competition (1): Competitive interaction
    - Cooperation (2): Cooperative interaction
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class GazePairDataset(Dataset):
    """
    PyTorch Dataset for loading paired gaze heatmaps.

    Each sample consists of two RGB heatmap images (from Player 1 and Player 2)
    and a class label indicating the interaction type.

    Args:
        metadata: List of metadata dictionaries containing sample info
        image_base_path: Base directory containing the gaze heatmap images
        image_extension: File extension for images (default: ".jpg")
        label2id: Mapping from class name to integer label
        transform: Optional transform to apply to images
        return_metadata: If True, also return metadata dict

    Returns:
        img_a: Tensor (3, H, W) - Player 1 gaze heatmap
        img_b: Tensor (3, H, W) - Player 2 gaze heatmap
        label: int - Class label (0, 1, or 2)
    """

    def __init__(
        self,
        metadata: List[Dict],
        image_base_path: str,
        image_extension: str = ".jpg",
        label2id: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        self.metadata = metadata
        self.image_base_path = Path(image_base_path)
        self.image_extension = image_extension
        self.return_metadata = return_metadata

        # Default label mapping
        self.label2id = label2id or {
            "Single": 0,
            "Competition": 1,
            "Cooperation": 2
        }

        # Default transform (just ToTensor and Normalize)
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            img_a: Player 1 heatmap tensor (3, H, W)
            img_b: Player 2 heatmap tensor (3, H, W)
            label: Class label (int)
        """
        sample = self.metadata[idx]

        # Construct image paths
        img_a_path = self.image_base_path / f"{sample['player1']}{self.image_extension}"
        img_b_path = self.image_base_path / f"{sample['player2']}{self.image_extension}"

        # Load images
        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')

        # Apply transforms
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        # Get label
        label = self.label2id[sample['class']]

        if self.return_metadata:
            return img_a, img_b, label, sample

        return img_a, img_b, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.

        Returns inverse frequency weights normalized to sum to num_classes.

        Returns:
            weights: Tensor of shape (num_classes,)
        """
        from collections import Counter

        # Count class occurrences
        class_counts = Counter(self.label2id[s['class']] for s in self.metadata)
        num_classes = len(self.label2id)
        total_samples = len(self.metadata)

        # Compute inverse frequency weights
        weights = torch.zeros(num_classes)
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (num_classes * count)

        return weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        from collections import Counter
        return Counter(s['class'] for s in self.metadata)


def create_train_val_datasets(
    metadata_path: str,
    image_base_path: str,
    val_pairs: List[int],
    image_size: int = 224,
    image_extension: str = ".jpg",
    label2id: Optional[Dict[str, int]] = None,
    augmentation_config: Optional[Dict] = None
) -> Tuple[GazePairDataset, GazePairDataset]:
    """
    Create train and validation datasets split by Pair ID.

    Args:
        metadata_path: Path to complete_metadata.json
        image_base_path: Base directory for images
        val_pairs: List of Pair IDs to use for validation
        image_size: Target image size
        image_extension: Image file extension
        label2id: Class name to label mapping
        augmentation_config: Augmentation settings for training

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)

    # Split by Pair ID
    train_metadata = [s for s in all_metadata if s['pair'] not in val_pairs]
    val_metadata = [s for s in all_metadata if s['pair'] in val_pairs]

    print(f"[Dataset] Train samples: {len(train_metadata)}, Val samples: {len(val_metadata)}")
    print(f"[Dataset] Val pairs: {val_pairs}")

    # Default label mapping
    label2id = label2id or {
        "Single": 0,
        "Competition": 1,
        "Cooperation": 2
    }

    # Build transforms
    # --- Training transform (with augmentation) ---
    train_transform_list = [
        T.Resize((image_size, image_size)),
    ]

    if augmentation_config:
        flip_prob = augmentation_config.get('random_horizontal_flip', 0)
        if flip_prob > 0:
            train_transform_list.append(T.RandomHorizontalFlip(p=flip_prob))

    train_transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    train_transform = T.Compose(train_transform_list)

    # --- Validation transform (no augmentation) ---
    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = GazePairDataset(
        metadata=train_metadata,
        image_base_path=image_base_path,
        image_extension=image_extension,
        label2id=label2id,
        transform=train_transform
    )

    val_dataset = GazePairDataset(
        metadata=val_metadata,
        image_base_path=image_base_path,
        image_extension=image_extension,
        label2id=label2id,
        transform=val_transform
    )

    # Print class distributions
    print(f"[Dataset] Train distribution: {train_dataset.get_class_distribution()}")
    print(f"[Dataset] Val distribution: {val_dataset.get_class_distribution()}")

    return train_dataset, val_dataset


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Stacks img_a, img_b, and labels into batches.

    Args:
        batch: List of (img_a, img_b, label) tuples

    Returns:
        img_a_batch: (B, 3, H, W)
        img_b_batch: (B, 3, H, W)
        labels: (B,)
    """
    img_a_list, img_b_list, labels = zip(*batch)

    img_a_batch = torch.stack(img_a_list, dim=0)
    img_b_batch = torch.stack(img_b_list, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return img_a_batch, img_b_batch, labels


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test the dataset
    print("=" * 60)
    print("Testing GazePairDataset")
    print("=" * 60)

    metadata_path = "1_Data/metadata/complete_metadata.json"
    image_base_path = "G:/共用雲端硬碟/CNElab_林佳誼_Gaze/B.GazeImage/01.data/bgOn_heatmapOn_trajOn"
    val_pairs = [33, 34, 35, 36, 37, 38, 39, 40]

    train_dataset, val_dataset = create_train_val_datasets(
        metadata_path=metadata_path,
        image_base_path=image_base_path,
        val_pairs=val_pairs,
        image_size=224,
        augmentation_config={'random_horizontal_flip': 0.5}
    )

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Test loading a sample
    img_a, img_b, label = train_dataset[0]
    print(f"\nSample shapes:")
    print(f"  img_a: {img_a.shape}")
    print(f"  img_b: {img_b.shape}")
    print(f"  label: {label}")

    # Test class weights
    weights = train_dataset.get_class_weights()
    print(f"\nClass weights: {weights}")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    img_a_batch, img_b_batch, labels = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  img_a_batch: {img_a_batch.shape}")
    print(f"  img_b_batch: {img_b_batch.shape}")
    print(f"  labels: {labels.shape}")

    print("\n[OK] All tests passed!")
