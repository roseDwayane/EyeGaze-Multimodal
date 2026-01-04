"""
Dual Image Dataset V2
不進行fusion，返回兩張獨立的圖像
用於可學習fusion模型
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Dict
from transformers import ViTImageProcessor
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DualImageDatasetV2(Dataset):
    """
    返回兩張獨立圖像的Dataset

    與原版DualImageDataset的區別：
    - 不在數據層做fusion
    - 返回player1和player2的獨立圖像
    - Fusion操作延後到模型中進行
    """

    def __init__(
        self,
        dataset,
        image_processor: ViTImageProcessor,
        image_base_path: str,
        label2id: Dict[str, int]
    ):
        """
        Args:
            dataset: HuggingFace dataset object
            image_processor: ViT image processor
            image_base_path: Base path for images
            label2id: Label to ID mapping
        """
        self.dataset = dataset
        self.image_processor = image_processor
        self.image_base_path = Path(image_base_path)
        self.label2id = label2id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            {
                'img1': Player1圖像 [C, H, W],
                'img2': Player2圖像 [C, H, W],
                'labels': 標籤 (int)
            }
        """
        item = self.dataset[idx]

        # 載入player1和player2圖像
        player1_path = self.image_base_path / f"{item['player1']}.jpg"
        player2_path = self.image_base_path / f"{item['player2']}.jpg"

        try:
            img1 = Image.open(player1_path).convert('RGB')
            img2 = Image.open(player2_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading images for idx {idx}: {e}")
            # Fallback：返回白色圖像
            img1 = Image.new('RGB', (224, 224), color='white')
            img2 = Image.new('RGB', (224, 224), color='white')

        # 使用ViT processor處理圖像
        # 注意：processor會resize到224x224並標準化
        inputs1 = self.image_processor(img1, return_tensors="pt")
        inputs2 = self.image_processor(img2, return_tensors="pt")

        # 獲取標籤
        label = self.label2id[item['class']]

        return {
            'img1': inputs1['pixel_values'].squeeze(0),  # [C, H, W]
            'img2': inputs2['pixel_values'].squeeze(0),  # [C, H, W]
            'labels': torch.tensor(label, dtype=torch.long)
        }


def collate_fn_v2(batch):
    """
    Custom collate function for DualImageDatasetV2

    Args:
        batch: List of dataset items

    Returns:
        Batched dictionary
    """
    img1 = torch.stack([item['img1'] for item in batch])
    img2 = torch.stack([item['img2'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'img1': img1,
        'img2': img2,
        'labels': labels
    }


class AugmentedDualImageDataset(DualImageDatasetV2):
    """
    帶數據增強的Dual Image Dataset

    在原版基礎上加入數據增強
    """

    def __init__(
        self,
        dataset,
        image_processor: ViTImageProcessor,
        image_base_path: str,
        label2id: Dict[str, int],
        augmentation_config: Dict = None
    ):
        super().__init__(dataset, image_processor, image_base_path, label2id)

        self.augmentation_config = augmentation_config or {}
        self.use_augmentation = self.augmentation_config.get('enabled', False)

        if self.use_augmentation:
            self._setup_augmentation()

    def _setup_augmentation(self):
        """設置數據增強"""
        from torchvision import transforms

        aug_transforms = []

        # Random horizontal flip
        if self.augmentation_config.get('random_horizontal_flip', 0) > 0:
            aug_transforms.append(
                transforms.RandomHorizontalFlip(
                    p=self.augmentation_config['random_horizontal_flip']
                )
            )

        # Random rotation
        if self.augmentation_config.get('random_rotation', 0) > 0:
            aug_transforms.append(
                transforms.RandomRotation(
                    degrees=self.augmentation_config['random_rotation']
                )
            )

        # Color jitter
        if 'color_jitter' in self.augmentation_config:
            cj = self.augmentation_config['color_jitter']
            aug_transforms.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0),
                    contrast=cj.get('contrast', 0),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0)
                )
            )

        if aug_transforms:
            self.augmentation = transforms.Compose(aug_transforms)
        else:
            self.augmentation = None

    def __getitem__(self, idx):
        """Get item with augmentation"""
        item = self.dataset[idx]

        # 載入圖像
        player1_path = self.image_base_path / f"{item['player1']}.jpg"
        player2_path = self.image_base_path / f"{item['player2']}.jpg"

        try:
            img1 = Image.open(player1_path).convert('RGB')
            img2 = Image.open(player2_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading images for idx {idx}: {e}")
            img1 = Image.new('RGB', (224, 224), color='white')
            img2 = Image.new('RGB', (224, 224), color='white')

        # 應用數據增強（如果啟用）
        if self.use_augmentation and self.augmentation:
            # 對兩張圖像應用相同的增強（保持一致性）
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            img1 = self.augmentation(img1)
            torch.manual_seed(seed)
            img2 = self.augmentation(img2)

        # 使用ViT processor
        inputs1 = self.image_processor(img1, return_tensors="pt")
        inputs2 = self.image_processor(img2, return_tensors="pt")

        label = self.label2id[item['class']]

        return {
            'img1': inputs1['pixel_values'].squeeze(0),
            'img2': inputs2['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


if __name__ == "__main__":
    # 測試Dataset
    from datasets import load_dataset
    from transformers import ViTImageProcessor

    print("=" * 80)
    print("測試DualImageDatasetV2")
    print("=" * 80)

    # 載入測試數據
    metadata_path = "Data/metadata/complete_metadata.json"
    image_base_path = "G:/共用雲端硬碟/CNElab_林佳誼_Gaze/B.GazeImage/01.data/bgOn_heatmapOn_trajOn"

    datasets = load_dataset("json", data_files=metadata_path, split="train")
    split_datasets = datasets.train_test_split(test_size=0.2, seed=42)

    label2id = {
        "Single": 0,
        "Competition": 1,
        "Cooperation": 2
    }

    # 初始化processor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # 創建dataset
    dataset = DualImageDatasetV2(
        split_datasets['train'],
        image_processor,
        image_base_path,
        label2id
    )

    print(f"\nDataset size: {len(dataset)}")

    # 測試幾個樣本
    print("\n測試樣本:")
    for i in range(3):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  img1 shape: {sample['img1'].shape}")
        print(f"  img2 shape: {sample['img2'].shape}")
        print(f"  labels: {sample['labels'].item()}")
        print(f"  img1 range: [{sample['img1'].min():.3f}, {sample['img1'].max():.3f}]")
        print(f"  img2 range: [{sample['img2'].min():.3f}, {sample['img2'].max():.3f}]")

    # 測試collate function
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn_v2
    )

    print("\n測試DataLoader:")
    for batch in dataloader:
        print(f"  img1 batch shape: {batch['img1'].shape}")
        print(f"  img2 batch shape: {batch['img2'].shape}")
        print(f"  labels batch shape: {batch['labels'].shape}")
        break

    print("\n" + "=" * 80)
    print("測試完成！")
    print("=" * 80)
