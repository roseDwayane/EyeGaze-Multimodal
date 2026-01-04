"""
Dual Image Fusion Module
Fuses player1 and player2 images for ViT classification
Supports multiple fusion strategies: concatenation, addition, multiplication, subtraction
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


class DualImageDataset(Dataset):
    """
    Dataset for dual-image classification (player1 + player2)
    Supports multiple fusion strategies:
    - Concatenation: "horizontal", "vertical"
    - Pixel-level fusion: "add", "multiply", "subtract"
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
            concat_mode: Fusion mode - one of:
                - "horizontal": concatenate side by side
                - "vertical": concatenate top to bottom
                - "add": pixel-wise addition (averaged)
                - "multiply": pixel-wise multiplication (normalized)
                - "subtract": pixel-wise subtraction (absolute difference)
        """
        self.dataset = dataset
        self.image_processor = image_processor
        self.image_base_path = Path(image_base_path)
        self.label2id = label2id
        self.concat_mode = concat_mode

        # Validate concat_mode
        valid_modes = ["horizontal", "vertical", "add", "multiply", "subtract"]
        if self.concat_mode not in valid_modes:
            raise ValueError(
                f"Invalid concat_mode: {self.concat_mode}. "
                f"Must be one of {valid_modes}"
            )

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

        # Fuse images based on concat_mode
        if self.concat_mode == "horizontal":
            # Concatenate horizontally (side by side)
            total_width = img1.width + img2.width
            max_height = max(img1.height, img2.height)
            fused = Image.new('RGB', (total_width, max_height))
            fused.paste(img1, (0, 0))
            fused.paste(img2, (img1.width, 0))

        elif self.concat_mode == "vertical":
            # Concatenate vertically (top to bottom)
            max_width = max(img1.width, img2.width)
            total_height = img1.height + img2.height
            fused = Image.new('RGB', (max_width, total_height))
            fused.paste(img1, (0, 0))
            fused.paste(img2, (0, img1.height))

        elif self.concat_mode == "add":
            # Pixel-wise addition (averaged to prevent overflow)
            # Resize to same size if needed
            if img1.size != img2.size:
                target_size = img1.size  # Use img1's size as reference
                img2 = img2.resize(target_size, Image.BILINEAR)

            # Convert to numpy arrays
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)

            # Add and average to keep values in [0, 255]
            fused_arr = (arr1 + arr2) / 2.0
            fused_arr = np.clip(fused_arr, 0, 255).astype(np.uint8)
            fused = Image.fromarray(fused_arr)

        elif self.concat_mode == "multiply":
            # Pixel-wise multiplication (normalized)
            # Resize to same size if needed
            if img1.size != img2.size:
                target_size = img1.size
                img2 = img2.resize(target_size, Image.BILINEAR)

            # Convert to numpy arrays and normalize to [0, 1]
            arr1 = np.array(img1, dtype=np.float32) / 255.0
            arr2 = np.array(img2, dtype=np.float32) / 255.0

            # Multiply and scale back to [0, 255]
            fused_arr = arr1 * arr2 * 255.0
            fused_arr = np.clip(fused_arr, 0, 255).astype(np.uint8)
            fused = Image.fromarray(fused_arr)

        elif self.concat_mode == "subtract":
            # Pixel-wise subtraction (absolute difference)
            # Resize to same size if needed
            if img1.size != img2.size:
                target_size = img1.size
                img2 = img2.resize(target_size, Image.BILINEAR)

            # Convert to numpy arrays
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)

            # Absolute difference
            fused_arr = np.abs(arr1 - arr2)
            fused_arr = np.clip(fused_arr, 0, 255).astype(np.uint8)
            fused = Image.fromarray(fused_arr)

        else:
            raise ValueError(f"Invalid concat_mode: {self.concat_mode}")

        # Process fused image using ViT processor
        inputs = self.image_processor(fused, return_tensors="pt")

        # Get label
        label = self.label2id[item['class']]

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def test_dual_image_fusion(
    metadata_path: str = "Data/metadata/complete_metadata.json",
    image_base_path: str = "Data/raw/Gaze/example",
    num_samples: int = 5,
    concat_mode: str = "horizontal",
    save_output: bool = True
):
    """
    Test the dual image fusion functionality

    Args:
        metadata_path: Path to metadata JSON
        image_base_path: Base path for images
        num_samples: Number of samples to test
        concat_mode: "horizontal" or "vertical"
        save_output: Whether to save concatenated images
    """
    import json
    import os
    from datasets import load_dataset

    print("=" * 80)
    print("TESTING DUAL IMAGE FUSION")
    print("=" * 80)

    # Load metadata
    print(f"\n1. Loading metadata from: {metadata_path}")
    datasets = load_dataset("json", data_files=metadata_path, split="train")
    print(f"   Total samples: {len(datasets)}")

    # Create label mapping
    label2id = {
        "Single": 0,
        "Competition": 1,
        "Cooperation": 2
    }

    # Initialize image processor
    print(f"\n2. Initializing ViT image processor...")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    print(f"   Image size: {image_processor.size}")

    # Create dataset
    print(f"\n3. Creating DualImageDataset with {concat_mode} concatenation...")
    dataset = DualImageDataset(
        datasets,
        image_processor,
        image_base_path,
        label2id,
        concat_mode
    )
    print(f"   Dataset size: {len(dataset)}")

    # Test samples
    print(f"\n4. Testing {num_samples} samples...")
    print("-" * 80)

    for i in range(min(num_samples, len(dataset))):
        try:
            # Get sample
            sample = dataset[i]
            raw_item = datasets[i]

            print(f"\nSample {i+1}:")
            print(f"  Player1: {raw_item['player1']}")
            print(f"  Player2: {raw_item['player2']}")
            print(f"  Class: {raw_item['class']} (label: {sample['labels'].item()})")
            print(f"  Pixel values shape: {sample['pixel_values'].shape}")
            print(f"  Pixel values dtype: {sample['pixel_values'].dtype}")
            print(f"  Pixel values range: [{sample['pixel_values'].min():.3f}, {sample['pixel_values'].max():.3f}]")

            # Save concatenated image if requested
            if save_output:
                # Load original images
                player1_path = Path(image_base_path) / f"{raw_item['player1']}.jpg"
                player2_path = Path(image_base_path) / f"{raw_item['player2']}.jpg"

                if player1_path.exists() and player2_path.exists():
                    img1 = Image.open(player1_path).convert('RGB')
                    img2 = Image.open(player2_path).convert('RGB')

                    # Concatenate
                    if concat_mode == "horizontal":
                        total_width = img1.width + img2.width
                        max_height = max(img1.height, img2.height)
                        concatenated = Image.new('RGB', (total_width, max_height))
                        concatenated.paste(img1, (0, 0))
                        concatenated.paste(img2, (img1.width, 0))
                    else:
                        max_width = max(img1.width, img2.width)
                        total_height = img1.height + img2.height
                        concatenated = Image.new('RGB', (max_width, total_height))
                        concatenated.paste(img1, (0, 0))
                        concatenated.paste(img2, (0, img1.height))

                    # Save
                    output_dir = Path("Data/processed/test_outputs")
                    output_dir.mkdir(exist_ok=True, parents=True)
                    output_path = output_dir / f"sample_{i+1}_{concat_mode}.jpg"
                    concatenated.save(output_path)
                    print(f"  Saved concatenated image: {output_path}")

            print(f"  Status: OK")

        except Exception as e:
            print(f"  Status: FAILED - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "-" * 80)
    print("\n5. Summary:")
    print(f"   Concatenation mode: {concat_mode}")
    print(f"   Samples tested: {min(num_samples, len(dataset))}")
    print(f"   Image base path: {image_base_path}")
    print(f"   Output saved to: Data/processed/test_outputs/")

    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Test dual image fusion")
    parser.add_argument(
        "--metadata",
        type=str,
        default="Data/metadata/complete_metadata.json",
        help="Path to metadata JSON"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="Data/raw/Gaze/example",
        help="Base path for images"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--concat-mode",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="Concatenation mode"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output images"
    )

    args = parser.parse_args()

    test_dual_image_fusion(
        metadata_path=args.metadata,
        image_base_path=args.images,
        num_samples=args.num_samples,
        concat_mode=args.concat_mode,
        save_output=not args.no_save
    )