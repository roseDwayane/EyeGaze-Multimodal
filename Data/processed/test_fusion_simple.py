"""
Simple test script for dual image fusion
Does not require HuggingFace datasets library
"""

import json
import sys
from pathlib import Path
from PIL import Image
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def concat_images(img1_path, img2_path, concat_mode="horizontal"):
    """
    Fuse two images using various strategies

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        concat_mode: Fusion mode - one of:
            - "horizontal": concatenate side by side
            - "vertical": concatenate top to bottom
            - "add": pixel-wise addition (averaged)
            - "multiply": pixel-wise multiplication (normalized)
            - "subtract": pixel-wise subtraction (absolute difference)

    Returns:
        Fused PIL Image
    """
    import numpy as np

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    if concat_mode == "horizontal":
        # Concatenate horizontally (side by side)
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        fused = Image.new('RGB', (total_width, max_height))
        fused.paste(img1, (0, 0))
        fused.paste(img2, (img1.width, 0))

    elif concat_mode == "vertical":
        # Concatenate vertically (top to bottom)
        max_width = max(img1.width, img2.width)
        total_height = img1.height + img2.height
        fused = Image.new('RGB', (max_width, total_height))
        fused.paste(img1, (0, 0))
        fused.paste(img2, (0, img1.height))

    elif concat_mode == "add":
        # Pixel-wise addition (averaged)
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.BILINEAR)
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        fused_arr = (arr1 + arr2) / 2.0
        fused_arr = np.clip(fused_arr, 0, 255).astype(np.uint8)
        fused = Image.fromarray(fused_arr)

    elif concat_mode == "multiply":
        # Pixel-wise multiplication (normalized)
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.BILINEAR)
        arr1 = np.array(img1, dtype=np.float32) / 255.0
        arr2 = np.array(img2, dtype=np.float32) / 255.0
        fused_arr = arr1 * arr2 * 255.0
        fused_arr = np.clip(fused_arr, 0, 255).astype(np.uint8)
        fused = Image.fromarray(fused_arr)

    elif concat_mode == "subtract":
        # Pixel-wise subtraction (absolute difference)
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.BILINEAR)
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        fused_arr = np.abs(arr1 - arr2)
        fused_arr = np.clip(fused_arr, 0, 255).astype(np.uint8)
        fused = Image.fromarray(fused_arr)

    else:
        raise ValueError(f"Invalid concat_mode: {concat_mode}")

    return fused


def test_image_fusion(
    metadata_path="Data/metadata/complete_metadata.json",
    image_base_path="Data/raw/Gaze/example",
    num_samples=5,
    concat_mode="horizontal"
):
    """
    Simple test of image fusion without HuggingFace dependencies

    Args:
        metadata_path: Path to metadata JSON
        image_base_path: Base path for images
        num_samples: Number of samples to test
        concat_mode: "horizontal" or "vertical"
    """
    print("=" * 80)
    print("TESTING DUAL IMAGE FUSION (Simple Version)")
    print("=" * 80)

    # Load metadata
    print(f"\n1. Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"   Total samples: {len(metadata)}")

    # Create output directory
    output_dir = Path("Data/processed/test_outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"   Output directory: {output_dir}")

    # Test samples
    print(f"\n2. Testing {num_samples} samples with {concat_mode} concatenation...")
    print("-" * 80)

    success_count = 0
    fail_count = 0

    for i in range(min(num_samples, len(metadata))):
        item = metadata[i]

        try:
            print(f"\nSample {i+1}:")
            print(f"  Pair: {item.get('pair', 'N/A')}")
            print(f"  Player1: {item['player1']}")
            print(f"  Player2: {item['player2']}")
            print(f"  Class: {item['class']}")

            # Get image paths
            player1_path = Path(image_base_path) / f"{item['player1']}.jpg"
            player2_path = Path(image_base_path) / f"{item['player2']}.jpg"

            # Check if files exist
            if not player1_path.exists():
                print(f"  [WARNING] Player1 image not found: {player1_path}")
                fail_count += 1
                continue

            if not player2_path.exists():
                print(f"  [WARNING] Player2 image not found: {player2_path}")
                fail_count += 1
                continue

            # Load images
            img1 = Image.open(player1_path).convert('RGB')
            img2 = Image.open(player2_path).convert('RGB')

            print(f"  Player1 size: {img1.size}")
            print(f"  Player2 size: {img2.size}")

            # Concatenate
            concatenated = concat_images(player1_path, player2_path, concat_mode)
            print(f"  Concatenated size: {concatenated.size}")

            # Save
            output_filename = f"sample_{i+1}_{item['class']}_{concat_mode}.jpg"
            output_path = output_dir / output_filename
            concatenated.save(output_path, quality=95)
            print(f"  Saved to: {output_path}")

            # Display some stats
            img_array = torch.tensor(list(concatenated.getdata())).float()
            mean_val = img_array.mean().item()
            std_val = img_array.std().item()
            print(f"  Pixel statistics: mean={mean_val:.2f}, std={std_val:.2f}")

            print(f"  [SUCCESS]")
            success_count += 1

        except Exception as e:
            print(f"  [FAILED] Error: {e}")
            fail_count += 1
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "-" * 80)
    print("\n3. Summary:")
    print(f"   Concatenation mode: {concat_mode}")
    print(f"   Total samples tested: {num_samples}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {fail_count}")
    print(f"   Output directory: {output_dir.absolute()}")

    # Test both concatenation modes
    if concat_mode == "horizontal":
        print(f"\n4. Testing vertical concatenation on first sample...")
        if len(metadata) > 0:
            item = metadata[0]
            player1_path = Path(image_base_path) / f"{item['player1']}.jpg"
            player2_path = Path(image_base_path) / f"{item['player2']}.jpg"

            if player1_path.exists() and player2_path.exists():
                concatenated_v = concat_images(player1_path, player2_path, "vertical")
                output_path_v = output_dir / f"sample_1_{item['class']}_vertical.jpg"
                concatenated_v.save(output_path_v, quality=95)
                print(f"   Saved vertical example: {output_path_v}")
                print(f"   Vertical size: {concatenated_v.size}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)

    return success_count, fail_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dual image fusion (simple)")
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
        choices=["horizontal", "vertical", "add", "multiply", "subtract"],
        help="Fusion mode"
    )

    args = parser.parse_args()

    success, fail = test_image_fusion(
        metadata_path=args.metadata,
        image_base_path=args.images,
        num_samples=args.num_samples,
        concat_mode=args.concat_mode
    )

    # Exit with appropriate code
    sys.exit(0 if fail == 0 else 1)
