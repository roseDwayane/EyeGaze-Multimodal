"""
Verification script to check if the setup is correct before training
"""

import os
import sys
import json
import yaml
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")

    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'PIL',
        'yaml',
        'sklearn',
        'numpy'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("All dependencies installed!\n")
    return True


def check_config(config_path):
    """Check if config file is valid"""
    print(f"Checking config file: {config_path}")

    if not os.path.exists(config_path):
        print(f"  [FAIL] Config file not found: {config_path}")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        required_keys = ['model', 'data', 'training', 'paths']
        for key in required_keys:
            if key in config:
                print(f"  [OK] {key} section found")
            else:
                print(f"  [FAIL] {key} section missing")
                return False

        print("Config file is valid!\n")
        return config

    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False


def check_metadata(metadata_path):
    """Check if metadata file exists and is valid"""
    print(f"Checking metadata file: {metadata_path}")

    if not os.path.exists(metadata_path):
        print(f"  [FAIL] Metadata file not found: {metadata_path}")
        return False

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"  [OK] Metadata loaded successfully")
        print(f"  [OK] Total samples: {len(metadata)}")

        # Check class distribution
        class_counts = {}
        for item in metadata:
            class_name = item.get('class', 'Unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"  [OK] Class distribution:")
        for class_name, count in class_counts.items():
            print(f"      - {class_name}: {count}")

        # Check required fields
        required_fields = ['player1', 'player2', 'class']
        sample = metadata[0]

        for field in required_fields:
            if field in sample:
                print(f"  [OK] Field '{field}' present")
            else:
                print(f"  [FAIL] Field '{field}' missing")
                return False

        print("Metadata file is valid!\n")
        return metadata

    except Exception as e:
        print(f"  ✗ Error loading metadata: {e}")
        return False


def check_images(metadata, image_base_path, max_check=10):
    """Check if image files exist"""
    print(f"Checking image files in: {image_base_path}")

    if not os.path.exists(image_base_path):
        print(f"  [FAIL] Image base path not found: {image_base_path}")
        return False

    missing_images = []
    valid_images = 0

    # Check a sample of images
    samples_to_check = min(max_check, len(metadata))

    for i in range(samples_to_check):
        item = metadata[i]
        player1_path = os.path.join(image_base_path, f"{item['player1']}.jpg")
        player2_path = os.path.join(image_base_path, f"{item['player2']}.jpg")

        # Check player1
        if os.path.exists(player1_path):
            try:
                img = Image.open(player1_path)
                img.verify()
                valid_images += 1
            except Exception as e:
                print(f"  [FAIL] Invalid image: {player1_path} ({e})")
                missing_images.append(player1_path)
        else:
            print(f"  [FAIL] Missing: {player1_path}")
            missing_images.append(player1_path)

        # Check player2
        if os.path.exists(player2_path):
            try:
                img = Image.open(player2_path)
                img.verify()
                valid_images += 1
            except Exception as e:
                print(f"  [FAIL] Invalid image: {player2_path} ({e})")
                missing_images.append(player2_path)
        else:
            print(f"  [FAIL] Missing: {player2_path}")
            missing_images.append(player2_path)

    print(f"  [OK] Valid images checked: {valid_images}/{samples_to_check * 2}")

    if missing_images:
        print(f"  [FAIL] Found {len(missing_images)} missing/invalid images")
        return False

    print("Image files are accessible!\n")
    return True


def check_model_loading():
    """Check if ViT model can be loaded"""
    print("Checking model loading...")

    try:
        from transformers import ViTForImageClassification

        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=3,
            ignore_mismatched_sizes=True
        )

        print(f"  [OK] Model loaded successfully")
        print(f"  [OK] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Model loading successful!\n")

        return True

    except Exception as e:
        print(f"  [FAIL] Error loading model: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 80)
    print("VERIFICATION SCRIPT - ViT Training Setup")
    print("=" * 80 + "\n")

    # Default paths
    config_path = "Experiments/configs/vit_single_vs_competition.yaml"

    # Check dependencies
    if not check_dependencies():
        print("\n[FAILED] Dependency check failed!")
        return False

    # Check config
    config = check_config(config_path)
    if not config:
        print("\n[FAILED] Config check failed!")
        return False

    # Check metadata
    metadata_path = config['data']['metadata_path']
    metadata = check_metadata(metadata_path)
    if not metadata:
        print("\n[FAILED] Metadata check failed!")
        return False

    # Check images
    image_base_path = config['data']['image_base_path']
    if not check_images(metadata, image_base_path):
        print("\n[FAILED] Image check failed!")
        return False

    # Check model loading
    if not check_model_loading():
        print("\n[FAILED] Model loading check failed!")
        return False

    print("=" * 80)
    print("[SUCCESS] ALL CHECKS PASSED!")
    print("=" * 80)
    print("\nYou can now run training with:")
    print(f"  python Experiments/scripts/train_vit.py --config {config_path}")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
