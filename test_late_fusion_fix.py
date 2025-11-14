"""
Quick test for late fusion model fixes
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Models.fusion.late_fusion import LateFusionModel
from Models.backbones.vit import ViTClassifier

print("=" * 60)
print("Testing Late Fusion Model Fixes")
print("=" * 60)

# Test 1: Check if ViTClassifier has get_cls_features method
print("\n1. Testing ViTClassifier.get_cls_features()...")
vit = ViTClassifier(num_labels=3, image_size=224, pretrained=False)
dummy_img = torch.randn(2, 3, 224, 224)

try:
    cls_features = vit.get_cls_features(pixel_values=dummy_img)
    print(f"[OK] CLS features shape: {cls_features.shape}")
    assert cls_features.shape == (2, 768), f"Expected (2, 768), got {cls_features.shape}"
    print("[OK] CLS feature extraction works!")
except Exception as e:
    print(f"[FAIL] CLS feature extraction failed: {e}")
    sys.exit(1)

# Test 2: Check LateFusionModel initialization
print("\n2. Testing LateFusionModel initialization...")
try:
    model = LateFusionModel(
        image_model_path=None,
        eeg_model_path=None,
        fusion_mode='features',
        num_classes=3,
        image_d_model=768,
        eeg_d_model=256
    )
    print("[OK] LateFusionModel initialized successfully!")
    print(f"[OK] Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
except Exception as e:
    print(f"[FAIL] LateFusionModel initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check forward pass
print("\n3. Testing LateFusionModel forward pass...")
try:
    img1 = torch.randn(2, 3, 224, 224)
    img2 = torch.randn(2, 3, 224, 224)
    eeg1 = torch.randn(2, 32, 1024)
    eeg2 = torch.randn(2, 32, 1024)
    labels = torch.randint(0, 3, (2,))

    model.eval()
    with torch.no_grad():
        outputs = model(img1, img2, eeg1, eeg2, labels)

    print(f"[OK] Output keys: {outputs.keys()}")
    print(f"[OK] Logits shape: {outputs['logits'].shape}")
    print(f"[OK] Loss: {outputs['loss'].item():.4f}")

    assert outputs['logits'].shape == (2, 3), f"Expected (2, 3), got {outputs['logits'].shape}"
    print("[OK] Forward pass successful!")
except Exception as e:
    print(f"[FAIL] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! [OK]")
print("=" * 60)
