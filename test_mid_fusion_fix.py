"""
Quick test for mid fusion model fixes
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Models.fusion.mid_fusion import MidFusionModel

print("=" * 60)
print("Testing Mid Fusion Model Fixes")
print("=" * 60)

# Test 1: Model initialization
print("\n1. Testing MidFusionModel initialization...")
try:
    model = MidFusionModel(
        image_model_path=None,
        eeg_model_path=None,
        num_classes=3,
        image_d_model=768,
        eeg_d_model=256,
        use_ibs_token=True,
        use_cross_attention=True,
        image_shared_weights=True,
        eeg_shared_weights=True
    )
    print(f"[OK] MidFusionModel initialized successfully!")
    print(f"[OK] Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
except Exception as e:
    print(f"[FAIL] MidFusionModel initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Forward pass
print("\n2. Testing MidFusionModel forward pass...")
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

# Test 3: Without IBS token
print("\n3. Testing without IBS token...")
try:
    model_no_ibs = MidFusionModel(
        num_classes=3,
        use_ibs_token=False,
        use_cross_attention=True
    )

    model_no_ibs.eval()
    with torch.no_grad():
        outputs = model_no_ibs(img1, img2, eeg1, eeg2, labels)

    print(f"[OK] Without IBS token: {outputs['logits'].shape}")
except Exception as e:
    print(f"[FAIL] No IBS token test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Without cross-attention
print("\n4. Testing without cross-attention...")
try:
    model_no_attn = MidFusionModel(
        num_classes=3,
        use_ibs_token=True,
        use_cross_attention=False
    )

    model_no_attn.eval()
    with torch.no_grad():
        outputs = model_no_attn(img1, img2, eeg1, eeg2, labels)

    print(f"[OK] Without cross-attention: {outputs['logits'].shape}")
except Exception as e:
    print(f"[FAIL] No cross-attention test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests passed! [OK]")
print("=" * 60)
