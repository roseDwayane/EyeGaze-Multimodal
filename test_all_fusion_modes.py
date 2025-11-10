"""
Quick test script to verify all fusion modes work correctly
"""

import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

from Data.processed.test_fusion_simple import test_image_fusion

print("=" * 80)
print("TESTING ALL FUSION MODES")
print("=" * 80)

fusion_modes = ["horizontal", "vertical", "add", "multiply", "subtract"]

for mode in fusion_modes:
    print(f"\n{'='*80}")
    print(f"Testing: {mode.upper()}")
    print(f"{'='*80}")

    try:
        success, fail = test_image_fusion(
            metadata_path="Data/metadata/complete_metadata.json",
            image_base_path="Data/raw/Gaze/example",
            num_samples=2,  # Test with 2 samples for speed
            concat_mode=mode
        )

        if fail == 0:
            print(f"[OK] {mode}: PASSED ({success} samples)")
        else:
            print(f"[FAIL] {mode}: FAILED ({fail} failures)")

    except Exception as e:
        print(f"[ERROR] {mode}: ERROR - {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("ALL TESTS COMPLETED")
print(f"{'='*80}")
print("\nCheck output images in: Data/processed/test_outputs/")
