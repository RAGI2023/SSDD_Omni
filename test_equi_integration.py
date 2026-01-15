#!/usr/bin/env python3
"""
Test script for EquiDataset integration with SSDD.

Usage:
    python test_equi_integration.py
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_equi_dataset():
    """Test EquiDataset standalone."""
    print("=" * 60)
    print("Test 1: EquiDataset Standalone")
    print("=" * 60)

    from utils.EquiDataset import EquiDataset, DEFAULT_JITTER_CONFIG

    # Check if data directory exists
    data_dir = "/data/360SP-data/train"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please update the path or create test data.")
        return False

    try:
        dataset = EquiDataset(
            folder_path=data_dir,
            canvas_size=(1024, 512),
            out_w=256,
            out_h=256,
            jitter_cfg=DEFAULT_JITTER_CONFIG,
            f_pix=220.0,
            xi=0.9,
            mask_mode="inscribed",
        )

        print(f"✓ Dataset created successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Views: {[name for name, _ in dataset.VIEWS]}")

        # Test __getitem__
        if len(dataset) > 0:
            imgs, img_original = dataset[0]
            print(f"✓ Sample loaded successfully")
            print(f"  - Fisheye views shape: {imgs.shape}")  # [4, 3, H, W]
            print(f"  - Original panorama shape: {img_original.shape}")  # [3, Hc, Wc]
            print(f"  - Value range: [{imgs.min():.3f}, {imgs.max():.3f}]")
        else:
            print("❌ No samples found in dataset")
            return False

        print("✓ Test 1 PASSED\n")
        return True

    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_equi_wrapper():
    """Test EquiDatasetWrapper (SSDD integration)."""
    print("=" * 60)
    print("Test 2: EquiDatasetWrapper (SSDD Integration)")
    print("=" * 60)

    # Mock TaskState for testing
    class MockAccelerator:
        num_processes = 1

    class MockTaskState:
        def __init__(self):
            self.accelerator = MockAccelerator()

        @classmethod
        def __call__(cls):
            return cls()

    # Monkey patch TaskState
    import ssdd.mutils.main_utils
    original_taskstate = ssdd.mutils.main_utils.TaskState
    ssdd.mutils.main_utils.TaskState = MockTaskState

    try:
        from ssdd.dataset_equi import EquiDatasetWrapper, load_equirect

        data_root = "/data/360SP-data"
        if not os.path.exists(data_root):
            print(f"❌ Data root not found: {data_root}")
            return False

        # Test EquiDatasetWrapper
        dataset = EquiDatasetWrapper(
            root=data_root,
            split="train",
            im_size=128,
            f_pix=220.0,
            xi=0.9,
        )

        print(f"✓ EquiDatasetWrapper created successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Split: {dataset.split}")
        print(f"  - Image size: {dataset.im_size}")

        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"✓ Sample loaded successfully")
            print(f"  - Image shape: {img.shape}")  # [3, H, W]
            print(f"  - Label: {label}")
            print(f"  - Value range: [{img.min():.3f}, {img.max():.3f}]")
        else:
            print("❌ No samples found in dataset")
            return False

        # Test load_equirect function
        print("\nTesting load_equirect function...")
        ds_cfg = {
            'imagenet_root': data_root,
            'im_size': 128,
            'batch_size': 4,
            'limit': 10,  # Limit to 10 samples for quick test
            'f_pix': 220.0,
            'xi': 0.9,
        }

        (train_ds, test_ds), (train_loader, test_loader) = load_equirect(ds_cfg)

        print(f"✓ load_equirect succeeded")
        print(f"  - Train dataset: {len(train_ds)} samples")
        print(f"  - Test dataset: {len(test_ds)} samples")

        # Test data loading
        print("\nTesting DataLoader...")
        for i, (imgs, labels) in enumerate(train_loader):
            print(f"✓ Batch {i}: shape={imgs.shape}, labels={labels.shape}")
            if i >= 1:  # Test 2 batches
                break

        print("\n✓ Test 2 PASSED\n")
        return True

    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore original TaskState
        ssdd.mutils.main_utils.TaskState = original_taskstate


def test_comparison():
    """Compare ImageFolder vs EquiDataset interface."""
    print("=" * 60)
    print("Test 3: Interface Comparison")
    print("=" * 60)

    print("\nImageFolder interface:")
    print("  Input:  root/train/class1/img1.jpg")
    print("  Output: (image, label)")
    print("         - image: [3, H, W] in [-1, 1]")
    print("         - label: int (class index)")

    print("\nEquiDataset interface:")
    print("  Input:  root/train/pano1.jpg")
    print("  Output: (fisheye_views, panorama)")
    print("         - fisheye_views: [4, 3, H, W] in [0, 1]")
    print("         - panorama: [3, Hc, Wc] in [0, 1]")

    print("\nEquiDatasetWrapper interface (SSDD compatible):")
    print("  Input:  root/train/pano1.jpg")
    print("  Output: (image, label)")
    print("         - image: [3, H, W] in [-1, 1] (first view)")
    print("         - label: 0 (dummy)")

    print("\n✓ Test 3 PASSED (informational)\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EquiDataset Integration Test Suite")
    print("=" * 60 + "\n")

    results = []

    # Test 1: EquiDataset standalone
    results.append(("EquiDataset Standalone", test_equi_dataset()))

    # Test 2: EquiDatasetWrapper
    results.append(("EquiDatasetWrapper", test_equi_wrapper()))

    # Test 3: Interface comparison (informational)
    results.append(("Interface Comparison", test_comparison()))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("❌ Some tests FAILED. Please check the output above.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
