#!/usr/bin/env python3
"""
Quick test script for DINOv2 encoder integration.

Usage:
    python scripts/test_dinov2_encoder.py
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ssdd.models.dinov2_encoder import DinoV2Encoder
from ssdd.models.ssdd.ssdd_multiview import SSDDMultiView


def test_dinov2_encoder():
    """Test DinoV2Encoder directly."""
    print("=" * 80)
    print("Testing DinoV2Encoder")
    print("=" * 80)

    # Create encoder
    print("\n1. Creating DinoV2Encoder (ViT-B, z_dim=4)...")
    encoder = DinoV2Encoder(
        z_dim=4,
        model_name='dinov2_vitb14',
        patch_size=14,
        freeze_backbone=True
    )
    print(f"   Encoder created: {encoder.__class__.__name__}")
    print(f"   DINOv2 embed_dim: {encoder.embed_dim}")
    print(f"   Output z_dim: {encoder.z_dim}")
    print(f"   Patch size: {encoder.patch_size}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    img_size = 224
    x = torch.randn(batch_size, 3, img_size, img_size)
    print(f"   Input shape: {x.shape}")

    with torch.no_grad():
        encoder.eval()
        output = encoder(x)

    print(f"   Output type: {type(output).__name__}")
    print(f"   Output params shape: {output.parameters.shape}")
    print(f"   Output mode shape: {output.mode().shape}")
    print(f"   Output sample shape: {output.sample().shape}")

    # Verify output shape
    expected_h = img_size // encoder.patch_size
    expected_w = img_size // encoder.patch_size
    expected_shape = (batch_size, encoder.z_dim, expected_h, expected_w)

    assert output.mode().shape == expected_shape, \
        f"Expected {expected_shape}, got {output.mode().shape}"

    print("   ✓ Forward pass successful!")

    # Check if backbone is frozen
    print("\n3. Checking if DINOv2 backbone is frozen...")
    frozen_params = sum(1 for p in encoder.dinov2.parameters() if not p.requires_grad)
    total_params = sum(1 for p in encoder.dinov2.parameters())
    print(f"   Frozen params: {frozen_params}/{total_params}")
    assert frozen_params == total_params, "Backbone should be frozen!"
    print("   ✓ Backbone is properly frozen!")

    return encoder


def test_ssdd_multiview_with_dinov2():
    """Test SSDDMultiView with DinoV2Encoder."""
    print("\n" + "=" * 80)
    print("Testing SSDDMultiView with DinoV2Encoder")
    print("=" * 80)

    print("\n1. Creating SSDDMultiView with dinov2_vitb14_p14_c4...")
    try:
        model = SSDDMultiView(
            encoder='dinov2_vitb14_p14_c4',
            encoder_checkpoint=None,
            encoder_train=False,
            decoder='S',  # Small decoder for testing
            decoder_image_size=[448, 224],  # 2:1 panorama
            n_views=4,
            fusion_type='concat_conv',
            use_view_encoding=True,
            view_encoding_type='sinusoidal',
            fm_sampler={'steps': 12},
        )
        print(f"   Model created: {model.__class__.__name__}")
        print(f"   Encoder type: {type(model.encoder).__name__}")
        print(f"   Encoder z_dim: {model.encoder.z_dim}")
        print(f"   Encoder patch_size: {model.encoder.patch_size}")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Test encoding views
    print("\n2. Testing multi-view encoding...")
    batch_size = 2
    n_views = 4
    img_size = 224
    views = torch.randn(batch_size, n_views, 3, img_size, img_size)
    print(f"   Input views shape: {views.shape}")

    try:
        with torch.no_grad():
            model.eval()
            z_views, encoded_dists = model.encode_views(views)

        print(f"   Encoded z_views shape: {z_views.shape}")
        print(f"   Number of distributions: {len(encoded_dists)}")
        print("   ✓ Multi-view encoding successful!")
    except Exception as e:
        print(f"   ✗ Failed to encode views: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Test fusion
    print("\n3. Testing fusion module...")
    try:
        with torch.no_grad():
            z_fused = model.fusion(z_views)
        print(f"   Fused latent shape: {z_fused.shape}")
        print("   ✓ Fusion successful!")
    except Exception as e:
        print(f"   ✗ Failed to fuse: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

    return model


def main():
    print("DINOv2 Encoder Integration Test\n")

    # Test encoder directly
    encoder = test_dinov2_encoder()

    # Test in SSDDMultiView
    model = test_ssdd_multiview_with_dinov2()

    if model is not None:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("✓ DinoV2Encoder works correctly")
        print("✓ SSDDMultiView integration successful")
        print("✓ Multi-view encoding pipeline functional")
        print("\nYou can now use 'dinov2_vitb14_p14_c4' as encoder config!")
        print("\nExample training command:")
        print("  accelerate launch ssdd/main_multiview.py \\")
        print("      --config-name=SpiderEye_dinov2 \\")
        print("      run_name=my_dinov2_experiment")
    else:
        print("\n✗ Tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
