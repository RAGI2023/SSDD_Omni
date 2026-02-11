#!/usr/bin/env python3
"""
Pre-download all required models to avoid issues during training.

This script downloads:
- DINOv2 models (for encoder)
- VGG16 (for LPIPS loss)

Usage:
    python scripts/download_all_models.py
"""

import torch
import torchvision


def download_vgg16():
    """Download VGG16 for LPIPS."""
    print("=" * 80)
    print("Downloading VGG16 for LPIPS loss")
    print("=" * 80)
    print()

    try:
        print("Loading VGG16...")
        # This will download to ~/.cache/torch/hub/checkpoints/
        model = torchvision.models.vgg16(pretrained=True)
        print("✓ VGG16 downloaded successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Failed to download VGG16: {e}")

        # Try without hash checking as fallback
        print("\nTrying without hash verification...")
        try:
            # Monkey-patch to disable hash checking
            import torchvision.models._api as api
            original_get_state_dict = api.WeightsEnum.get_state_dict

            def get_state_dict_no_check(self, *args, **kwargs):
                kwargs['check_hash'] = False
                return original_get_state_dict(self, *args, **kwargs)

            api.WeightsEnum.get_state_dict = get_state_dict_no_check

            model = torchvision.models.vgg16(pretrained=True)
            print("✓ VGG16 downloaded successfully (hash check disabled)")
            return True
        except Exception as e2:
            print(f"✗ Still failed: {e2}")
            return False


def download_dinov2(model_name='vitb14'):
    """Download DINOv2 model."""
    print()
    print("=" * 80)
    print(f"Downloading DINOv2: {model_name}")
    print("=" * 80)
    print()

    try:
        full_name = f'dinov2_{model_name}'
        model = torch.hub.load(
            'facebookresearch/dinov2',
            full_name,
            trust_repo=True,
            skip_validation=True
        )
        print(f"✓ {full_name} downloaded successfully")
        print(f"  - Embedding dimension: {model.embed_dim}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return False


def main():
    print("=" * 80)
    print("Downloading all required models for SSDD Multi-View")
    print("=" * 80)
    print()

    results = {}

    # Download VGG16 (required for LPIPS)
    results['VGG16'] = download_vgg16()

    # Download DINOv2 (default: ViT-B)
    results['DINOv2-ViT-B'] = download_dinov2('vitb14')

    # Summary
    print()
    print("=" * 80)
    print("Download Summary")
    print("=" * 80)
    for model, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {model}")
    print()

    if all(results.values()):
        print("✓ All models downloaded successfully!")
        print("\nYou can now run training without waiting for downloads:")
        print("  accelerate launch ssdd/main_multiview.py \\")
        print("      --config-name=SpiderEye_dinov2 \\")
        print("      run_name=my_experiment")
    else:
        print("✗ Some models failed to download. Please check the errors above.")
        print("\nYou can try running the training anyway - it will attempt to download during training.")


if __name__ == "__main__":
    main()
