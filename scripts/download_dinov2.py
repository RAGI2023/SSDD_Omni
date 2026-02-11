#!/usr/bin/env python3
"""
Pre-download DINOv2 models to avoid waiting during training.

Usage:
    python scripts/download_dinov2.py [model_name]

    model_name: vits14, vitb14, vitl14, vitg14 (default: vitb14)
"""

import sys
import torch


def download_dinov2(model_name='vitb14', use_registers=False):
    """Download a specific DINOv2 model."""

    full_model_name = f'dinov2_{model_name}'
    if use_registers:
        full_model_name += '_reg'

    print(f"=" * 80)
    print(f"Downloading DINOv2 model: {full_model_name}")
    print(f"=" * 80)
    print()

    try:
        print(f"Loading {full_model_name} from torch.hub...")
        model = torch.hub.load(
            'facebookresearch/dinov2',
            full_model_name,
            trust_repo=True,
            skip_validation=True
        )

        print()
        print(f"✓ Successfully downloaded {full_model_name}")
        print(f"  - Embedding dimension: {model.embed_dim}")
        print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test with dummy input
        print()
        print("Testing model with dummy input...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            model.eval()
            output = model.forward_features(dummy_input)

        print(f"✓ Model works correctly!")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output patches shape: {output['x_norm_patchtokens'].shape}")

        return True

    except Exception as e:
        print(f"✗ Failed to download {full_model_name}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_all():
    """Download all common DINOv2 models."""
    models = ['vits14', 'vitb14', 'vitl14', 'vitg14']

    print("=" * 80)
    print("Downloading all DINOv2 models")
    print("=" * 80)
    print()

    results = {}
    for model_name in models:
        success = download_dinov2(model_name)
        results[model_name] = success
        print()

    print("=" * 80)
    print("Download Summary")
    print("=" * 80)
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} dinov2_{model_name}")
    print()


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            download_all()
        else:
            model_name = sys.argv[1]
            if not model_name.startswith('vit'):
                model_name = f'vit{model_name}'

            use_registers = '--reg' in sys.argv
            download_dinov2(model_name, use_registers)
    else:
        # Default: download ViT-B (most commonly used)
        download_dinov2('vitb14')

        print()
        print("=" * 80)
        print("To download other models:")
        print("  python scripts/download_dinov2.py vits14  # ViT-Small")
        print("  python scripts/download_dinov2.py vitl14  # ViT-Large")
        print("  python scripts/download_dinov2.py vitg14  # ViT-Giant")
        print("  python scripts/download_dinov2.py --all   # Download all")
        print("=" * 80)


if __name__ == "__main__":
    main()
