#!/usr/bin/env python3
"""
Test script for EquiDataset - verify data loading and visualization.
Tests:
1. Dataset loading
2. Data shapes and ranges
3. Multi-view fisheye images
4. Panorama output
5. Visualization of samples
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf


def visualize_sample(views, panorama, sample_idx=0):
    """
    Visualize one sample: 4 fisheye views + GT panorama.

    Args:
        views: [N_views, 3, H, W] - Input fisheye views for one sample
        panorama: [3, H_pano, W_pano] - Ground truth panorama for one sample
        sample_idx: Sample index for title
    """
    n_views = views.shape[0]

    # Create figure: 4 fisheye views + 1 panorama
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Normalize from [-1, 1] to [0, 1] for display
    def normalize_for_display(img):
        return ((img + 1) / 2).clamp(0, 1)

    # Show fisheye views (4 views in 2x2 grid)
    for i_view in range(min(4, n_views)):
        row = i_view // 2
        col = i_view % 2
        ax = axes[row, col]

        view_img = normalize_for_display(views[i_view])
        ax.imshow(view_img.cpu().permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Fisheye View {i_view + 1}", fontsize=12, fontweight='bold')

    # Show panorama (spans 2 columns on the right)
    # Remove the empty subplot at position [0, 2]
    fig.delaxes(axes[0, 2])

    # Create a new subplot spanning 2 rows
    ax_pano = plt.subplot(1, 3, 3)
    pano_img = normalize_for_display(panorama)
    ax_pano.imshow(pano_img.cpu().permute(1, 2, 0))
    ax_pano.axis("off")
    ax_pano.set_title(f"Ground Truth Panorama (2:1)\nShape: {list(panorama.shape)}",
                      fontsize=12, fontweight='bold')

    # Also remove [1, 2] to make room
    fig.delaxes(axes[1, 2])

    plt.suptitle(f"Dataset Sample {sample_idx}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


@hydra.main(version_base=None, config_path="config", config_name="SpiderEye")
def main(cfg: DictConfig):
    """Test dataset loading and visualization."""

    print("=" * 80)
    print("DATASET TESTING SCRIPT")
    print("=" * 80)

    # Import dataset loader
    from ssdd.dataset_equi import load_equirect

    # Configure for testing
    cfg_dataset = OmegaConf.to_container(cfg.dataset, resolve=True)
    cfg_dataset['return_all_views'] = True  # Multi-view mode
    cfg_dataset['batch_size'] = 4  # Small batch for testing
    cfg_dataset['limit'] = 20  # Only load 20 samples for quick test

    print(f"\nConfiguration:")
    print(f"  Dataset root: {cfg_dataset['imagenet_root']}")
    print(f"  Image size: {cfg_dataset['im_size']}")
    print(f"  Batch size: {cfg_dataset['batch_size']}")
    print(f"  Dataset limit: {cfg_dataset['limit']}")
    print(f"  Num workers: {cfg_dataset.get('num_workers', 0)}")
    print(f"  UCM parameters:")
    print(f"    - f_pix: {cfg_dataset.get('f_pix', 'N/A')}")
    print(f"    - xi: {cfg_dataset.get('xi', 'N/A')}")
    print(f"    - mask_mode: {cfg_dataset.get('mask_mode', 'N/A')}")
    print("=" * 80)

    # Step 1: Load dataset
    print("\n[Step 1/5] Loading dataset...")
    try:
        (train_dataset, test_dataset), (train_loader, test_loader) = load_equirect(cfg_dataset)
        print(f"✓ Dataset loaded successfully")
        print(f"  Train dataset: {len(train_dataset)} samples")
        print(f"  Test dataset: {len(test_dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Test data loader
    print("\n[Step 2/5] Testing data loader...")
    try:
        batch = next(iter(train_loader))
        views, panorama = batch
        print(f"✓ Data loader OK")
        print(f"  Batch views shape: {views.shape}")
        print(f"  Batch panorama shape: {panorama.shape}")

        # Verify shapes
        batch_size = views.shape[0]
        n_views = views.shape[1]
        h_view, w_view = views.shape[-2:]
        h_pano, w_pano = panorama.shape[-2:]

        print(f"\n  Detailed shapes:")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Number of views: {n_views}")
        print(f"    - View resolution: {h_view}x{w_view}")
        print(f"    - Panorama resolution: {h_pano}x{w_pano}")
        print(f"    - Panorama aspect ratio: {w_pano/h_pano:.2f}:1 (should be 2:1)")

        # Check aspect ratio
        if abs(w_pano / h_pano - 2.0) < 0.01:
            print(f"  ✓ Panorama has correct 2:1 aspect ratio")
        else:
            print(f"  ✗ WARNING: Panorama aspect ratio is {w_pano/h_pano:.2f}:1, expected 2:1")

    except Exception as e:
        print(f"✗ Data loader failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Check data ranges
    print("\n[Step 3/5] Checking data ranges...")
    try:
        views_min, views_max = views.min().item(), views.max().item()
        pano_min, pano_max = panorama.min().item(), panorama.max().item()

        print(f"  Views range: [{views_min:.3f}, {views_max:.3f}]")
        print(f"  Panorama range: [{pano_min:.3f}, {pano_max:.3f}]")

        # Check if in expected range [-1, 1]
        if -1.1 <= views_min <= -0.9 and 0.9 <= views_max <= 1.1:
            print(f"  ✓ Views in expected range [-1, 1]")
        else:
            print(f"  ✗ WARNING: Views not in expected range [-1, 1]")

        if -1.1 <= pano_min <= -0.9 and 0.9 <= pano_max <= 1.1:
            print(f"  ✓ Panorama in expected range [-1, 1]")
        else:
            print(f"  ✗ WARNING: Panorama not in expected range [-1, 1]")

    except Exception as e:
        print(f"✗ Data range check failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Test multiple batches
    print("\n[Step 4/5] Testing multiple batches...")
    try:
        n_batches = min(3, len(train_loader))
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            views_b, pano_b = batch
            print(f"  Batch {i+1}: views {views_b.shape}, panorama {pano_b.shape}")
        print(f"✓ Successfully loaded {n_batches} batches")
    except Exception as e:
        print(f"✗ Multiple batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Visualize samples
    print("\n[Step 5/5] Creating visualizations...")
    try:
        # Get first batch
        batch = next(iter(train_loader))
        views, panorama = batch

        # Save directory
        save_dir = Path("./test_dataset_output")
        save_dir.mkdir(exist_ok=True)

        # Visualize first 2 samples
        n_visualize = min(2, views.shape[0])
        for i in range(n_visualize):
            fig = visualize_sample(views[i], panorama[i], sample_idx=i)

            save_path = save_dir / f"sample_{i}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved visualization to {save_path}")
            plt.close(fig)

        print(f"\n✓ Visualizations created successfully")

    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Success summary
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nDataset Summary:")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Input: {n_views} fisheye views @ {h_view}x{w_view}")
    print(f"  - Output: 1 panorama @ {h_pano}x{w_pano} (2:1 aspect ratio)")
    print(f"  - Data range: [-1, 1]")
    print(f"\nVisualizations saved to: {save_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
