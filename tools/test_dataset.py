#!/usr/bin/env python3
"""
Dataset Testing Tool

Test EquiDataset (image) and EquiVideoDataset (video) with configurable UCM jitter parameters.
Visualizes fisheye views and panorama samples to verify data augmentation effects.

Usage:
    # Test using config file
    python tools/test_dataset.py --config config/SpiderEye.yaml

    # Test video dataset using config
    python tools/test_dataset.py --config config/SpiderVideo.yaml

    # Override config parameters via command line
    python tools/test_dataset.py --config config/SpiderEye.yaml --xi_jitter 0.2 --f_jitter 0.15

    # Show multiple samples from same image to see jitter effect
    python tools/test_dataset.py --config config/SpiderEye.yaml --repeat 5 --sample_idx 0

    # Direct mode without config (legacy)
    python tools/test_dataset.py --data_path /data/360SP-data --type image
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "utils"))

from utils.EquiDataset import EquiDataset, EquiVideoDataset, DEFAULT_JITTER_CONFIG, NO_JITTER_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test EquiDataset with UCM jitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file
  python tools/test_dataset.py --config config/SpiderEye.yaml

  # Override jitter parameters
  python tools/test_dataset.py --config config/SpiderEye.yaml --xi_jitter 0.2

  # Show jitter effect (repeat same sample)
  python tools/test_dataset.py --config config/SpiderEye.yaml --repeat 5 --sample_idx 0
        """
    )

    # Config file
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config file (SpiderEye.yaml or SpiderVideo.yaml)")

    # Dataset path (can override config or use directly)
    parser.add_argument("--data_path", type=str, default=None,
                        help="Root path to dataset (overrides config.dataset.imagenet_root or video_root)")
    parser.add_argument("--type", type=str, choices=["image", "video"], default=None,
                        help="Dataset type: 'image' or 'video' (auto-detect from config if not specified)")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train",
                        help="Dataset split to test")

    # UCM parameters (override config)
    parser.add_argument("--im_size", type=int, default=None,
                        help="Output image size")
    parser.add_argument("--f_pix", type=float, default=None,
                        help="Focal length in pixels")
    parser.add_argument("--xi", type=float, default=None,
                        help="UCM xi parameter")
    parser.add_argument("--mask_mode", type=str, default=None,
                        choices=["inscribed", "diagonal", "none"],
                        help="Fisheye mask mode")

    # Jitter parameters (override config)
    parser.add_argument("--xi_jitter", type=float, default=None,
                        help="Xi jitter range: xi * (1 ± xi_jitter)")
    parser.add_argument("--f_jitter", type=float, default=None,
                        help="f_pix jitter range: f_pix / (1 ± f_jitter)")
    parser.add_argument("--jitter_target", type=str, default=None,
                        choices=["xi", "f", "both"],
                        help="Which parameter(s) to jitter")
    parser.add_argument("--no_jitter", action="store_true",
                        help="Disable all jitter (ignore config)")

    # Video-specific parameters (override config)
    parser.add_argument("--frame_interval", type=int, default=None,
                        help="[Video] Extract one frame every N frames")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="[Video] Max frames per video")

    # Visualization options
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Specific sample index to visualize (random if None)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat same sample N times to show jitter variation")
    parser.add_argument("--save_path", type=str, default='runs/dataset_test.png',
                        help="Save visualization to this path (show if None)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def load_config(config_path):
    """Load and return OmegaConf config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def merge_args_with_config(args, cfg):
    """
    Merge command line args with config.
    Command line args take precedence over config values.

    Returns a SimpleNamespace with all parameters.
    """
    ds_cfg = cfg.get("dataset", {})

    # Determine dataset type
    if args.type is not None:
        dataset_type = args.type
    else:
        dataset_type = ds_cfg.get("type", "image")

    # Determine data path
    if args.data_path is not None:
        data_path = args.data_path
    elif dataset_type == "video":
        data_path = ds_cfg.get("video_root", ds_cfg.get("imagenet_root"))
    else:
        data_path = ds_cfg.get("imagenet_root")

    if data_path is None:
        raise ValueError("data_path must be specified either via --data_path or in config file")

    # Build merged config
    class Config:
        pass

    merged = Config()

    # Dataset basics
    merged.data_path = data_path
    merged.type = dataset_type
    merged.split = args.split

    # UCM parameters (command line > config > defaults)
    merged.im_size = args.im_size if args.im_size is not None else ds_cfg.get("im_size", 128)
    merged.f_pix = args.f_pix if args.f_pix is not None else ds_cfg.get("f_pix", 220.0)
    merged.xi = args.xi if args.xi is not None else ds_cfg.get("xi", 0.9)
    merged.mask_mode = args.mask_mode if args.mask_mode is not None else ds_cfg.get("mask_mode", "inscribed")

    # Jitter parameters
    merged.no_jitter = args.no_jitter
    merged.xi_jitter = args.xi_jitter if args.xi_jitter is not None else ds_cfg.get("xi_jitter", 0.0)
    merged.f_jitter = args.f_jitter if args.f_jitter is not None else ds_cfg.get("f_jitter", 0.0)
    merged.jitter_target = args.jitter_target if args.jitter_target is not None else ds_cfg.get("jitter_target", "both")

    # Video parameters
    merged.frame_interval = args.frame_interval if args.frame_interval is not None else ds_cfg.get("frame_interval", 30)
    merged.max_frames = args.max_frames if args.max_frames is not None else ds_cfg.get("max_frames_per_video", None)

    # Visualization options (always from command line)
    merged.n_samples = args.n_samples
    merged.sample_idx = args.sample_idx
    merged.repeat = args.repeat
    merged.save_path = args.save_path
    merged.seed = args.seed

    return merged


def get_default_args(args):
    """Get default values when no config is provided."""
    class Config:
        pass

    merged = Config()

    if args.data_path is None:
        raise ValueError("Either --config or --data_path must be specified")

    merged.data_path = args.data_path
    merged.type = args.type if args.type is not None else "image"
    merged.split = args.split

    merged.im_size = args.im_size if args.im_size is not None else 128
    merged.f_pix = args.f_pix if args.f_pix is not None else 220.0
    merged.xi = args.xi if args.xi is not None else 0.9
    merged.mask_mode = args.mask_mode if args.mask_mode is not None else "inscribed"

    merged.no_jitter = args.no_jitter
    merged.xi_jitter = args.xi_jitter if args.xi_jitter is not None else 0.1
    merged.f_jitter = args.f_jitter if args.f_jitter is not None else 0.1
    merged.jitter_target = args.jitter_target if args.jitter_target is not None else "both"

    merged.frame_interval = args.frame_interval if args.frame_interval is not None else 30
    merged.max_frames = args.max_frames

    merged.n_samples = args.n_samples
    merged.sample_idx = args.sample_idx
    merged.repeat = args.repeat
    merged.save_path = args.save_path
    merged.seed = args.seed

    return merged


def create_dataset(cfg):
    """Create dataset based on merged config."""
    folder_path = os.path.join(cfg.data_path, cfg.split)

    if not os.path.exists(folder_path):
        raise ValueError(f"Dataset path does not exist: {folder_path}")

    # Jitter config for rotation/lighting
    if cfg.no_jitter:
        jitter_cfg = NO_JITTER_CONFIG
    else:
        jitter_cfg = DEFAULT_JITTER_CONFIG if cfg.split == "train" else NO_JITTER_CONFIG

    # Common UCM parameters
    ucm_params = {
        "canvas_size": (cfg.im_size * 2, cfg.im_size),
        "out_w": cfg.im_size,
        "out_h": cfg.im_size,
        "f_pix": cfg.f_pix,
        "xi": cfg.xi,
        "mask_mode": cfg.mask_mode,
        "jitter_cfg": jitter_cfg,
        "xi_jitter": cfg.xi_jitter if not cfg.no_jitter else 0.0,
        "f_jitter": cfg.f_jitter if not cfg.no_jitter else 0.0,
        "jitter_target": cfg.jitter_target,
    }

    if cfg.type == "image":
        dataset = EquiDataset(folder_path, **ucm_params)
        print(f"Created EquiDataset with {len(dataset)} images")
    else:
        ucm_params["frame_interval"] = cfg.frame_interval
        ucm_params["max_frames_per_video"] = cfg.max_frames
        dataset = EquiVideoDataset(folder_path, **ucm_params)
        print(f"Created EquiVideoDataset with {len(dataset)} frames")

    return dataset


def visualize_sample(views, panorama, title="Sample", ax_row=None):
    """
    Visualize a single sample (4 views + panorama).

    Args:
        views: [4, 3, H, W] tensor in [0, 1]
        panorama: [3, H_pano, W_pano] tensor in [0, 1]
        title: Title for this sample
        ax_row: List of axes to plot on (if None, create new figure)
    """
    n_views = views.shape[0]

    if ax_row is None:
        fig, ax_row = plt.subplots(1, n_views + 1, figsize=(3 * (n_views + 2), 3))

    view_names = ["Front", "Right", "Back", "Left"]

    # Plot views
    for i in range(n_views):
        view_img = views[i].permute(1, 2, 0).numpy()
        ax_row[i].imshow(view_img)
        ax_row[i].axis("off")
        ax_row[i].set_title(view_names[i] if i < len(view_names) else f"View {i}")

    # Plot panorama
    pano_img = panorama.permute(1, 2, 0).numpy()
    ax_row[n_views].imshow(pano_img)
    ax_row[n_views].axis("off")
    ax_row[n_views].set_title("Panorama")

    return ax_row


def visualize_jitter_effect(dataset, sample_idx, n_repeats, cfg):
    """
    Visualize the same sample multiple times to show jitter effect.
    """
    n_views = 4
    fig, axes = plt.subplots(n_repeats, n_views + 1, figsize=(3 * (n_views + 2), 3 * n_repeats))

    if n_repeats == 1:
        axes = axes.reshape(1, -1)

    print(f"\nSampling index {sample_idx} repeated {n_repeats} times:")

    for i in range(n_repeats):
        views, panorama = dataset[sample_idx]

        # Get current UCM params (they change each call due to jitter)
        xi, f_pix = dataset._sample_ucm_params()
        print(f"  Repeat {i+1}: xi={xi:.4f}, f_pix={f_pix:.2f}" if f_pix else f"  Repeat {i+1}: xi={xi:.4f}")

        visualize_sample(views, panorama, title=f"Repeat {i+1}", ax_row=axes[i])
        axes[i][0].set_ylabel(f"xi={xi:.3f}", fontsize=10)

    plt.suptitle(f"Jitter Effect: xi_jitter={cfg.xi_jitter}, f_jitter={cfg.f_jitter}, target={cfg.jitter_target}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def visualize_samples(dataset, sample_indices, cfg):
    """
    Visualize multiple different samples.
    """
    n_samples = len(sample_indices)
    n_views = 4
    fig, axes = plt.subplots(n_samples, n_views + 1, figsize=(3 * (n_views + 2), 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    print(f"\nVisualizing {n_samples} samples:")

    for i, idx in enumerate(sample_indices):
        views, panorama = dataset[idx]

        # Get current UCM params
        xi, f_pix = dataset._sample_ucm_params()
        print(f"  Sample {idx}: xi={xi:.4f}, f_pix={f_pix:.2f}" if f_pix else f"  Sample {idx}: xi={xi:.4f}")

        visualize_sample(views, panorama, title=f"Sample {idx}", ax_row=axes[i])
        axes[i][0].set_ylabel(f"idx={idx}", fontsize=10)

    jitter_status = "OFF" if cfg.no_jitter else f"xi={cfg.xi_jitter}, f={cfg.f_jitter}"
    plt.suptitle(f"Dataset: {cfg.type} | Split: {cfg.split} | Jitter: {jitter_status}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def print_dataset_info(dataset, cfg, config_path=None):
    """Print dataset information."""
    print("\n" + "=" * 60)
    print("Dataset Configuration")
    print("=" * 60)
    if config_path:
        print(f"  Config:       {config_path}")
    print(f"  Type:         {cfg.type}")
    print(f"  Path:         {cfg.data_path}/{cfg.split}")
    print(f"  Samples:      {len(dataset)}")
    print(f"  Image size:   {cfg.im_size}x{cfg.im_size}")
    print(f"  Panorama:     {cfg.im_size * 2}x{cfg.im_size}")
    print()
    print("UCM Parameters:")
    print(f"  f_pix:        {cfg.f_pix}")
    print(f"  xi:           {cfg.xi}")
    print(f"  mask_mode:    {cfg.mask_mode}")
    print()
    print("Jitter Configuration:")
    if cfg.no_jitter:
        print("  Jitter:       DISABLED")
    else:
        print(f"  xi_jitter:    {cfg.xi_jitter} (range: [{cfg.xi * (1 - cfg.xi_jitter):.3f}, {cfg.xi * (1 + cfg.xi_jitter):.3f}])")
        if cfg.f_jitter > 0:
            f_min = cfg.f_pix / (1 + cfg.f_jitter)
            f_max = cfg.f_pix / (1 - cfg.f_jitter) if cfg.f_jitter < 1.0 else float('inf')
            print(f"  f_jitter:     {cfg.f_jitter} (range: [{f_min:.1f}, {f_max:.1f}])")
        else:
            print(f"  f_jitter:     {cfg.f_jitter} (disabled)")
        print(f"  target:       {cfg.jitter_target}")
    print("=" * 60)


def main():
    args = parse_args()

    # Load config and merge with command line args
    config_path = None
    if args.config is not None:
        config_path = args.config
        cfg_file = load_config(config_path)
        cfg = merge_args_with_config(args, cfg_file)
    else:
        cfg = get_default_args(args)

    # Set random seed
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # Create dataset
    dataset = create_dataset(cfg)
    print_dataset_info(dataset, cfg, config_path)

    # Determine sample indices
    if cfg.sample_idx is not None:
        sample_indices = [cfg.sample_idx]
    else:
        max_idx = len(dataset) - 1
        sample_indices = np.random.choice(max_idx + 1, min(cfg.n_samples, len(dataset)), replace=False).tolist()

    # Visualize
    if cfg.repeat > 1:
        # Show jitter effect on single sample
        fig = visualize_jitter_effect(dataset, sample_indices[0], cfg.repeat, cfg)
    else:
        # Show multiple different samples
        fig = visualize_samples(dataset, sample_indices, cfg)

    # Save or show
    if cfg.save_path:
        fig.savefig(cfg.save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved visualization to: {cfg.save_path}")
    else:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
