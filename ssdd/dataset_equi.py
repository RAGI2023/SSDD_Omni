# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Any

import torch
import torchvision.transforms.v2 as transforms

from .mutils.main_utils import TaskState

# Import EquiDataset from utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.EquiDataset import EquiDataset, DEFAULT_JITTER_CONFIG, NO_JITTER_CONFIG


class EquiDatasetWrapper(EquiDataset):
    """
    Wrapper around EquiDataset for multi-view SSDD training.

    Returns:
        - Multi-view fisheye images: [N_views, 3, H, W]
        - Original panorama: [3, H_pano, W_pano]

    Both normalized to [-1, 1] for SSDD compatibility.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        im_size: int = 128,
        transform: Any = None,
        return_all_views: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            root: Root directory containing panorama images
            split: "train" or "val" (used to select train/val subfolder)
            im_size: Output image size (used for canvas_size and out_w/out_h)
            transform: Optional torchvision transforms (applied after fisheye conversion)
            return_all_views: If True, return all 4 views; if False, return only first view (backward compat)
            **kwargs: Additional arguments for EquiDataset
        """
        assert split in ["train", "val"]
        self.split = split
        self.im_size = im_size
        self.custom_transform = transform
        self.return_all_views = return_all_views

        # Construct folder path based on split
        folder_path = os.path.join(root, split)

        # Determine jitter config based on split
        jitter_cfg = DEFAULT_JITTER_CONFIG if split == "train" else NO_JITTER_CONFIG

        # Default UCM parameters (can be overridden by kwargs)
        # Equirectangular panorama should be 2:1 aspect ratio (360° x 180°)
        ucm_params = {
            "canvas_size": (im_size * 2, im_size),  # 2:1 panorama (width, height) = (256, 128)
            "out_w": im_size,
            "out_h": im_size,
            "f_pix": 220.0,
            "xi": 0.9,
            "mask_mode": "inscribed",
            "jitter_cfg": jitter_cfg,
        }

        # Override with user-provided kwargs
        ucm_params.update(kwargs)

        # Initialize parent EquiDataset
        super().__init__(folder_path, **ucm_params)

    def __getitem__(self, idx):
        """
        Returns:
            imgs: Tensor [Nviews, 3, H, W] in range [-1, 1] OR [3, H, W] if return_all_views=False
            img_original: Tensor [3, H_pano, W_pano] in range [-1, 1] (panorama)
        """
        # Get fisheye views and original panorama from parent class
        imgs, img_original = super().__getitem__(idx)

        # imgs: [Nviews, 3, H, W] in [0, 1]
        # img_original: [3, Hc, Wc] in [0, 1]

        # Normalize to [-1, 1] (SSDD standard)
        imgs = imgs * 2.0 - 1.0
        img_original = img_original * 2.0 - 1.0

        # Apply custom transforms if provided
        if self.custom_transform is not None:
            if self.return_all_views:
                # Apply transform to each view
                imgs = torch.stack([self.custom_transform(imgs[i]) for i in range(imgs.shape[0])])
            else:
                imgs = self.custom_transform(imgs[0])
            img_original = self.custom_transform(img_original)

        # Return format based on return_all_views flag
        if self.return_all_views:
            # Multi-view mode: return all views + panorama
            return imgs, img_original  # [N_views, 3, H, W], [3, H_pano, W_pano]
        else:
            # Single-view mode (backward compatibility): return first view + dummy label
            return imgs[0], 0  # [3, H, W], int

    def extra_repr(self) -> str:
        return f"Split: {self.split}, Image size: {self.im_size}, Multi-view: {self.return_all_views}"


def make_transform(is_train_split, im_size, aug_scale):
    """
    Create transform pipeline.
    Note: For EquiDataset, most augmentations are done in the fisheye conversion.
    Here we just pass through since normalization is done in __getitem__.
    """
    # Identity transform - normalization already done in __getitem__
    return transforms.Lambda(lambda x: x)


def make_dataset_and_loader(
    is_train,
    *,
    imagenet_root,
    im_size,
    batch_size,
    aug_scale=None,
    limit=None,
    num_workers=10,
    return_all_views=True,
    **equi_kwargs
):
    """
    Create EquiDataset and DataLoader.

    Args:
        is_train: Whether this is training split
        imagenet_root: Root directory (will look for train/val subdirs)
        im_size: Image size
        batch_size: Total batch size across all GPUs
        aug_scale: Unused for EquiDataset (augmentation handled internally)
        limit: Limit number of samples
        return_all_views: If True, return multi-view data; if False, single view + label
        **equi_kwargs: Additional arguments for EquiDataset (f_pix, xi, etc.)
    """
    transform = make_transform(is_train, im_size=im_size, aug_scale=aug_scale)

    dataset = EquiDatasetWrapper(
        imagenet_root,
        "train" if is_train else "val",
        im_size=im_size,
        transform=transform,
        return_all_views=return_all_views,
        **equi_kwargs
    )

    if limit is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(limit, len(dataset)))))

    num_proc = TaskState().accelerator.num_processes
    gpu_batch_size = batch_size // num_proc
    assert gpu_batch_size * num_proc == batch_size, f"Batch size {batch_size} not divisible by number of processes {num_proc}"

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=gpu_batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=10 if num_workers > 0 else None,  # Prefetch
    )

    return dataset, loader


def load_equirect(ds_cfg):
    """
    Load equirectangular panorama dataset (replacement for load_imagenet).

    Args:
        ds_cfg: Dataset config dict with keys:
            - imagenet_root: root path
            - im_size: image size
            - batch_size: batch size
            - aug_scale: (unused)
            - limit: optional sample limit
            - return_all_views: if True, return multi-view data (default: True)
            - Additional EquiDataset params (f_pix, xi, mask_mode, etc.)
    """
    train_dataset, train_loader = make_dataset_and_loader(True, **ds_cfg)
    test_dataset, test_loader = make_dataset_and_loader(False, **ds_cfg)

    return (train_dataset, test_dataset), (train_loader, test_loader)
