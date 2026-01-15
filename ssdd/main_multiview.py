# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main entry point for Multi-View SSDD training/evaluation.

Usage:
    # Train from scratch
    accelerate launch ssdd/main_multiview.py run_name=mv_train

    # Distillation
    accelerate launch ssdd/main_multiview.py \
        run_name=mv_distill \
        distill_teacher=true \
        ssdd.checkpoint=teacher@best

    # Evaluation
    accelerate launch ssdd/main_multiview.py \
        task=eval \
        ssdd.checkpoint=mv_train@best \
        ssdd.fm_sampler.steps=1
"""

import hydra
import lpips
import torch
from omegaconf import DictConfig

from ssdd.SpiderTask_MultiView import SpiderTasksMultiView


# Patching lpips loss to avoid NaN issues during training by increasing eps from 1e-10 to 1e-8
def _normalize_tensor(in_feat, eps=1e-8):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)


lpips.normalize_tensor = _normalize_tensor


@hydra.main(version_base=None, config_path="../config", config_name="SpiderEye")
def main(cfg: DictConfig):
    # Ensure return_all_views is True for multi-view training
    if 'return_all_views' not in cfg.dataset:
        cfg.dataset['return_all_views'] = True

    task = SpiderTasksMultiView(cfg)
    task()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
