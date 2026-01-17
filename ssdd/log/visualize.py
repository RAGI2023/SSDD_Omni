# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt


def show_generation_result(x_result, img_conditions=None, title="Generated samples"):
    n_samples = len(x_result)
    fig, axes = plt.subplots(n_samples, 2, figsize=(3 * 2, 3 * n_samples), sharex=True, sharey=True)

    if img_conditions is not None:
        for i_sample in range(n_samples):
            ax = axes[i_sample, 0]
            ax.imshow(img_conditions[i_sample].cpu().permute(1, 2, 0))
            ax.axis("off")
            ax.set_title("Image condition")

    for i_sample in range(n_samples):
        ax = axes[i_sample, 1]
        ax.imshow(x_result[i_sample].cpu().permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Reconstructed i={i_sample}")

    fig.tight_layout()
    if title:
        fig.suptitle(title)
    return fig


def show_multiview_result(views, gt_panorama, pred_panorama, title="Multi-View Panorama Reconstruction"):
    """
    Visualize multi-view fisheye inputs, GT panorama, and predicted panorama.

    Args:
        views: [N_samples, N_views, 3, H, W] - Input fisheye views
        gt_panorama: [N_samples, 3, H_pano, W_pano] - Ground truth panorama
        pred_panorama: [N_samples, 3, H_pano, W_pano] - Predicted panorama
        title: Figure title

    Returns:
        matplotlib Figure object
    """
    n_samples = len(views)
    n_views = views.shape[1]

    # Layout: n_views columns for fisheye + 1 column GT + 1 column Pred = n_views + 2 columns
    n_cols = n_views + 2
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.5 * n_cols, 3 * n_samples))

    # Ensure axes is 2D
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # Normalize images from [-1, 1] to [0, 1] for display
    def normalize_for_display(img):
        return ((img + 1) / 2).clamp(0, 1)

    for i_sample in range(n_samples):
        # Show fisheye views
        for i_view in range(n_views):
            ax = axes[i_sample, i_view]
            view_img = normalize_for_display(views[i_sample, i_view])
            ax.imshow(view_img.cpu().permute(1, 2, 0))
            ax.axis("off")
            if i_sample == 0:
                ax.set_title(f"View {i_view + 1}")

        # Show GT panorama
        ax = axes[i_sample, n_views]
        gt_img = normalize_for_display(gt_panorama[i_sample])
        ax.imshow(gt_img.cpu().permute(1, 2, 0))
        ax.axis("off")
        if i_sample == 0:
            ax.set_title("GT Panorama")

        # Show predicted panorama
        ax = axes[i_sample, n_views + 1]
        pred_img = normalize_for_display(pred_panorama[i_sample])
        ax.imshow(pred_img.cpu().permute(1, 2, 0))
        ax.axis("off")
        if i_sample == 0:
            ax.set_title("Pred Panorama")

    fig.tight_layout()
    if title:
        fig.suptitle(title, y=1.02)
    return fig
