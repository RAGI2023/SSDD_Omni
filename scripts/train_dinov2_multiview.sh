#!/bin/bash
# Training script for SSDD Multi-View with DINOv2 encoder

# Basic training with DINOv2 ViT-B
echo "Training SSDD Multi-View with DINOv2 encoder..."

accelerate launch ssdd/main_multiview.py \
    --config-name=SpiderEye_dinov2 \
    run_name=dinov2_vitb_multiview \
    dataset.batch_size=8 \
    training.epochs=300 \
    training.eval_freq=1 \
    training.log_freq=100

# Alternative: Use different DINOv2 variants
# ViT-Small (faster, less memory)
# accelerate launch ssdd/main_multiview.py \
#     --config-name=SpiderEye_dinov2 \
#     ssdd.encoder=dinov2_vits14_p14_c4 \
#     run_name=dinov2_vits_multiview

# ViT-Large (better performance, more memory)
# accelerate launch ssdd/main_multiview.py \
#     --config-name=SpiderEye_dinov2 \
#     ssdd.encoder=dinov2_vitl14_p14_c4 \
#     dataset.batch_size=4 \
#     run_name=dinov2_vitl_multiview
