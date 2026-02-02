#!/usr/bin/env python3
"""从 SSDDMultiView checkpoint 中提取 encoder 权重"""

from safetensors.torch import safe_open, save_file
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python extract_encoder.py <input_checkpoint> <output_file>")
    print("\nExample:")
    print("  python extract_encoder.py runs/jobs/train_enc_f8c4_2/checkpoints/best/model.safetensors encoder_weights.safetensors")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

print(f"从 {input_path} 提取 encoder 权重...")

# 读取 checkpoint
encoder_weights = {}
with safe_open(input_path, framework="pt") as f:
    keys = list(f.keys())

    # 查找 encoder 权重（支持多种前缀）
    encoder_prefixes = [
        "_orig_mod.ema.ema_model.base_ssdd.encoder.",  # EMA + compile
        "_orig_mod.base_ssdd.encoder.",                # compile only
        "ema.ema_model.base_ssdd.encoder.",            # EMA only
        "base_ssdd.encoder.",                          # base
        "encoder.",                                     # direct
        "ae.encoder.",                                  # ae wrapper
    ]

    for key in keys:
        for prefix in encoder_prefixes:
            if key.startswith(prefix):
                # 移除前缀，只保留 encoder 的权重名
                new_key = key.replace(prefix, "")
                encoder_weights[new_key] = f.get_tensor(key)
                break

if not encoder_weights:
    print("错误: 没有找到 encoder 权重!")
    print(f"\n文件中的前 10 个 key:")
    with safe_open(input_path, framework="pt") as f:
        for i, key in enumerate(list(f.keys())[:10], 1):
            print(f"  {i}. {key}")
    sys.exit(1)

print(f"找到 {len(encoder_weights)} 个 encoder 参数")

# 保存提取的权重
save_file(encoder_weights, output_path)
print(f"✅ 成功保存到 {output_path}")

# 显示一些提取的权重 key
print("\n提取的权重示例:")
for i, key in enumerate(list(encoder_weights.keys())[:5], 1):
    print(f"  {i}. {key}")
