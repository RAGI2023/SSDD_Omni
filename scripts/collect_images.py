#!/usr/bin/env python3
"""
将目录及其子目录下的所有图片复制或移动到指定目录。
每张图片会有唯一的名字，避免重名覆盖。
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}


def collect_images(src_dir: Path, dst_dir: Path, mode: str = 'copy') -> dict:
    """
    收集源目录下的所有图片到目标目录。

    Args:
        src_dir: 源目录
        dst_dir: 目标目录
        mode: 'copy' 或 'move'

    Returns:
        统计信息字典
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 用于跟踪同名文件
    name_counter = defaultdict(int)
    stats = {'total': 0, 'success': 0, 'failed': 0}

    # 递归查找所有图片
    for img_path in src_dir.rglob('*'):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stats['total'] += 1

        # 生成唯一文件名
        stem = img_path.stem
        suffix = img_path.suffix.lower()

        # 检查是否已存在同名文件，如果是则添加序号
        count = name_counter[stem + suffix]
        name_counter[stem + suffix] += 1

        if count == 0:
            new_name = f"{stem}{suffix}"
        else:
            new_name = f"{stem}_{count}{suffix}"

        dst_path = dst_dir / new_name

        # 确保目标文件名唯一（处理目标目录已有文件的情况）
        while dst_path.exists():
            count += 1
            name_counter[stem + suffix] = count + 1
            new_name = f"{stem}_{count}{suffix}"
            dst_path = dst_dir / new_name

        try:
            if mode == 'copy':
                shutil.copy2(img_path, dst_path)
            else:
                shutil.move(str(img_path), str(dst_path))
            stats['success'] += 1
            print(f"{'Copied' if mode == 'copy' else 'Moved'}: {img_path} -> {dst_path}")
        except Exception as e:
            stats['failed'] += 1
            print(f"Failed: {img_path} - {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='收集目录下所有图片到指定目录'
    )
    parser.add_argument('src', type=str, help='源目录路径')
    parser.add_argument('dst', type=str, help='目标目录路径')
    parser.add_argument(
        '--mode', '-m',
        choices=['copy', 'move'],
        default='copy',
        help='操作模式: copy(默认) 或 move'
    )

    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()

    if not src_dir.exists():
        print(f"错误: 源目录不存在: {src_dir}")
        return 1

    if not src_dir.is_dir():
        print(f"错误: 源路径不是目录: {src_dir}")
        return 1

    print(f"源目录: {src_dir}")
    print(f"目标目录: {dst_dir}")
    print(f"模式: {args.mode}")
    print("-" * 50)

    stats = collect_images(src_dir, dst_dir, args.mode)

    print("-" * 50)
    print(f"完成! 总计: {stats['total']}, 成功: {stats['success']}, 失败: {stats['failed']}")

    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
