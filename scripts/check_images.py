#!/usr/bin/env python3
"""检测并处理目录下损坏的图片文件"""

import argparse
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}


def is_valid_image(path: Path) -> bool:
    """检查图片是否有效"""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description='检测损坏的图片')
    parser.add_argument('dir', help='目录路径')
    parser.add_argument('--remove', '-r', action='store_true', help='删除损坏的图片')
    args = parser.parse_args()

    src = Path(args.dir)
    if not src.is_dir():
        print(f"错误: {src} 不是有效目录")
        return 1

    corrupted = []
    total = 0

    for f in src.rglob('*'):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            total += 1
            if not is_valid_image(f):
                corrupted.append(f)
                print(f"[损坏] {f}")
                if args.remove:
                    f.unlink()
                    print(f"  已删除")

    print(f"\n总计: {total}, 损坏: {len(corrupted)}")
    return 0


if __name__ == '__main__':
    exit(main())
