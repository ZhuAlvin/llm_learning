#!/usr/bin/env python
"""
数据集下载脚本

下载所有学习所需的轻量级数据集
"""

import os
import urllib.request
from pathlib import Path

# 数据集目录
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# 数据集URL
DATASETS = {
    "tiny_shakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}

def download_file(url, filepath):
    """下载文件"""
    print(f"下载 {filepath.name}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ {filepath.name} 下载完成")
        return True
    except Exception as e:
        print(f"✗ {filepath.name} 下载失败: {e}")
        return False

def main():
    print("="*60)
    print("数据集下载")
    print("="*60)
    print()

    # 确保目录存在
    DATASETS_DIR.mkdir(exist_ok=True)

    # 下载数据集
    success_count = 0
    for filename, url in DATASETS.items():
        filepath = DATASETS_DIR / filename
        
        # 检查是否已存在
        if filepath.exists():
            print(f"⚠ {filename} 已存在，跳过")
            success_count += 1
            continue
        
        # 下载
        if download_file(url, filepath):
            success_count += 1

    print()
    print("="*60)
    print(f"完成！成功下载 {success_count}/{len(DATASETS)} 个数据集")
    print("="*60)

if __name__ == "__main__":
    main()
