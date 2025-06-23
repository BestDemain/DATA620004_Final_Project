#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集划分脚本
用于从COLMAP重建的图像中自动选择测试集图像
"""

import os
import random
import argparse
from pathlib import Path
import numpy as np

def create_test_split(data_path, test_ratio=0.1, method='random', output_file=None, seed=42):
    """
    创建测试集划分
    
    Args:
        data_path: 数据路径，包含images文件夹和sparse文件夹
        test_ratio: 测试集比例 (0.0-1.0)
        method: 划分方法 ('random', 'uniform', 'manual')
        output_file: 输出文件路径，默认为sparse/0/test.txt
        seed: 随机种子
    """
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 检查路径
    data_path = Path(data_path)
    images_path = data_path / "images"
    sparse_path = data_path / "sparse" / "0"
    
    if not images_path.exists():
        raise FileNotFoundError(f"Images folder not found: {images_path}")
    if not sparse_path.exists():
        raise FileNotFoundError(f"Sparse folder not found: {sparse_path}")
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_path.glob(f"*{ext}")))
        image_files.extend(list(images_path.glob(f"*{ext.upper()}")))
    
    image_names = [img.name for img in sorted(image_files)]
    
    if not image_names:
        raise ValueError(f"No image files found in {images_path}")
    
    print(f"Found {len(image_names)} images")
    
    # 计算测试集大小
    num_test = max(1, int(len(image_names) * test_ratio))
    print(f"Selecting {num_test} images for test set ({test_ratio*100:.1f}%)")
    
    # 根据方法选择测试图像
    if method == 'random':
        test_images = random.sample(image_names, num_test)
    elif method == 'uniform':
        # 均匀采样
        step = len(image_names) // num_test
        indices = [i * step for i in range(num_test)]
        test_images = [image_names[i] for i in indices]
    elif method == 'manual':
        print("Available images:")
        for i, name in enumerate(image_names):
            print(f"{i:3d}: {name}")
        
        print(f"\nPlease select {num_test} images for test set (enter indices separated by spaces):")
        selected_indices = input().strip().split()
        try:
            indices = [int(idx) for idx in selected_indices]
            if len(indices) != num_test:
                print(f"Warning: Expected {num_test} indices, got {len(indices)}")
            test_images = [image_names[i] for i in indices if 0 <= i < len(image_names)]
        except ValueError:
            print("Invalid input, falling back to random selection")
            test_images = random.sample(image_names, num_test)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 设置输出文件路径
    if output_file is None:
        output_file = sparse_path / "test.txt"
    else:
        output_file = Path(output_file)
    
    # 写入测试集文件
    with open(output_file, 'w') as f:
        for img_name in sorted(test_images):
            f.write(f"{img_name}\n")
    
    print(f"\nTest set saved to: {output_file}")
    print("Selected test images:")
    for img_name in sorted(test_images):
        print(f"  {img_name}")
    
    # 统计信息
    train_images = [name for name in image_names if name not in test_images]
    print(f"\nSummary:")
    print(f"  Total images: {len(image_names)}")
    print(f"  Training images: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")
    print(f"  Test ratio: {len(test_images)/len(image_names)*100:.1f}%")
    
    return test_images, train_images

def main():
    parser = argparse.ArgumentParser(description="Create test split for Gaussian Splatting")
    parser.add_argument("data_path", help="Path to data folder containing images and sparse folders")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    parser.add_argument("--method", choices=['random', 'uniform', 'manual'], default='random',
                       help="Selection method (default: random)")
    parser.add_argument("--output", help="Output file path (default: sparse/0/test.txt)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    try:
        create_test_split(
            data_path=args.data_path,
            test_ratio=args.test_ratio,
            method=args.method,
            output_file=args.output,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())