#!/usr/bin/env python3
"""
简化的测试集划分脚本
功能：从现有的transforms.json中划分出测试集
"""

import json
import random
import shutil
from pathlib import Path
import argparse

def split_dataset(transforms_path, test_ratio=0.2, seed=42):
    """
    划分数据集为训练集和测试集
    
    Args:
        transforms_path: transforms.json文件路径
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 加载transforms.json
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    frames = transforms['frames'].copy()
    random.shuffle(frames)
    
    # 计算测试集大小
    n_total = len(frames)
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_test
    
    # 划分数据
    test_frames = frames[:n_test]
    train_frames = frames[n_test:]
    
    print(f"数据集划分完成:")
    print(f"  总图像数: {n_total}")
    print(f"  训练集: {n_train} 张 ({(n_train/n_total)*100:.1f}%)")
    print(f"  测试集: {n_test} 张 ({(n_test/n_total)*100:.1f}%)")
    
    # 创建训练集transforms
    train_transforms = transforms.copy()
    train_transforms['frames'] = train_frames
    
    # 创建测试集transforms
    test_transforms = transforms.copy()
    test_transforms['frames'] = test_frames
    
    # 保存文件
    base_path = Path(transforms_path).parent
    train_path = base_path / "transforms_train.json"
    test_path = base_path / "transforms_test.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
    
    print(f"\n文件已保存:")
    print(f"  训练集配置: {train_path}")
    print(f"  测试集配置: {test_path}")
    
    # 打印测试集图像列表
    print(f"\n测试集图像:")
    for i, frame in enumerate(test_frames):
        print(f"  {i+1:2d}. {frame['file_path']}")
    
    return train_frames, test_frames

def main():
    parser = argparse.ArgumentParser(description="划分NeRF数据集")
    parser.add_argument("--transforms", type=str, default="transforms.json", 
                       help="transforms.json文件路径")
    parser.add_argument("--test_ratio", type=float, default=0.2, 
                       help="测试集比例 (默认: 0.2)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    transforms_path = Path(args.transforms)
    if not transforms_path.exists():
        print(f"错误: 找不到文件 {transforms_path}")
        return
    
    print(f"正在处理: {transforms_path}")
    print(f"测试集比例: {args.test_ratio}")
    print(f"随机种子: {args.seed}")
    print("-" * 50)
    
    split_dataset(transforms_path, args.test_ratio, args.seed)

if __name__ == "__main__":
    main()