#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估工具演示脚本
展示如何使用测试集划分和评估工具
"""

import os
import sys
import json
from pathlib import Path

def demo_test_split():
    """演示测试集划分功能"""
    print("=" * 60)
    print("演示：测试集划分")
    print("=" * 60)
    
    data_path = Path("./data")
    
    # 检查数据
    images_path = data_path / "images"
    if not images_path.exists():
        print(f"错误：找不到图像文件夹 {images_path}")
        return False
    
    # 统计图像数量
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    print(f"发现 {len(image_files)} 张图像")
    
    # 检查测试集文件
    test_file = data_path / "sparse" / "0" / "test.txt"
    if test_file.exists():
        with open(test_file, 'r') as f:
            test_images = [line.strip() for line in f if line.strip()]
        
        print(f"\n当前测试集包含 {len(test_images)} 张图像：")
        for img in test_images:
            print(f"  - {img}")
        
        print(f"\n测试集比例: {len(test_images)/len(image_files)*100:.1f}%")
        print(f"训练集图像数量: {len(image_files) - len(test_images)}")
    else:
        print("\n测试集文件不存在，请运行以下命令创建：")
        print("python create_test_split.py ./data --test_ratio 0.1")
    
    return True

def demo_model_check():
    """演示模型检查功能"""
    print("\n" + "=" * 60)
    print("演示：模型状态检查")
    print("=" * 60)
    
    # 检查可能的模型路径
    possible_paths = [
        Path("./data/output"),
        Path("./output"),
        Path("./models")
    ]
    
    model_found = False
    for model_path in possible_paths:
        print(f"\n检查模型路径: {model_path}")
        
        if not model_path.exists():
            print("  ❌ 路径不存在")
            continue
        
        # 检查point_cloud目录
        point_cloud_dir = model_path / "point_cloud"
        if not point_cloud_dir.exists():
            print("  ❌ 未找到 point_cloud 目录")
            continue
        
        # 查找迭代目录
        iteration_dirs = [d for d in point_cloud_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("iteration_")]
        
        if not iteration_dirs:
            print("  ❌ 未找到迭代目录")
            continue
        
        # 检查最新迭代
        latest_iteration = max(iteration_dirs, key=lambda x: int(x.name.split("_")[1]))
        ply_file = latest_iteration / "point_cloud.ply"
        
        if ply_file.exists():
            iteration_num = int(latest_iteration.name.split("_")[1])
            file_size = ply_file.stat().st_size / (1024*1024)  # MB
            print(f"  ✅ 找到训练好的模型")
            print(f"     迭代次数: {iteration_num}")
            print(f"     模型文件: {ply_file}")
            print(f"     文件大小: {file_size:.1f} MB")
            model_found = True
            break
        else:
            print(f"  ❌ 迭代 {latest_iteration.name} 中未找到 point_cloud.ply")
    
    if not model_found:
        print("\n❌ 未找到训练好的模型")
        print("\n要训练模型，请运行：")
        print("python train.py -s ./data -m ./data/output --eval")
        print("\n注意：训练可能需要几个小时，取决于您的硬件配置")
    
    return model_found

def demo_evaluation_workflow():
    """演示完整评估流程"""
    print("\n" + "=" * 60)
    print("演示：完整评估流程")
    print("=" * 60)
    
    print("\n完整的评估流程包括以下步骤：")
    print("\n1. 测试集划分")
    print("   - 从所有图像中选择一部分作为测试集")
    print("   - 支持随机、均匀、手动三种选择方式")
    print("   - 生成 sparse/0/test.txt 文件")
    
    print("\n2. 模型训练（如果需要）")
    print("   - 使用训练集训练3D Gaussian模型")
    print("   - 启用评估模式以支持测试集评估")
    print("   - 生成 point_cloud.ply 等模型文件")
    
    print("\n3. 定量评估")
    print("   - 在测试集上渲染图像")
    print("   - 计算PSNR、SSIM、LPIPS指标")
    print("   - 生成详细的评估报告")
    
    print("\n可用的脚本：")
    print("   - create_test_split.py: 创建测试集划分")
    print("   - evaluate_test_set.py: 评估训练好的模型")
    print("   - run_evaluation.py: 完整流程自动化")
    
    print("\n使用示例：")
    print("   # 快速开始（推荐）")
    print("   python run_evaluation.py --data_path ./data")
    print("")
    print("   # 分步执行")
    print("   python create_test_split.py ./data --test_ratio 0.1")
    print("   python train.py -s ./data -m ./data/output --eval")
    print("   python evaluate_test_set.py --model_path ./data/output --source_path ./data")

def demo_metrics_explanation():
    """演示评估指标说明"""
    print("\n" + "=" * 60)
    print("演示：评估指标说明")
    print("=" * 60)
    
    print("\n评估指标详解：")
    
    print("\n📊 PSNR (Peak Signal-to-Noise Ratio)")
    print("   - 峰值信噪比，衡量图像重建的像素级准确性")
    print("   - 数值越高越好，通常范围：20-40 dB")
    print("   - 计算公式基于均方误差(MSE)")
    
    print("\n📊 SSIM (Structural Similarity Index)")
    print("   - 结构相似性指数，衡量图像的结构相似性")
    print("   - 数值越高越好，范围：0-1")
    print("   - 更符合人眼视觉感知")
    
    print("\n📊 LPIPS (Learned Perceptual Image Patch Similarity)")
    print("   - 学习感知图像块相似性")
    print("   - 数值越低越好")
    print("   - 基于深度网络，更好地反映人眼感知差异")
    
    print("\n典型的好结果范围：")
    print("   - PSNR: > 25 dB (优秀: > 30 dB)")
    print("   - SSIM: > 0.8 (优秀: > 0.9)")
    print("   - LPIPS: < 0.2 (优秀: < 0.1)")

def main():
    print("3D Gaussian Splatting 评估工具演示")
    print("" * 60)
    
    # 检查当前目录
    if not Path("./data").exists():
        print("错误：当前目录下未找到 data 文件夹")
        print("请确保在 gaussian-splatting 项目根目录下运行此脚本")
        return 1
    
    # 运行演示
    demo_test_split()
    demo_model_check()
    demo_evaluation_workflow()
    demo_metrics_explanation()
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n下一步建议：")
    print("1. 如果还没有测试集，运行: python create_test_split.py ./data")
    print("2. 如果还没有训练模型，运行: python train.py -s ./data -m ./data/output --eval")
    print("3. 评估模型性能，运行: python evaluate_test_set.py --model_path ./data/output --source_path ./data")
    print("4. 或者使用一键脚本: python run_evaluation.py --data_path ./data")
    print("\n详细说明请参考: EVALUATION_README.md")
    
    return 0

if __name__ == "__main__":
    exit(main())