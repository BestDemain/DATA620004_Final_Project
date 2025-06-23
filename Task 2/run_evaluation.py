#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整评估流程脚本
整合测试集划分、模型训练检查和定量评估
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_model_exists(model_path):
    """检查模型是否存在"""
    model_path = Path(model_path)
    
    # 检查模型目录
    if not model_path.exists():
        return False, f"Model directory does not exist: {model_path}"
    
    # 检查point_cloud目录
    point_cloud_dir = model_path / "point_cloud"
    if not point_cloud_dir.exists():
        return False, f"Point cloud directory does not exist: {point_cloud_dir}"
    
    # 查找最新的迭代
    iteration_dirs = [d for d in point_cloud_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")]
    if not iteration_dirs:
        return False, f"No iteration directories found in: {point_cloud_dir}"
    
    # 获取最新迭代
    latest_iteration = max(iteration_dirs, key=lambda x: int(x.name.split("_")[1]))
    ply_file = latest_iteration / "point_cloud.ply"
    
    if not ply_file.exists():
        return False, f"Point cloud file does not exist: {ply_file}"
    
    iteration_num = int(latest_iteration.name.split("_")[1])
    return True, f"Model found at iteration {iteration_num}"

def run_command(cmd, description):
    """运行命令并处理输出"""
    print(f"\n{description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete evaluation pipeline for Gaussian Splatting")
    parser.add_argument("--data_path", required=True, help="Path to data folder")
    parser.add_argument("--model_path", help="Path to trained model (default: data_path/output)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    parser.add_argument("--split_method", choices=['random', 'uniform', 'manual'], default='random',
                       help="Test split method (default: random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip_split", action="store_true", help="Skip test split creation")
    parser.add_argument("--skip_training_check", action="store_true", help="Skip training status check")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain even if model exists")
    parser.add_argument("--iteration", type=int, default=-1, help="Model iteration to evaluate (-1 for latest)")
    
    args = parser.parse_args()
    
    # 设置路径
    data_path = Path(args.data_path)
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = data_path / "output"
    
    print("Gaussian Splatting Evaluation Pipeline")
    print("=" * 50)
    print(f"Data path: {data_path}")
    print(f"Model path: {model_path}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Split method: {args.split_method}")
    
    # 检查数据路径
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        return 1
    
    images_path = data_path / "images"
    sparse_path = data_path / "sparse" / "0"
    
    if not images_path.exists():
        print(f"Error: Images folder not found: {images_path}")
        return 1
    
    if not sparse_path.exists():
        print(f"Error: Sparse folder not found: {sparse_path}")
        return 1
    
    # 步骤1: 创建测试集划分
    test_file = sparse_path / "test.txt"
    if not args.skip_split:
        if test_file.exists():
            print(f"\nTest file already exists: {test_file}")
            response = input("Do you want to recreate it? (y/N): ")
            if response.lower() != 'y':
                print("Skipping test split creation")
            else:
                cmd = [
                    sys.executable, "create_test_split.py",
                    str(data_path),
                    "--test_ratio", str(args.test_ratio),
                    "--method", args.split_method,
                    "--seed", str(args.seed)
                ]
                if not run_command(cmd, "Creating test split"):
                    return 1
        else:
            cmd = [
                sys.executable, "create_test_split.py",
                str(data_path),
                "--test_ratio", str(args.test_ratio),
                "--method", args.split_method,
                "--seed", str(args.seed)
            ]
            if not run_command(cmd, "Creating test split"):
                return 1
    
    # 检查测试集文件
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        return 1
    
    # 步骤2: 检查模型训练状态
    model_exists = False
    if not args.skip_training_check:
        print("\nChecking model training status...")
        model_exists, message = check_model_exists(model_path)
        print(message)
        
        if not model_exists or args.force_retrain:
            if args.force_retrain:
                print("Force retrain requested")
            
            print("\nStarting model training...")
            cmd = [
                sys.executable, "train.py",
                "-s", str(data_path),
                "-m", str(model_path),
                "--eval"
            ]
            
            print("Training command:")
            print(" ".join(cmd))
            print("\nNote: Training may take several hours depending on your hardware.")
            print("You can monitor progress in the output directory.")
            
            if not run_command(cmd, "Training model"):
                print("Training failed. Please check the error messages above.")
                return 1
            
            model_exists = True
    
    # 步骤3: 评估模型
    if model_exists or args.skip_training_check:
        print("\nEvaluating model on test set...")
        
        output_file = model_path / "test_evaluation.json"
        cmd = [
            sys.executable, "evaluate_test_set.py",
            "--model_path", str(model_path),
            "--source_path", str(data_path),
            "--iteration", str(args.iteration),
            "--output", str(output_file)
        ]
        
        if not run_command(cmd, "Evaluating model"):
            return 1
        
        print(f"\nEvaluation complete! Results saved to: {output_file}")
    else:
        print("\nSkipping evaluation due to missing model")
    
    print("\nPipeline completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())