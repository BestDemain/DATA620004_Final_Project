#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集评估脚本
用于在测试集上评估3D Gaussian Splatting模型的定量结果
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel

def load_image(image_path):
    """加载图像并转换为tensor"""
    image = Image.open(image_path)
    return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :]

def evaluate_metrics(gt_image, rendered_image):
    """计算评估指标"""
    # 确保图像在GPU上
    if torch.cuda.is_available():
        gt_image = gt_image.cuda()
        rendered_image = rendered_image.cuda()
    
    # 计算PSNR
    psnr_value = psnr(rendered_image, gt_image).mean().item()
    
    # 计算SSIM
    ssim_value = ssim(rendered_image, gt_image).mean().item()
    
    # 计算LPIPS
    lpips_value = lpips(rendered_image, gt_image, net_type='vgg').item()
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'lpips': lpips_value
    }

def render_test_images(model_path, source_path, iteration=-1):
    """渲染测试集图像"""
    # 创建参数解析器
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # 设置参数
    model_params = ModelParams(parser)
    model_params._model_path = model_path
    model_params._source_path = source_path
    model_params._images = "images"
    model_params.eval = True
    
    pipeline_params = PipelineParams(parser)
    
    # 创建args对象用于Scene初始化
    from argparse import Namespace
    args = Namespace()
    args.model_path = model_path
    args.source_path = source_path
    args.images = "images"
    args.depths = ""
    args.eval = True
    args.train_test_exp = False
    args.white_background = False
    args.resolution = -1  # 添加缺失的resolution属性
    args.data_device = "cuda"
    
    # 加载场景和模型
    print("Loading scene...")
    gaussians = GaussianModel(model_params.sh_degree)
    scene = Scene(args, gaussians, load_iteration=iteration)
    
    # 获取测试相机
    test_cameras = scene.test_cameras[1.0]  # 使用原始分辨率
    
    if not test_cameras:
        raise ValueError("No test cameras found. Make sure test.txt exists and contains valid image names.")
    
    print(f"Found {len(test_cameras)} test cameras")
    
    # 渲染测试图像
    results = []
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    for idx, camera in enumerate(tqdm(test_cameras, desc="Rendering test images")):
        # 渲染图像
        with torch.no_grad():
            rendered_image = render(camera, gaussians, pipeline_params, background)["render"]
        
        # 获取真实图像
        gt_image = camera.original_image[0:3, :, :].unsqueeze(0)
        
        # 计算指标
        metrics = evaluate_metrics(gt_image, rendered_image.unsqueeze(0))
        metrics['image_name'] = camera.image_name
        
        results.append(metrics)
        
        print(f"Image {camera.image_name}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}, LPIPS={metrics['lpips']:.3f}")
    
    return results

def save_results(results, output_path):
    """保存评估结果"""
    # 计算平均指标
    avg_metrics = {
        'psnr': np.mean([r['psnr'] for r in results]),
        'ssim': np.mean([r['ssim'] for r in results]),
        'lpips': np.mean([r['lpips'] for r in results])
    }
    
    # 计算标准差
    std_metrics = {
        'psnr_std': np.std([r['psnr'] for r in results]),
        'ssim_std': np.std([r['ssim'] for r in results]),
        'lpips_std': np.std([r['lpips'] for r in results])
    }
    
    # 组织结果
    evaluation_results = {
        'summary': {
            'num_test_images': len(results),
            'average_metrics': avg_metrics,
            'std_metrics': std_metrics
        },
        'per_image_results': results
    }
    
    # 保存到JSON文件
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # 打印摘要
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Number of test images: {len(results)}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} ± {std_metrics['psnr_std']:.2f}")
    print(f"Average SSIM: {avg_metrics['ssim']:.3f} ± {std_metrics['ssim_std']:.3f}")
    print(f"Average LPIPS: {avg_metrics['lpips']:.3f} ± {std_metrics['lpips_std']:.3f}")
    print("="*50)
    
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaussian Splatting model on test set")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--source_path", required=True, help="Path to source data (containing images and sparse folders)")
    parser.add_argument("--iteration", type=int, default=-1, help="Model iteration to load (-1 for latest)")
    parser.add_argument("--output", help="Output JSON file path (default: model_path/test_evaluation.json)")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1
    
    if not os.path.exists(args.source_path):
        print(f"Error: Source path does not exist: {args.source_path}")
        return 1
    
    # 检查test.txt是否存在
    test_file = os.path.join(args.source_path, "sparse", "0", "test.txt")
    if not os.path.exists(test_file):
        print(f"Error: test.txt not found at {test_file}")
        print("Please run create_test_split.py first to create test split.")
        return 1
    
    # 设置输出路径
    if args.output is None:
        args.output = os.path.join(args.model_path, "test_evaluation.json")
    
    try:
        # 检查CUDA
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU (will be slow)")
        
        # 渲染和评估
        print(f"Evaluating model: {args.model_path}")
        print(f"Source data: {args.source_path}")
        print(f"Iteration: {args.iteration}")
        
        results = render_test_images(args.model_path, args.source_path, args.iteration)
        
        # 保存结果
        save_results(results, args.output)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())