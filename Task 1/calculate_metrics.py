#!/usr/bin/env python3
"""
图像质量指标计算脚本
功能：计算渲染图像与真实图像之间的PSNR、SSIM、LPIPS等指标
"""

import json
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys
from typing import List, Tuple, Dict

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except ImportError:
    print("请安装scikit-image: pip install scikit-image")
    sys.exit(1)

try:
    import lpips
    import torch
    LPIPS_AVAILABLE = True
except ImportError:
    print("警告: LPIPS不可用，如需使用请安装: pip install lpips torch")
    LPIPS_AVAILABLE = False

class ImageMetrics:
    def __init__(self):
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex')
            if torch.cuda.is_available():
                self.lpips_fn = self.lpips_fn.cuda()
                print("使用GPU计算LPIPS")
            else:
                print("使用CPU计算LPIPS")
    
    def load_image(self, path: str) -> np.ndarray:
        """加载图像并转换为RGB格式"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"无法加载图像: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """调整图像尺寸使其匹配"""
        if img1.shape != img2.shape:
            # 使用较小的尺寸
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        return img1, img2
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算PSNR"""
        img1, img2 = self.resize_to_match(img1, img2)
        return psnr(img1, img2, data_range=255)
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算SSIM"""
        img1, img2 = self.resize_to_match(img1, img2)
        return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=255)
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算LPIPS"""
        if not LPIPS_AVAILABLE:
            return -1.0
        
        img1, img2 = self.resize_to_match(img1, img2)
        
        # 转换为tensor并归一化到[-1, 1]
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 127.5 - 1.0
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 127.5 - 1.0
        
        if torch.cuda.is_available():
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()
        
        img1_tensor = img1_tensor.unsqueeze(0)
        img2_tensor = img2_tensor.unsqueeze(0)
        
        with torch.no_grad():
            lpips_score = self.lpips_fn(img1_tensor, img2_tensor)
        
        return lpips_score.item()
    
    def calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算均方误差"""
        img1, img2 = self.resize_to_match(img1, img2)
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    def calculate_mae(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算平均绝对误差"""
        img1, img2 = self.resize_to_match(img1, img2)
        return np.mean(np.abs(img1.astype(float) - img2.astype(float)))

def evaluate_image_pair(metrics: ImageMetrics, gt_path: str, pred_path: str) -> Dict[str, float]:
    """评估单对图像"""
    try:
        gt_img = metrics.load_image(gt_path)
        pred_img = metrics.load_image(pred_path)
        
        results = {
            'psnr': metrics.calculate_psnr(gt_img, pred_img),
            'ssim': metrics.calculate_ssim(gt_img, pred_img),
            'lpips': metrics.calculate_lpips(gt_img, pred_img),
            'mse': metrics.calculate_mse(gt_img, pred_img),
            'mae': metrics.calculate_mae(gt_img, pred_img)
        }
        
        return results
    
    except Exception as e:
        print(f"评估图像对时出错 ({gt_path}, {pred_path}): {e}")
        return None

def evaluate_from_transforms(transforms_path: str, render_dir: str) -> Dict[str, float]:
    """基于transforms.json文件评估渲染结果"""
    transforms_path = Path(transforms_path)
    render_dir = Path(render_dir)
    
    if not transforms_path.exists():
        raise FileNotFoundError(f"找不到transforms文件: {transforms_path}")
    
    if not render_dir.exists():
        raise FileNotFoundError(f"找不到渲染目录: {render_dir}")
    
    # 加载transforms
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    metrics = ImageMetrics()
    all_results = []
    
    print(f"评估 {len(transforms['frames'])} 张图像...")
    print("-" * 80)
    
    for i, frame in enumerate(transforms['frames']):
        # 真实图像路径
        gt_path = transforms_path.parent / frame['file_path']
        
        # 从file_path中提取图像编号
        img_number = Path(frame['file_path']).stem
        
        # 渲染图像路径（尝试多种命名方式）
        possible_names = [
            f"{i:04d}.png",
            f"{i:04d}.jpg",
            f"render_{i:04d}.png",
            f"render_{i:04d}.jpg",
            f"render_{img_number}.png",
            f"render_{img_number}.jpg",
            img_number + "_render.png",
            img_number + "_render.jpg"
        ]
        
        render_path = None
        for name in possible_names:
            candidate = render_dir / name
            if candidate.exists():
                render_path = candidate
                break
        
        if render_path is None:
            print(f"警告: 找不到图像 {frame['file_path']} 对应的渲染结果")
            continue
        
        # 计算指标
        result = evaluate_image_pair(metrics, str(gt_path), str(render_path))
        if result is not None:
            all_results.append(result)
            
            print(f"{i+1:3d}. {frame['file_path']:20s} -> {render_path.name:20s}")
            print(f"     PSNR: {result['psnr']:6.2f} dB, SSIM: {result['ssim']:6.4f}, "
                  f"LPIPS: {result['lpips']:6.4f}, MSE: {result['mse']:8.2f}")
    
    if not all_results:
        print("没有找到可评估的图像对")
        return {}
    
    # 计算统计结果
    stats = {}
    for metric in ['psnr', 'ssim', 'lpips', 'mse', 'mae']:
        values = [r[metric] for r in all_results if r[metric] >= 0]
        if values:
            stats[f'mean_{metric}'] = np.mean(values)
            stats[f'std_{metric}'] = np.std(values)
            stats[f'min_{metric}'] = np.min(values)
            stats[f'max_{metric}'] = np.max(values)
        else:
            stats[f'mean_{metric}'] = -1
            stats[f'std_{metric}'] = -1
            stats[f'min_{metric}'] = -1
            stats[f'max_{metric}'] = -1
    
    stats['num_images'] = len(all_results)
    
    return stats

def evaluate_directories(gt_dir: str, pred_dir: str) -> Dict[str, float]:
    """评估两个目录中的对应图像"""
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    
    # 获取所有图像文件
    gt_images = sorted([f for f in gt_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    pred_images = sorted([f for f in pred_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if len(gt_images) != len(pred_images):
        print(f"警告: 真实图像数量({len(gt_images)}) != 预测图像数量({len(pred_images)})")
    
    metrics = ImageMetrics()
    all_results = []
    
    min_count = min(len(gt_images), len(pred_images))
    print(f"评估 {min_count} 张图像...")
    print("-" * 80)
    
    for i in range(min_count):
        result = evaluate_image_pair(metrics, str(gt_images[i]), str(pred_images[i]))
        if result is not None:
            all_results.append(result)
            print(f"{i+1:3d}. {gt_images[i].name:20s} vs {pred_images[i].name:20s}")
            print(f"     PSNR: {result['psnr']:6.2f} dB, SSIM: {result['ssim']:6.4f}, "
                  f"LPIPS: {result['lpips']:6.4f}")
    
    # 计算统计结果
    stats = {}
    for metric in ['psnr', 'ssim', 'lpips', 'mse', 'mae']:
        values = [r[metric] for r in all_results if r[metric] >= 0]
        if values:
            stats[f'mean_{metric}'] = np.mean(values)
            stats[f'std_{metric}'] = np.std(values)
        else:
            stats[f'mean_{metric}'] = -1
            stats[f'std_{metric}'] = -1
    
    stats['num_images'] = len(all_results)
    
    return stats

def print_results(stats: Dict[str, float]):
    """打印评估结果"""
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    print(f"评估图像数量: {stats['num_images']}")
    print()
    
    if stats['mean_psnr'] >= 0:
        print(f"PSNR:  {stats['mean_psnr']:6.2f} ± {stats['std_psnr']:5.2f} dB")
    if stats['mean_ssim'] >= 0:
        print(f"SSIM:  {stats['mean_ssim']:6.4f} ± {stats['std_ssim']:6.4f}")
    if stats['mean_lpips'] >= 0:
        print(f"LPIPS: {stats['mean_lpips']:6.4f} ± {stats['std_lpips']:6.4f}")
    if stats['mean_mse'] >= 0:
        print(f"MSE:   {stats['mean_mse']:8.2f} ± {stats['std_mse']:8.2f}")
    if stats['mean_mae'] >= 0:
        print(f"MAE:   {stats['mean_mae']:6.2f} ± {stats['std_mae']:6.2f}")

def main():
    parser = argparse.ArgumentParser(description="计算图像质量指标")
    parser.add_argument("--mode", choices=['transforms', 'dirs'], default='transforms',
                       help="评估模式: transforms(基于transforms.json) 或 dirs(比较两个目录)")
    parser.add_argument("--transforms", type=str, default="transforms_test.json",
                       help="测试集transforms文件路径")
    parser.add_argument("--render_dir", type=str, default="test_renders",
                       help="渲染图像目录")
    parser.add_argument("--gt_dir", type=str, help="真实图像目录 (仅在dirs模式下使用)")
    parser.add_argument("--pred_dir", type=str, help="预测图像目录 (仅在dirs模式下使用)")
    parser.add_argument("--output", type=str, help="结果保存路径 (JSON格式)")
    
    args = parser.parse_args()
    
    if args.mode == 'transforms':
        print(f"使用transforms模式评估")
        print(f"Transforms文件: {args.transforms}")
        print(f"渲染目录: {args.render_dir}")
        stats = evaluate_from_transforms(args.transforms, args.render_dir)
    
    elif args.mode == 'dirs':
        if not args.gt_dir or not args.pred_dir:
            print("错误: dirs模式需要指定 --gt_dir 和 --pred_dir")
            return
        print(f"使用目录比较模式评估")
        print(f"真实图像目录: {args.gt_dir}")
        print(f"预测图像目录: {args.pred_dir}")
        stats = evaluate_directories(args.gt_dir, args.pred_dir)
    
    print_results(stats)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n结果已保存到: {args.output}")

if __name__ == "__main__":
    main()