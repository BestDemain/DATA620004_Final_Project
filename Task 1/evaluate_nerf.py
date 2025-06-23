#!/usr/bin/env python3
"""
NeRF模型评估脚本
功能：
1. 从训练数据中划分测试集
2. 使用训练好的NeRF模型渲染测试图像
3. 计算定量评估指标（PSNR, SSIM, LPIPS）
"""

import json
import os
import shutil
import random
import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import subprocess
import sys

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
    print("警告: LPIPS不可用，请安装: pip install lpips torch")
    LPIPS_AVAILABLE = False

class NeRFEvaluator:
    def __init__(self, data_dir: str, model_path: str, test_ratio: float = 0.2):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.test_ratio = test_ratio
        self.transforms_path = self.data_dir / "transforms.json"
        self.test_transforms_path = self.data_dir / "transforms_test.json"
        self.train_transforms_path = self.data_dir / "transforms_train.json"
        
        # 初始化LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex')
            if torch.cuda.is_available():
                self.lpips_fn = self.lpips_fn.cuda()
    
    def load_transforms(self) -> Dict:
        """加载transforms.json文件"""
        with open(self.transforms_path, 'r') as f:
            return json.load(f)
    
    def split_dataset(self, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """划分训练集和测试集"""
        transforms = self.load_transforms()
        frames = transforms['frames']
        
        # 设置随机种子确保可重复性
        random.seed(seed)
        np.random.seed(seed)
        
        # 随机打乱并划分
        random.shuffle(frames)
        n_test = max(1, int(len(frames) * self.test_ratio))
        
        test_frames = frames[:n_test]
        train_frames = frames[n_test:]
        
        print(f"总图像数: {len(frames)}")
        print(f"训练集: {len(train_frames)} 张")
        print(f"测试集: {len(test_frames)} 张")
        
        return train_frames, test_frames
    
    def save_split_transforms(self, train_frames: List[Dict], test_frames: List[Dict]):
        """保存划分后的transforms文件"""
        transforms = self.load_transforms()
        
        # 保存训练集transforms
        train_transforms = transforms.copy()
        train_transforms['frames'] = train_frames
        with open(self.train_transforms_path, 'w') as f:
            json.dump(train_transforms, f, indent=2)
        
        # 保存测试集transforms
        test_transforms = transforms.copy()
        test_transforms['frames'] = test_frames
        with open(self.test_transforms_path, 'w') as f:
            json.dump(test_transforms, f, indent=2)
        
        print(f"已保存训练集配置: {self.train_transforms_path}")
        print(f"已保存测试集配置: {self.test_transforms_path}")
    
    def render_test_images(self, output_dir: str = "test_renders"):
        """使用instant-ngp渲染测试图像"""
        output_path = self.data_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        # 构建instant-ngp命令
        ngp_exe = self.data_dir.parent / "instant-ngp.exe"
        if not ngp_exe.exists():
            raise FileNotFoundError(f"找不到instant-ngp.exe: {ngp_exe}")
        
        cmd = [
            str(ngp_exe),
            str(self.test_transforms_path),
            "--load_snapshot", str(self.model_path),
            "--screenshot_transforms", str(self.test_transforms_path),
            "--screenshot_dir", str(output_path),
            "--width", "720",
            "--height", "1280"
        ]
        
        print(f"执行渲染命令: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.data_dir.parent))
            if result.returncode != 0:
                print(f"渲染失败: {result.stderr}")
                return False
            print("渲染完成")
            return True
        except Exception as e:
            print(f"执行渲染时出错: {e}")
            return False
    
    def load_image(self, path: str) -> np.ndarray:
        """加载图像并转换为RGB格式"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"无法加载图像: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算PSNR"""
        return psnr(img1, img2, data_range=255)
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算SSIM"""
        return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=255)
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算LPIPS"""
        if not LPIPS_AVAILABLE:
            return -1.0
        
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
    
    def evaluate_metrics(self, render_dir: str = "test_renders") -> Dict[str, float]:
        """评估渲染结果的定量指标"""
        render_path = self.data_dir / render_dir
        test_transforms = json.load(open(self.test_transforms_path))
        
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        
        print("\n开始评估测试图像...")
        
        for i, frame in enumerate(test_transforms['frames']):
            # 原始图像路径
            gt_path = self.data_dir / frame['file_path']
            
            # 渲染图像路径（假设按顺序命名）
            render_filename = f"{i:04d}.png"  # 或者根据实际命名规则调整
            render_path_full = render_path / render_filename
            
            if not render_path_full.exists():
                print(f"警告: 找不到渲染图像 {render_path_full}")
                continue
            
            try:
                # 加载图像
                gt_img = self.load_image(gt_path)
                render_img = self.load_image(render_path_full)
                
                # 确保图像尺寸一致
                if gt_img.shape != render_img.shape:
                    render_img = cv2.resize(render_img, (gt_img.shape[1], gt_img.shape[0]))
                
                # 计算指标
                psnr_val = self.calculate_psnr(gt_img, render_img)
                ssim_val = self.calculate_ssim(gt_img, render_img)
                lpips_val = self.calculate_lpips(gt_img, render_img)
                
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
                if lpips_val >= 0:
                    lpips_scores.append(lpips_val)
                
                print(f"图像 {frame['file_path']}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
                
            except Exception as e:
                print(f"处理图像 {frame['file_path']} 时出错: {e}")
                continue
        
        # 计算平均值
        results = {
            'mean_psnr': np.mean(psnr_scores) if psnr_scores else 0,
            'mean_ssim': np.mean(ssim_scores) if ssim_scores else 0,
            'mean_lpips': np.mean(lpips_scores) if lpips_scores else -1,
            'std_psnr': np.std(psnr_scores) if psnr_scores else 0,
            'std_ssim': np.std(ssim_scores) if ssim_scores else 0,
            'std_lpips': np.std(lpips_scores) if lpips_scores else -1,
            'num_images': len(psnr_scores)
        }
        
        return results
    
    def run_evaluation(self, render_images: bool = True):
        """运行完整的评估流程"""
        print("=== NeRF模型评估 ===")
        
        # 1. 划分数据集
        print("\n1. 划分训练集和测试集...")
        train_frames, test_frames = self.split_dataset()
        self.save_split_transforms(train_frames, test_frames)
        
        # 2. 渲染测试图像（可选）
        if render_images:
            print("\n2. 渲染测试图像...")
            if not self.render_test_images():
                print("渲染失败，跳过定量评估")
                return
        
        # 3. 计算评估指标
        print("\n3. 计算评估指标...")
        results = self.evaluate_metrics()
        
        # 4. 输出结果
        print("\n=== 评估结果 ===")
        print(f"测试图像数量: {results['num_images']}")
        print(f"平均PSNR: {results['mean_psnr']:.2f} ± {results['std_psnr']:.2f} dB")
        print(f"平均SSIM: {results['mean_ssim']:.4f} ± {results['std_ssim']:.4f}")
        if results['mean_lpips'] >= 0:
            print(f"平均LPIPS: {results['mean_lpips']:.4f} ± {results['std_lpips']:.4f}")
        else:
            print("LPIPS: 不可用")
        
        # 保存结果到文件
        results_path = self.data_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {results_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="NeRF模型评估")
    parser.add_argument("--data_dir", type=str, default=".", help="数据目录路径")
    parser.add_argument("--model_path", type=str, default="transforms_base.ingp", help="训练好的模型路径")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--no_render", action="store_true", help="跳过渲染步骤")
    
    args = parser.parse_args()
    
    evaluator = NeRFEvaluator(
        data_dir=args.data_dir,
        model_path=args.model_path,
        test_ratio=args.test_ratio
    )
    
    evaluator.run_evaluation(render_images=not args.no_render)

if __name__ == "__main__":
    main()