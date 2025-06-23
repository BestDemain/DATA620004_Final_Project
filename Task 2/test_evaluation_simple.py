#!/usr/bin/env python3
"""
简化的测试评估脚本，用于验证评估功能
"""

import os
import sys
import torch
import json
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

def test_evaluation():
    """测试评估功能"""
    model_path = "./data/output"
    source_path = "./data"
    
    print("Starting evaluation test...")
    print(f"Model path: {model_path}")
    print(f"Source path: {source_path}")
    
    # 检查必要文件
    test_file = os.path.join(source_path, "sparse", "0", "test.txt")
    if not os.path.exists(test_file):
        print(f"Error: test.txt not found at {test_file}")
        return False
    
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return False
    
    try:
        # 创建参数解析器
        parser = ArgumentParser()
        
        # 设置参数
        model_params = ModelParams(parser)
        model_params._model_path = model_path
        model_params._source_path = source_path
        model_params._images = "images"
        model_params.eval = True
        
        pipeline_params = PipelineParams(parser)
        
        # 创建args对象用于Scene初始化
        args = Namespace()
        args.model_path = model_path
        args.source_path = source_path
        args.images = "images"
        args.depths = ""
        args.eval = True
        args.train_test_exp = False
        args.white_background = False
        args.resolution = -1
        args.data_device = "cuda"
        
        # 加载场景和模型
        print("Loading scene...")
        gaussians = GaussianModel(model_params.sh_degree)
        scene = Scene(args, gaussians, load_iteration=-1)
        
        # 获取测试相机
        test_cameras = scene.test_cameras[1.0]
        
        if not test_cameras:
            print("Error: No test cameras found")
            return False
        
        print(f"Found {len(test_cameras)} test cameras")
        
        # 测试渲染一张图像
        print("Testing rendering...")
        camera = test_cameras[0]
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        with torch.no_grad():
            rendered_image = render(camera, gaussians, pipeline_params, background)["render"]
        
        print(f"Rendered image shape: {rendered_image.shape}")
        print("Evaluation test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation()
    if success:
        print("\n✅ Evaluation test passed!")
        sys.exit(0)
    else:
        print("\n❌ Evaluation test failed!")
        sys.exit(1)