# 3D Gaussian Splatting 评估工具

本文档介绍如何使用提供的脚本对3D Gaussian Splatting模型进行测试集划分和定量评估。

## 文件说明

- `create_test_split.py`: 测试集划分脚本
- `evaluate_test_set.py`: 测试集评估脚本
- `run_evaluation.py`: 完整评估流程脚本
- `EVALUATION_README.md`: 本说明文档

## 快速开始

### 方法1: 使用完整流程脚本（推荐）

```bash
# 在项目根目录运行
python run_evaluation.py --data_path ./data
```

这个命令会自动完成以下步骤：
1. 创建测试集划分（10%的图像作为测试集）
2. 检查模型训练状态，如果需要会自动训练
3. 在测试集上评估模型性能

### 方法2: 分步执行

#### 步骤1: 创建测试集划分

```bash
# 随机选择10%的图像作为测试集
python create_test_split.py ./data --test_ratio 0.1 --method random

# 均匀采样选择测试集
python create_test_split.py ./data --test_ratio 0.1 --method uniform

# 手动选择测试集
python create_test_split.py ./data --test_ratio 0.1 --method manual
```

#### 步骤2: 训练模型（如果还没有训练）

```bash
# 训练模型，启用评估模式
python train.py -s ./data -m ./data/output --eval
```

#### 步骤3: 评估模型

```bash
# 评估训练好的模型
python evaluate_test_set.py --model_path ./data/output --source_path ./data
```

## 详细参数说明

### create_test_split.py

```bash
python create_test_split.py <data_path> [options]
```

**必需参数:**
- `data_path`: 数据文件夹路径，包含`images`和`sparse`文件夹

**可选参数:**
- `--test_ratio`: 测试集比例，默认0.1 (10%)
- `--method`: 选择方法，可选`random`、`uniform`、`manual`，默认`random`
- `--output`: 输出文件路径，默认为`sparse/0/test.txt`
- `--seed`: 随机种子，默认42

**示例:**
```bash
# 选择15%的图像作为测试集，使用均匀采样
python create_test_split.py ./data --test_ratio 0.15 --method uniform --seed 123
```

### evaluate_test_set.py

```bash
python evaluate_test_set.py --model_path <model_path> --source_path <source_path> [options]
```

**必需参数:**
- `--model_path`: 训练好的模型路径
- `--source_path`: 源数据路径（包含images和sparse文件夹）

**可选参数:**
- `--iteration`: 模型迭代次数，-1表示最新，默认-1
- `--output`: 输出JSON文件路径，默认为`model_path/test_evaluation.json`

**示例:**
```bash
# 评估特定迭代的模型
python evaluate_test_set.py --model_path ./output --source_path ./data --iteration 30000
```

### run_evaluation.py

```bash
python run_evaluation.py --data_path <data_path> [options]
```

**必需参数:**
- `--data_path`: 数据文件夹路径

**可选参数:**
- `--model_path`: 模型路径，默认为`data_path/output`
- `--test_ratio`: 测试集比例，默认0.1
- `--split_method`: 测试集划分方法，默认`random`
- `--seed`: 随机种子，默认42
- `--skip_split`: 跳过测试集划分
- `--skip_training_check`: 跳过训练状态检查
- `--force_retrain`: 强制重新训练
- `--iteration`: 评估的模型迭代次数，默认-1（最新）

## 评估指标说明

脚本会计算以下三个主要指标：

1. **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比
   - 数值越高越好
   - 通常范围：20-40 dB
   - 衡量图像重建的像素级准确性

2. **SSIM (Structural Similarity Index)**: 结构相似性指数
   - 数值越高越好，范围：0-1
   - 衡量图像的结构相似性
   - 更符合人眼视觉感知

3. **LPIPS (Learned Perceptual Image Patch Similarity)**: 学习感知图像块相似性
   - 数值越低越好
   - 基于深度网络的感知相似性度量
   - 更好地反映人眼感知差异

## 输出文件格式

评估结果保存为JSON格式，包含：

```json
{
  "summary": {
    "num_test_images": 10,
    "average_metrics": {
      "psnr": 28.45,
      "ssim": 0.892,
      "lpips": 0.123
    },
    "std_metrics": {
      "psnr_std": 2.34,
      "ssim_std": 0.045,
      "lpips_std": 0.032
    }
  },
  "per_image_results": [
    {
      "image_name": "0001.jpg",
      "psnr": 29.12,
      "ssim": 0.901,
      "lpips": 0.115
    }
  ]
}
```

## 注意事项

1. **GPU要求**: 评估脚本需要CUDA支持，确保有足够的GPU内存

2. **依赖检查**: 确保已安装所有必需的依赖包：
   ```bash
   pip install torch torchvision tqdm pillow numpy
   ```

3. **数据结构**: 确保数据文件夹结构正确：
   ```
   data/
   ├── images/          # 原始图像
   ├── sparse/0/        # COLMAP输出
   │   ├── cameras.bin
   │   ├── images.bin
   │   ├── points3D.bin
   │   └── test.txt     # 测试集文件（脚本生成）
   └── output/          # 训练输出（可选）
   ```

4. **测试集选择**: 建议测试集比例在10-20%之间，确保有足够的训练数据

5. **重现性**: 使用相同的随机种子可以确保测试集划分的重现性

## 故障排除

### 常见错误及解决方案

1. **"No test cameras found"**
   - 检查`test.txt`文件是否存在且包含有效的图像名称
   - 确保图像名称与`images`文件夹中的文件匹配

2. **"CUDA out of memory"**
   - 减少批处理大小或使用较小的图像分辨率
   - 关闭其他占用GPU内存的程序

3. **"Model not found"**
   - 检查模型路径是否正确
   - 确保训练已完成并生成了`point_cloud.ply`文件

4. **"Images folder not found"**
   - 检查数据路径是否正确
   - 确保`images`文件夹存在且包含图像文件

## 示例工作流程

```bash
# 1. 进入项目目录
cd /path/to/gaussian-splatting

# 2. 运行完整评估流程
python run_evaluation.py --data_path ./data --test_ratio 0.15

# 3. 查看结果
cat ./data/output/test_evaluation.json
```

这将自动完成测试集划分、模型训练（如需要）和性能评估的完整流程。