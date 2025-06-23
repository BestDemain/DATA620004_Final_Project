# NeRF模型评估工具

本目录包含了用于评估NeRF模型性能的Python脚本工具集。

## 文件说明

### 1. `split_test_set.py` - 测试集划分工具
**功能**: 从完整数据集中划分出训练集和测试集

**使用方法**:
```bash
python split_test_set.py --transforms transforms.json --test_ratio 0.2 --seed 42
```

**参数说明**:
- `--transforms`: transforms.json文件路径 (默认: transforms.json)
- `--test_ratio`: 测试集比例 (默认: 0.2，即20%)
- `--seed`: 随机种子，确保结果可重复 (默认: 42)

**输出文件**:
- `transforms_train.json`: 训练集配置文件
- `transforms_test.json`: 测试集配置文件

### 2. `calculate_metrics.py` - 图像质量指标计算
**功能**: 计算渲染图像与真实图像之间的定量指标

**支持的指标**:
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比，越高越好
- **SSIM** (Structural Similarity Index): 结构相似性指数，越高越好 (0-1)
- **LPIPS** (Learned Perceptual Image Patch Similarity): 感知相似性，越低越好
- **MSE** (Mean Squared Error): 均方误差
- **MAE** (Mean Absolute Error): 平均绝对误差

**使用方法**:

**模式1: 基于transforms.json评估**
```bash
python calculate_metrics.py --mode transforms --transforms transforms_test.json --render_dir test_renders
```

**模式2: 比较两个目录**
```bash
python calculate_metrics.py --mode dirs --gt_dir ground_truth_images --pred_dir rendered_images
```

**参数说明**:
- `--mode`: 评估模式 (transforms 或 dirs)
- `--transforms`: 测试集transforms文件路径
- `--render_dir`: 渲染图像目录
- `--gt_dir`: 真实图像目录 (仅dirs模式)
- `--pred_dir`: 预测图像目录 (仅dirs模式)
- `--output`: 结果保存路径 (JSON格式)

### 3. `evaluate_nerf.py` - 完整评估流程
**功能**: 集成了数据集划分、模型渲染和指标计算的完整评估流程

**使用方法**:
```bash
python evaluate_nerf.py --data_dir . --model_path transforms_base.ingp --test_ratio 0.2
```

**参数说明**:
- `--data_dir`: 数据目录路径
- `--model_path`: 训练好的模型路径
- `--test_ratio`: 测试集比例
- `--no_render`: 跳过渲染步骤

## 完整评估流程

### 步骤1: 划分测试集
```bash
cd scripts
python split_test_set.py --transforms transforms.json --test_ratio 0.2
```

### 步骤2: 使用instant-ngp渲染测试图像
```bash
# 方法1: 使用GUI界面
# 1. 启动 instant-ngp.exe
# 2. 加载 transforms_base.ingp 模型
# 3. 加载 transforms_test.json
# 4. 使用 "Render" 功能渲染所有测试视角

# 方法2: 使用命令行 (如果支持)
../instant-ngp.exe transforms_test.json --load_snapshot transforms_base.ingp --screenshot_transforms transforms_test.json --screenshot_dir test_renders
```

### 步骤3: 计算评估指标
```bash
python calculate_metrics.py --mode transforms --transforms transforms_test.json --render_dir test_renders --output evaluation_results.json
```

## 依赖安装

### 基础依赖
```bash
pip install numpy opencv-python scikit-image
```

### LPIPS支持 (可选，用于感知质量评估)
```bash
pip install lpips torch
```

## 评估指标解释

### PSNR (Peak Signal-to-Noise Ratio)
- **范围**: 通常20-40 dB
- **含义**: 峰值信噪比，衡量图像重建质量
- **评判**: 越高越好，>30dB通常认为质量较好

### SSIM (Structural Similarity Index)
- **范围**: 0-1
- **含义**: 结构相似性，考虑亮度、对比度和结构
- **评判**: 越高越好，>0.9通常认为质量很好

### LPIPS (Learned Perceptual Image Patch Similarity)
- **范围**: 0-1
- **含义**: 基于深度学习的感知相似性
- **评判**: 越低越好，<0.1通常认为感知质量很好

## 典型NeRF评估结果参考

**优秀结果**:
- PSNR: >30 dB
- SSIM: >0.9
- LPIPS: <0.1

**良好结果**:
- PSNR: 25-30 dB
- SSIM: 0.8-0.9
- LPIPS: 0.1-0.2

**需要改进**:
- PSNR: <25 dB
- SSIM: <0.8
- LPIPS: >0.2

## 故障排除

### 1. 找不到渲染图像
- 检查渲染目录路径是否正确
- 确认图像命名格式 (支持 0000.png, 0001.png 等格式)

### 2. LPIPS计算失败
- 安装PyTorch: `pip install torch`
- 安装LPIPS: `pip install lpips`

### 3. 图像尺寸不匹配
- 脚本会自动调整图像尺寸进行比较
- 建议渲染时使用与原图相同的分辨率

### 4. 内存不足
- 减少批处理大小
- 使用较小的图像分辨率
- 关闭LPIPS计算 (如果不需要)

## 注意事项

1. **随机种子**: 使用相同的随机种子确保测试集划分的可重复性
2. **图像格式**: 支持 JPG, PNG 等常见格式
3. **路径分隔符**: 脚本会自动处理Windows/Linux路径差异
4. **GPU加速**: LPIPS计算会自动使用GPU (如果可用)
5. **结果保存**: 评估结果会保存为JSON格式，便于后续分析