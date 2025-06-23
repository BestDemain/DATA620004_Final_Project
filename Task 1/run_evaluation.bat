@echo off
echo ========================================
echo NeRF模型评估工具
echo ========================================
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

REM 检查必要文件
if not exist "transforms.json" (
    echo 错误: 未找到transforms.json文件
    pause
    exit /b 1
)

if not exist "transforms_base.ingp" (
    echo 错误: 未找到训练好的模型文件 transforms_base.ingp
    pause
    exit /b 1
)

echo 步骤1: 划分测试集...
echo ----------------------------------------
python split_test_set.py --transforms transforms.json --test_ratio 0.2 --seed 42
if errorlevel 1 (
    echo 错误: 测试集划分失败
    pause
    exit /b 1
)

echo.
echo 步骤2: 准备渲染测试图像
echo ----------------------------------------
echo 请按照以下步骤使用instant-ngp渲染测试图像:
echo.
echo 1. 启动 instant-ngp.exe
echo 2. 加载模型文件: transforms_base.ingp
echo 3. 加载测试集配置: transforms_test.json
echo 4. 在GUI中选择 "Render" 功能
echo 5. 设置输出目录为: test_renders
echo 6. 渲染所有测试视角的图像
echo.
echo 或者，如果支持命令行渲染，可以运行:
echo ../instant-ngp.exe transforms_test.json --load_snapshot transforms_base.ingp --screenshot_transforms transforms_test.json --screenshot_dir test_renders
echo.
echo 渲染完成后，按任意键继续...
pause

REM 检查渲染目录是否存在
if not exist "test_renders" (
    echo 警告: 未找到test_renders目录，请确保已完成渲染步骤
    echo 是否继续? (y/n)
    set /p continue=""
    if /i not "%continue%"=="y" (
        echo 评估已取消
        pause
        exit /b 1
    )
)

echo.
echo 步骤3: 计算评估指标...
echo ----------------------------------------
python calculate_metrics.py --mode transforms --transforms transforms_test.json --render_dir test_renders --output evaluation_results.json
if errorlevel 1 (
    echo 错误: 指标计算失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 评估完成!
echo ========================================
echo.
echo 生成的文件:
echo - transforms_train.json  (训练集配置)
echo - transforms_test.json   (测试集配置)
echo - evaluation_results.json (评估结果)
echo.
echo 详细使用说明请参考: README_evaluation.md
echo.
pause