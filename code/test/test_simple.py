import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_environment():
    """测试运行环境是否正常"""
    print("\n=== 环境测试 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NumPy版本: {np.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"当前工作目录: {os.getcwd()}")
    print("环境测试通过！\n")
    return True

def test_performance(size=1000):
    """简单的性能测试"""
    print("\n=== 性能测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试CPU计算
    start_time = time.time()
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    c = torch.matmul(a, b)
    cpu_time = time.time() - start_time
    print(f"CPU矩阵乘法 ({size}x{size}): {cpu_time:.4f}秒")
    
    # 如果CUDA可用，测试GPU计算
    if torch.cuda.is_available():
        start_time = time.time()
        a = a.to(device)
        b = b.to(device)
        torch.cuda.synchronize()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU矩阵乘法 ({size}x{size}): {gpu_time:.4f}秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
    
    print("性能测试完成！\n")
    return True

def test_project_structure():
    """测试项目结构是否符合预期"""
    print("\n=== 项目结构测试 ===")
    
    # 检查主要目录
    essential_dirs = ["decoder", "test"]
    for dir_name in essential_dirs:
        assert os.path.isdir(dir_name), f"缺少必要目录: {dir_name}"
    
    # 检查decoder文件
    decoder_files = ["imageDecoder.py", "tabularDecoder.py", "omicsDecoder.py"]
    for file_name in decoder_files:
        file_path = os.path.join("decoder", file_name)
        assert os.path.isfile(file_path), f"缺少必要文件: {file_path}"
    
    print("所有必要的目录和文件都存在")
    print("项目结构测试通过！\n")
    return True

if __name__ == "__main__":
    print("===== 运行基本测试 =====")
    test_environment()
    test_project_structure()
    test_performance(size=500)  # 使用较小的size以加快测试
    print("===== 基本测试全部通过 =====")



