#!/usr/bin/env python
"""
环境测试脚本
检查所有必要的依赖是否正确安装
"""

import sys

def test_python_version():
    """测试Python版本"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3 and version.minor >= 9, "需要Python 3.9+"

def test_imports():
    """测试关键包导入"""
    packages = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
        ('jupyter', 'Jupyter'),
        ('tqdm', 'tqdm'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
    ]

    failed = []
    for pkg, name in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:20s} {version}")
        except ImportError:
            print(f"✗ {name:20s} - 未安装")
            failed.append(name)

    return len(failed) == 0, failed

def test_torch_cuda():
    """测试PyTorch CUDA支持"""
    import torch
    print(f"\nPyTorch版本: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA不可用 (将使用CPU模式)")
        print("  提示: 如需GPU加速，请安装CUDA并重新安装PyTorch GPU版本")

def test_jupyter():
    """测试Jupyter环境"""
    try:
        import jupyterlab
        print(f"\n✓ JupyterLab {jupyterlab.__version__}")
        print("  启动命令: jupyter lab")
    except ImportError:
        print("\n✗ JupyterLab未安装")

def main():
    print("="*70)
    print(" "*20 + "大模型学习指南 - 环境测试")
    print("="*70)
    print()

    # 测试Python版本
    try:
        test_python_version()
    except AssertionError as e:
        print(f"✗ {e}")
        sys.exit(1)

    print()
    print("-"*70)
    print("检查依赖包...")
    print("-"*70)

    # 测试包导入
    success, failed = test_imports()

    if not success:
        print(f"\n✗ 以下包未安装: {', '.join(failed)}")
        print("\n请运行以下命令安装:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print()
    print("-"*70)
    print("检查GPU支持...")
    print("-"*70)

    # 测试CUDA
    test_torch_cuda()

    # 测试Jupyter
    test_jupyter()

    print()
    print("="*70)
    print("✓ 环境配置完成！所有依赖已正确安装。")
    print("="*70)
    print()
    print("下一步:")
    print("  1. 启动Jupyter Lab: jupyter lab")
    print("  2. 打开 notebooks/Module01_Foundation/ 开始学习")
    print()

if __name__ == "__main__":
    main()
