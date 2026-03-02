# 环境搭建指南

本文档提供详细的环境搭建步骤，确保您能够顺利运行所有学习材料。

---

## 系统要求

### 最低配置（CPU模式）

- **CPU**: 4核心以上
- **内存**: 8GB RAM
- **硬盘**: 20GB可用空间
- **操作系统**:
  - Windows 10/11
  - macOS 10.15+
  - Ubuntu 20.04+

### 推荐配置（GPU加速）

- **GPU**: NVIDIA GPU with 6GB+ VRAM (GTX 1660 or better)
- **内存**: 16GB RAM
- **硬盘**: 50GB可用空间
- **CUDA**: 11.8 or 12.1

---

## 安装步骤

### 方式1: 使用pip（推荐）

#### 1. 安装Python

确保安装了Python 3.9或更高版本：

```bash
python --version  # 应该显示 Python 3.9.x 或更高
```

如果没有安装，请从 [python.org](https://www.python.org/downloads/) 下载安装。

#### 2. 克隆仓库

```bash
git clone <repository-url>
cd llm-learning-guide
```

#### 3. 创建虚拟环境

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 4. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. 验证安装

```bash
python scripts/test_environment.py
```

---

### 方式2: 使用Conda

#### 1. 安装Anaconda/Miniconda

从 [Anaconda官网](https://www.anaconda.com/download) 或 [Miniconda官网](https://docs.conda.io/en/latest/miniconda.html) 下载安装。

#### 2. 创建Conda环境

```bash
conda create -n llm-guide python=3.9
conda activate llm-guide
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 验证安装

```bash
python scripts/test_environment.py
```

---

## GPU支持配置

### CUDA安装（NVIDIA GPU用户）

#### 1. 检查GPU

```bash
nvidia-smi
```

如果命令不存在，需要先安装NVIDIA驱动。

#### 2. 安装CUDA Toolkit

访问 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads) 下载并安装CUDA 11.8或12.1。

#### 3. 安装PyTorch GPU版本

卸载CPU版本并安装GPU版本：

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

对于CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 4. 验证GPU可用性

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## 启动Jupyter Lab

安装完成后，启动Jupyter Lab：

```bash
jupyter lab
```

浏览器会自动打开，导航到 `notebooks/Module01_Foundation/` 开始学习。

---

## 常见问题

### Q1: pip安装速度慢

**解决方案**: 使用国内镜像源

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 依赖冲突

**解决方案**: 使用干净的虚拟环境

```bash
# 删除旧环境
rm -rf venv  # Linux/macOS
# 或
rmdir /s venv  # Windows

# 重新创建
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 重新安装
pip install -r requirements.txt
```

### Q3: GPU不可用

**检查步骤**:

1. 确认NVIDIA驱动已安装: `nvidia-smi`
2. 确认CUDA版本匹配
3. 重新安装PyTorch GPU版本
4. 重启系统

### Q4: 内存不足

**解决方案**:

1. 使用更小的batch size
2. 使用梯度累积
3. 使用模型并行
4. 关闭其他程序

### Q5: Jupyter Kernel崩溃

**解决方案**:

```bash
# 重新安装ipykernel
pip install --upgrade ipykernel
python -m ipykernel install --user --name=llm-guide
```

---

## 可选组件安装

### RAG相关（Module 8需要）

```bash
pip install chromadb langchain llama-index
```

### 向量数据库GPU版本

如果有GPU，可以安装faiss-gpu:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 实验追踪

```bash
pip install wandb
wandb login
```

---

## 环境测试脚本

创建 `scripts/test_environment.py` 并运行：

```python
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
        'numpy', 'torch', 'transformers', 'matplotlib',
        'pandas', 'jupyter', 'tqdm'
    ]

    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} - 未安装")
            return False
    return True

def test_torch_cuda():
    """测试PyTorch CUDA支持"""
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available (CPU mode)")

if __name__ == "__main__":
    print("="*60)
    print("环境测试")
    print("="*60)

    test_python_version()
    print()

    if test_imports():
        print("\n✓ 所有必要的包已安装")
    else:
        print("\n✗ 部分包未安装，请运行: pip install -r requirements.txt")
        sys.exit(1)

    print()
    test_torch_cuda()

    print("\n" + "="*60)
    print("✓ 环境配置完成！可以开始学习了。")
    print("="*60)
```

运行测试：

```bash
python scripts/test_environment.py
```

---

## 下一步

环境配置完成后：

1. 阅读 [README.md](README.md) 了解项目概览
2. 查看 [CODEBASE.md](CODEBASE.md) 了解内容规范
3. 开始学习 `notebooks/Module01_Foundation/01_math_review.ipynb`

---

## 获取帮助

如果遇到问题：

1. 查看本文档的常见问题部分
2. 查看 [docs/troubleshooting.md](docs/troubleshooting.md)
3. 在GitHub Issues中提问
4. 加入讨论社区

---

**祝学习愉快！** 🚀
