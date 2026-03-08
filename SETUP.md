# 环境搭建指南

本文档用于快速完成大模型学习指南的运行环境准备，并给出 CPU/GPU 两种典型配置路径。

---

## 1. 快速开始（推荐）

### 1.1 前置要求

- Python **3.9+**
- 推荐使用虚拟环境（`venv` 或 `conda`）
- 可选：NVIDIA GPU + CUDA（用于加速训练与推理）

### 1.2 5 步完成安装

```bash
# 1) 克隆仓库
git clone <your-repo-url>
cd llm-learning-guide

# 2) 创建虚拟环境
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3) 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4) 验证环境
python scripts/test_environment.py

# 5) 启动 Jupyter
jupyter lab
```

---

## 2. 系统与硬件建议

### 2.1 最低配置（CPU 学习）

- CPU：4 核以上
- 内存：8GB+
- 硬盘：20GB+
- 操作系统：Windows 10/11、macOS 10.15+、Ubuntu 20.04+

### 2.2 推荐配置（GPU 加速）

- GPU：NVIDIA 6GB+ VRAM（如 GTX 1660 或更高）
- 内存：16GB+
- 硬盘：50GB+
- CUDA：11.8 或 12.1

---

## 3. 安装方式

### 3.1 方式 A：pip + venv（推荐）

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/test_environment.py
```

### 3.2 方式 B：conda

```bash
conda create -n llm-guide python=3.9
conda activate llm-guide
pip install -r requirements.txt
python scripts/test_environment.py
```

---

## 4. GPU 环境配置（可选）

### 4.1 检查 GPU

```bash
nvidia-smi
```

### 4.2 安装 CUDA 与 GPU 版 PyTorch

根据你的 CUDA 版本选择安装源。

```bash
# CUDA 11.8
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4.3 验证 GPU 可用性

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## 5. 数据与可选依赖

### 5.1 下载基础学习数据

```bash
python scripts/download_datasets.py
```

### 5.2 Module 8 可选组件

```bash
pip install chromadb langchain llama-index
```

### 5.3 实验追踪（可选）

```bash
pip install wandb
wandb login
```

---

## 6. 运行入口

### 6.1 启动 Jupyter

```bash
jupyter lab
```

建议从以下路径开始：

```bash
notebooks/Module01_Foundation/01_math_review.ipynb
```

### 6.2 仅运行脚本验证

```bash
python scripts/test_environment.py
```

---

## 7. 平台兼容性

### 7.1 Intel Mac (x86_64) 重要限制

PyTorch **2.3+** 已停止为 Intel Mac 提供 PyPI 预编译包，最高可用版本为 **2.2.2**。
由此产生以下连锁版本约束：

| 包 | Intel Mac 约束 | 原因 |
|---|---|---|
| `torch` | <= 2.2.2 | PyPI 无更高版本 wheel |
| `numpy` | < 2.0.0 | torch 2.2.x 使用 NumPy 1.x ABI 编译，运行时会崩溃 |
| `transformers` | < 5.0.0 | 5.x 硬性要求 torch >= 2.4 |

**实际可用版本组合（已验证）**：

```
torch==2.2.2
numpy==1.26.4
transformers==4.57.6
```

> Apple Silicon Mac (M1/M2/M3/M4) 和 Linux/Windows 用户不受此限制，
> 可直接安装最新版 PyTorch。

### 7.2 安装后验证

```bash
source .venv/bin/activate
python -c "
import torch; print('torch:', torch.__version__)
import numpy; print('numpy:', numpy.__version__)
import transformers; print('transformers:', transformers.__version__)
x = torch.randn(3, 3)
print('tensor test:', x.shape, '- OK')
"
```

如果看到 `_ARRAY_API not found` 或 `NumPy 2.x cannot be run` 错误，
说明 NumPy 版本过高，需降级：

```bash
pip install "numpy<2"
```

---

## 8. 常见问题

### Q1: 安装很慢怎么办？

使用国内镜像源安装：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 依赖冲突怎么办？

删除旧环境后重建：

```bash
rm -rf venv                      # Windows 可手动删除目录
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Q3: `python scripts/test_environment.py` 报错怎么办？

优先检查：

1. Python 版本是否为 3.9+
2. 虚拟环境是否已激活
3. 依赖是否安装完整（重新执行 `pip install -r requirements.txt`）

### Q4: GPU 不可用怎么办？

依次检查：

1. `nvidia-smi` 是否正常
2. CUDA 与 PyTorch 版本是否匹配
3. 是否误装 CPU 版 PyTorch

### Q5: Jupyter Kernel 崩溃怎么办？

```bash
pip install --upgrade ipykernel
python -m ipykernel install --user --name=llm-guide
```

### Q6: 内存不足怎么办？

- 降低 batch size
- 使用梯度累积
- 关闭不必要程序
- 优先运行基础模块（Module 1-3）

---

## 9. 维护建议

- 每次拉取新代码后，先执行：
  - `pip install -r requirements.txt`
  - `python scripts/test_environment.py`
- 保持 `requirements.txt` 与环境一致，避免在 notebook 内临时安装依赖。

---

如需进一步排查，请在 Issue 中附上：Python 版本、操作系统、错误日志和执行命令。
