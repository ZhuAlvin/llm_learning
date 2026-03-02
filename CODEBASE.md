# 内容创建规范 (CODEBASE)

本文档是大模型学习指南的内容创建标准，用于指导后续内容的补充和更新。

> 📊 **项目状态**：29/29 notebooks 已完成 (100%) | 平均质量评分 93.7/100 ⭐⭐⭐⭐⭐

---

## 目录

1. [快速参考](#快速参考)
2. [文件命名规范](#文件命名规范)
3. [Notebook 结构规范](#notebook-结构规范)
4. [代码规范](#代码规范)
5. [数学推导规范](#数学推导规范)
6. [README 文档规范](#readme-文档规范)
7. [学习指南文档规范](#学习指南文档规范)
8. [质量检查清单](#质量检查清单)
9. [创建新内容的步骤](#创建新内容的步骤)
10. [技术栈使用指南](#技术栈使用指南)
11. [项目状态与质量](#项目状态与质量)
12. [项目结构](#项目结构)
13. [常见问题](#常见问题)
14. [参考资源](#参考资源)

---

## 快速参考

### 核心原则

1. ✅ **理论完整且有深度**：主线完整推导 + 附录深度证明
2. ✅ **实践紧密嵌入**：微型实践立即验证理论
3. ✅ **渐进式技术栈**：NumPy → PyTorch → Transformers
4. ✅ **统一结构**：所有Notebook遵循相同模板
5. ✅ **质量保证**：通过完整的检查清单
6. ✅ **模块间协调**：避免内容重复，形成"概览→深入"递进关系（见下方原则）
7. ✅ **英文缩写规范**：所有英文缩写在首次出现时必须给出完整表达（见下方规范）
8. ✅ **问题驱动叙事**：优先回答“为什么学、解决什么问题、如何落地”
9. ✅ **统一业务主线**：模块说明需映射到同一真实场景（当前主线：电商客服智能助理）
10. ✅ **读者导向交付**：每个模块必须定义可交付产出和最低完成标准

### 文档叙事基线（2026-03-02）

从 **2026-03-02** 起，模块 README 采用以下强制叙事结构：

1. **问题导向概览**：用 2-3 个真实问题定义模块价值，而非仅列技术名词
2. **业务主线映射**：每个 notebook 至少 1 条“业务问题映射”
3. **学习曲线说明**：明确分段（如“先理解 -> 再实现 -> 再优化”）
4. **固定阅读顺序**：采用统一链路  
   `问题 -> 直觉 -> 最小实现 -> 公式 -> 评估 -> 取舍`
5. **可交付产出**：模块级 2-3 个可验证产出
6. **最低完成标准**：每条学习路径给出可检查的验收标准

> 目标：降低“技术堆砌感”，提升学习闭环和真实场景迁移能力。

### 模块间内容安排原则

**基线标准**（2026-02-27 确立）：

#### 原则 1：基础模块"广而浅"，专题模块"窄而深"

| 模块类型 | 定位 | 内容深度 | 示例 |
|---------|------|---------|------|
| **基础模块**（Module 1） | 架构概览 | 点到为止，建立直觉 | Module 1.5 介绍 RNN/CNN/Transformer 核心思想 |
| **专题模块**（Module 2+） | 深入实现 | 完整推导 + 从零实现 | Module 2.1 详细推导 LSTM 四个门 + NumPy 实现 |

#### 原则 2：避免内容重复，明确指向关系

**反例**（已修正）：
- ❌ Module 1.5 详细推导 LSTM 四个门 + Module 2.1 再次详细推导
- ❌ 两个模块都有完整的梯度消失数学证明

**正例**（当前标准）：
- ✅ Module 1.5：简述 LSTM 核心思想（"门控机制"），明确指向 Module 2.1
- ✅ Module 2.1：完整推导 LSTM 四个门、从零实现、可视化

**指向语句模板**：
```markdown
> **详细内容**：[主题]的完整推导、从零实现、可视化见 **Module X.Y: [标题]**
```

#### 原则 3：学习曲线平滑过渡

**基础模块的内容结构**（以 RNN 为例）：

```
1. 为什么需要？（动机，1-2 段）
   └─ 前馈网络的局限 + 具体例子

2. 核心思想是什么？（概念，1-2 段）
   └─ 基本公式 + 展开图 + 关键特性

3. 有什么问题？（挑战，1 段）
   └─ 长程依赖问题 + 简单例子

4. 如何解决？（方案，1 段）
   └─ LSTM/GRU 核心思想（不展开公式）

5. 实际应用？（应用，1 段）
   └─ 代表应用 + 局限性 + 演进方向
```

**篇幅控制**：
- 基础模块单个架构：300-500 词
- 专题模块单个架构：1500-3000 词 + 代码实现

#### 原则 4：内容分层标准

| 层级 | 包含内容 | 不包含内容 |
|------|---------|-----------|
| **概览级**（基础模块） | 动机、基本公式、核心思想、应用场景 | 完整推导、门控机制详解、从零实现 |
| **深入级**（专题模块） | 完整数学推导、逐步实现、可视化、性能对比 | 其他架构的概览（专注单一主题） |

#### 原则 5：检查清单

创建或修改内容时，检查：

- [ ] 是否与其他模块有 >30% 的内容重复？
  - 若是，精简为概览级别，添加指向语句
- [ ] 基础模块是否过于详细（>500 词/架构）？
  - 若是，删除详细推导，保留核心思想
- [ ] 专题模块是否过于简略（<1000 词）？
  - 若是，补充完整推导和实现
- [ ] 是否有明确的"概览→深入"指向关系？
  - 若无，添加指向语句

---

### 英文缩写规范

**基线标准**（2026-02-27 确立）：

#### 规则：首次出现必须给出完整表达

所有英文缩写在文档中**首次出现**时，必须按以下格式给出完整表达：

**格式模板**：
```markdown
缩写 (Full English Name, 中文翻译)
```

**示例**：

| 缩写 | 首次出现格式 | 后续使用 |
|------|------------|---------|
| MLP | MLP (Multi-Layer Perceptron, 多层感知机) | MLP |
| CNN | CNN (Convolutional Neural Network, 卷积神经网络) | CNN |
| RNN | RNN (Recurrent Neural Network, 循环神经网络) | RNN |
| LSTM | LSTM (Long Short-Term Memory, 长短期记忆网络) | LSTM |
| GRU | GRU (Gated Recurrent Unit, 门控循环单元) | GRU |
| SGD | SGD (Stochastic Gradient Descent, 随机梯度下降) | SGD |
| Adam | Adam (Adaptive Moment Estimation) | Adam |
| BN | BN (Batch Normalization, 批归一化) | BN 或 Batch Normalization |
| ReLU | ReLU (Rectified Linear Unit, 修正线性单元) | ReLU |
| SVM | SVM (Support Vector Machine, 支持向量机) | SVM |
| DBN | DBN (Deep Belief Network, 深度信念网络) | DBN |

#### 适用范围

1. **Notebook 文档**：每个 `.ipynb` 文件独立检查，首次出现时给出完整表达
2. **README 文档**：每个 `README.md` 文件独立检查
3. **LEARNING_GUIDE 文档**：每个 `LEARNING_GUIDE.md` 文件独立检查
4. **跨文件不继承**：即使在前一个文件中已经给出完整表达，在新文件中首次出现时仍需重复

#### 例外情况

以下情况可以不给出完整表达：
- 极其常见的缩写：API, URL, GPU, CPU, RAM
- 编程语言名称：Python, JavaScript, C++
- 框架名称：PyTorch, TensorFlow, NumPy
- 数学符号：sin, cos, log, exp

#### 检查清单

创建或修改内容时，检查：

- [ ] 是否所有英文缩写在首次出现时都给出了完整表达？
- [ ] 格式是否符合 `缩写 (Full English Name, 中文翻译)` 模板？
- [ ] 是否有遗漏的缩写（使用正则表达式 `\b[A-Z]{2,}\b` 检查）？

---

### Notebook标准结构

所有 Notebook 遵循统一的 8 部分结构：

```
1. 📚 本章概览 (Overview)
   - 学习目标（3-5 个明确目标）
   - 核心问题（要解决的关键问题）
   - 知识地图（本章在整体课程中的位置）
   - 预计学习时间

2. 🎯 动机与背景 (Motivation)
   - 为什么需要这个技术？
   - 要解决什么实际问题？
   - 🔬 微型实践：问题演示
   - 历史发展脉络

3. 📖 理论基础 (Theory)
   - 核心概念讲解（清晰定义）
   - 数学原理推导（主线完整推导）
   - 🔬 微型实践：概念验证
   - 可视化与直觉理解
   - 📚 扩展阅读：深度推导（链接到附录）

4. 🔨 从零实现 (Implementation from Scratch)
   - NumPy/纯 Python 实现
   - 详细代码注释（英文）
   - 🔬 微型实践：单元测试
   - 性能分析与局限性讨论

5. ⚙️ 工程化实现 (Engineering Implementation)
   - PyTorch 实现
   - 优化技巧与最佳实践
   - 🔬 微型实践：性能对比
   - 与标准库（Transformers）对比

6. 🚀 综合项目 (Capstone Project)
   - 项目概述与需求
   - 数据准备与预处理
   - 基础实现（必做）
   - 结果分析与可视化
   - 进阶挑战（选做，2-3 个）
   - 扩展方向建议

7. ❓ 常见问题与调试 (FAQ & Debugging)
   - 典型错误与解决方案（5-8 个）
   - 调试技巧与工具
   - 性能优化建议

8. 📝 总结与展望 (Summary)
   - 核心要点回顾（3-5 个要点）
   - 与其他技术的联系
   - 💡 思考题（3-5 个开放性问题）
   - 下一步学习建议
```

## 代码规范

### 通用代码规范

- **语言**：Python 3.8+
- **风格**：遵循 PEP 8
- **注释**：所有代码注释使用英文
- **文档字符串**：使用 Google 风格
- **变量命名**：清晰、有意义、符合 Python 规范
- **类型提示**：推荐使用（可选）

### 微型实践代码块

微型实践是紧跟理论的小型验证实验，用于立即验证刚学到的概念。

**特点**：
- 代码量：10-30 行
- 运行时间：< 10 秒
- 学习时间：5-10 分钟
- 目标：验证一个具体概念

**模板**：

```python
# 🔬 Micro Practice: [简短描述，如 "Verify attention score calculation"]
# Goal: [明确的学习目标，如 "Understand how attention scores are computed"]
# Expected outcome: [预期结果，如 "Scores sum to 1 after softmax"]

import numpy as np
import matplotlib.pyplot as plt

def example_function(param):
    """
    Brief description of what this function does.

    Args:
        param (type): Description of parameter

    Returns:
        type: Description of return value

    Example:
        >>> result = example_function(5)
        >>> print(result)
        10
    """
    # Step 1: Clear explanation of this step
    intermediate = param * 2

    # Step 2: Clear explanation of this step
    result = intermediate + param

    return result

# Test the function
result = example_function(5)
print(f"Result: {result}")

# Visualize if applicable
plt.figure(figsize=(8, 4))
plt.plot(result)
plt.title("Visualization of Result")
plt.show()
```

### 综合项目代码块

综合项目整合模块所有知识，是完整的实践项目。

**特点**：
- 代码量：100-300 行
- 运行时间：5-30 分钟
- 学习时间：1-3 小时
- 目标：整合模块所有知识

**模板**：

```python
# 🚀 Capstone Project: [项目名称]
# Description: [项目描述]
# Integrates: [整合的所有概念]

import torch
import torch.nn as nn
from typing import Optional, Tuple

class MainImplementation(nn.Module):
    """
    Complete implementation with extensive documentation.

    This class implements [功能描述] using [技术栈].

    Args:
        config (dict): Configuration dictionary containing:
            - param1 (int): Description
            - param2 (float): Description

    Attributes:
        attr1: Description
        attr2: Description

    Example:
        >>> config = {'param1': 10, 'param2': 0.1}
        >>> model = MainImplementation(config)
        >>> output = model(input_tensor)
        >>> print(output.shape)
        torch.Size([batch_size, seq_len, hidden_dim])
    """

    def __init__(self, config: dict):
        """Initialize the model with given configuration."""
        super().__init__()

        # Extract configuration
        self.param1 = config['param1']
        self.param2 = config['param2']

        # Initialize layers
        self.layer1 = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.layer2 = nn.Linear(config['hidden_dim'], config['output_dim'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Step 1: First transformation
        hidden = self.layer1(x)  # (batch_size, seq_len, hidden_dim)
        hidden = torch.relu(hidden)

        # Step 2: Second transformation
        output = self.layer2(hidden)  # (batch_size, seq_len, output_dim)

        return output

    def training_step(self, batch: dict) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch (dict): Batch containing 'input' and 'target'

        Returns:
            torch.Tensor: Loss value
        """
        # Forward pass
        output = self.forward(batch['input'])

        # Compute loss
        loss = nn.functional.mse_loss(output, batch['target'])

        return loss

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'input_dim': 512,
        'hidden_dim': 256,
        'output_dim': 128,
        'param1': 10,
        'param2': 0.1
    }

    # Create model
    model = MainImplementation(config)

    # Test forward pass
    batch_size, seq_len = 32, 100
    x = torch.randn(batch_size, seq_len, config['input_dim'])
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

### 数学推导规范

#### 主线推导（Notebook中）

- 完整的推导步骤，不跳步
- 每步都有文字解释
- 配合代码验证
- 提供直觉理解

示例：
```markdown
### 3.2 数学推导

**步骤1：线性变换**

$$Q = XW_q$$

这一步将输入映射到Query空间...

🔬 **微型实践：验证线性变换**
[代码]

**步骤2：计算注意力分数**

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$$

为什么要除以 $\sqrt{d_k}$？让我们通过实验理解...

📚 **扩展阅读：** 完整证明见 [附录A.2](../appendix/A_matrix_calculus.ipynb#A.2)
```

#### 附录推导（appendix/中）

- 严格的数学符号
- 定理-证明结构
- 多种推导方法
- 理论性质讨论

---

## 文件命名规范

### Notebook命名

格式：`XX_topic_name.ipynb`

- `XX`：章节编号（01-03）
- `topic_name`：主题名称（小写+下划线）

示例：
- `01_self_attention.ipynb`
- `02_parameter_efficient_finetuning.ipynb`
- `03_research_frontiers.ipynb`

### 学习指南文件命名

格式：`LEARNING_GUIDE.md`

- 所有模块的学习指南文件统一命名为 `LEARNING_GUIDE.md`
- 放置在对应的模块目录下

示例：
- `notebooks/Module01_Foundation/LEARNING_GUIDE.md`
- `notebooks/Module02_Evolution/LEARNING_GUIDE.md`

### 模块目录命名

格式：`ModuleXX_ModuleName`

- `XX`：模块编号（01-09）
- `ModuleName`：模块名称（PascalCase）

实际使用的模块目录：
- `Module01_Foundation` - 基础知识
- `Module02_Evolution` - 模型演进
- `Module03_Transformer` - Transformer 架构
- `Module04_PreTraining` - 预训练（注意：大写 T）
- `Module05_FineTuning` - 微调技术（注意：驼峰命名）
- `Module06_AdvancedTraining` - 高级训练
- `Module07_Deployment` - 部署与优化
- `Module08_Applications` - 实际应用
- `Module09_Frontiers` - 前沿探索

### 代码文件命名

- Python文件：`snake_case.py`
- 类名：`PascalCase`
- 函数名：`snake_case`
- 常量：`UPPER_CASE`

---

## 质量检查清单

### ✅ 理论部分

- [ ] 概念清晰，定义明确
- [ ] 逻辑连贯，章节流畅
- [ ] 数学正确，符号一致
- [ ] 有直觉解释和可视化
- [ ] 附录链接正确

### ✅ 代码部分

- [ ] 所有单元格可按顺序运行
- [ ] 输出结果符合预期
- [ ] 代码有详细英文注释
- [ ] 变量命名清晰有意义
- [ ] 运行时间合理

### ✅ 实践部分

- [ ] 微型实践紧跟理论
- [ ] 综合项目完整可行
- [ ] 提供多个难度级别
- [ ] 有明确评估标准
- [ ] 提供参考实现

### ✅ 整体结构

- [ ] 严格遵循统一模板
- [ ] 学习时间估算准确
- [ ] 与前后章节衔接良好
- [ ] 所有链接有效
- [ ] 格式规范统一

### ✅ 文档叙事（2026-03-02 新增）

- [ ] 模块概览是否以“真实问题”开场（而非纯技术清单）？
- [ ] 是否明确映射统一业务主线（当前：电商客服智能助理）？
- [ ] 是否包含“可交付产出”（2-3 个）？
- [ ] 是否包含“学习曲线设计”与“每章建议阅读顺序”？
- [ ] 每个 notebook 描述是否包含至少 1 条业务问题映射？
- [ ] 每条学习路径是否包含“最低完成标准”？

---

## 创建新内容的步骤

### Step 1: 规划

1. 明确模块要回答的核心问题（2-3 个）
2. 映射统一业务主线（当前：电商客服智能助理）
3. 确定学习目标（3-6 个）
4. 定义可交付产出与最低完成标准
5. 列出核心概念与实践项目
6. 估算学习时间与学习曲线分段

### Step 2: 创建文件

```bash
# 使用模板创建
cp templates/notebook_template.ipynb \
   notebooks/ModuleXX_YY_topic_name.ipynb
```

### Step 3: 填充内容

按模板结构依次填充各部分，优先保证：

1. 问题导向叙事（先问题，再技术）
2. 每个 notebook 的业务问题映射
3. 每条学习路径的最低完成标准

### Step 4: 自我审查

使用质量检查清单进行审查

### Step 5: 测试运行

```bash
# 清除输出并重新运行
jupyter nbconvert --clear-output --inplace notebook.ipynb
jupyter nbconvert --execute --inplace notebook.ipynb
```

### Step 6: 提交

```bash
git add notebooks/ModuleXX_YY_topic_name.ipynb
git commit -m "Add: ModuleXX_YY topic_name"
git push
```

---

## 技术栈使用指南

### 渐进式技术栈策略

本项目采用渐进式技术栈，从底层实现逐步过渡到工程化工具：

```
Module 1-2: NumPy 为主
    ↓
Module 3-4: NumPy + PyTorch
    ↓
Module 5-6: PyTorch + Transformers
    ↓
Module 7-9: 完整工具链
```

### Module 1-2: NumPy 为主

**目标**：理解底层机制，掌握数学原理

**技术栈**：
- NumPy - 数值计算
- Matplotlib - 可视化
- 纯 Python - 算法实现

**示例**：

```python
import numpy as np

# 纯 NumPy 实现，理解底层机制
def attention_numpy(Q, K, V):
    """
    Attention mechanism using pure NumPy.

    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)

    Returns:
        Output matrix (seq_len, d_v)
    """
    # Step 1: Compute attention scores
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # Step 2: Apply softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 3: Weighted sum of values
    output = weights @ V

    return output, weights
```

### Module 3-4: NumPy + PyTorch

**目标**：学习工程化实现，理解自动微分

**技术栈**：
- PyTorch - 深度学习框架
- torch.nn - 神经网络模块
- torch.optim - 优化器
- NumPy - 对比验证

**示例**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 实现，学习工程化
class Attention(nn.Module):
    """
    Attention mechanism using PyTorch.

    Args:
        d_model (int): Model dimension
        dropout (float): Dropout rate
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Learnable projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.matmul(weights, V)

        return output, weights
```

### Module 5-6: PyTorch + Transformers

**目标**：使用标准库，学习最佳实践

**技术栈**：
- Transformers - Hugging Face 模型库
- Datasets - 数据集处理
- PEFT - 参数高效微调
- Accelerate - 分布式训练

**示例**：

```python
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
import torch

# 使用标准库
def load_pretrained_model(model_name: str):
    """
    Load pretrained model with LoRA.

    Args:
        model_name: Model identifier from Hugging Face

    Returns:
        model, tokenizer
    """
    # Load base model
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    return model, tokenizer

# Example usage
model, tokenizer = load_pretrained_model("bert-base-uncased")
print(f"Trainable parameters: {model.print_trainable_parameters()}")
```

### Module 7-9: 完整工具链

**目标**：生产级应用，掌握完整工具链

**技术栈**：
- **推理加速**：vLLM, TGI, ONNX Runtime
- **向量检索**：FAISS, Milvus, Pinecone
- **Web 服务**：FastAPI, Flask
- **前端**：Streamlit, Gradio, React
- **监控**：Prometheus, Grafana
- **部署**：Docker, Kubernetes

**示例**：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import faiss
import numpy as np

# FastAPI application
app = FastAPI(title="LLM Service")

# Initialize LLM
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Initialize vector index
dimension = 768
index = faiss.IndexFlatL2(dimension)

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 100
    temperature: float = 0.7

class QueryResponse(BaseModel):
    response: str
    retrieved_docs: list

@app.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):
    """
    Query the LLM with RAG.

    Args:
        request: Query request with parameters

    Returns:
        Response with generated text and retrieved documents
    """
    try:
        # Step 1: Retrieve relevant documents
        query_embedding = get_embedding(request.query)
        D, I = index.search(query_embedding, k=5)
        retrieved_docs = get_documents(I[0])

        # Step 2: Construct prompt
        context = "\n".join(retrieved_docs)
        prompt = f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:"

        # Step 3: Generate response
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        return QueryResponse(
            response=response,
            retrieved_docs=retrieved_docs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 技术栈选择原则

1. **Module 1-2**：只用 NumPy，深入理解原理
2. **Module 3-4**：引入 PyTorch，学习自动微分和模块化
3. **Module 5-6**：使用 Transformers，学习标准实践
4. **Module 7-9**：完整工具链，构建生产应用

### 依赖管理

所有依赖在 `requirements.txt` 中管理：

```bash
# 安装所有依赖
pip install -r requirements.txt

# 测试环境
python scripts/test_environment.py
```

---

## README 文档规范

### 模块 README 结构

每个模块应包含完整的 README.md 文档（建议 220+ 行），并强制遵循“问题驱动 + 业务映射 + 可交付验收”结构。

推荐参考：Module 1、2、3、4、5、6、7、8、9 的最新 README（2026-03-02 后版本）。

```markdown
# Module X: 模块名称 (English Name)

## 📚 模块概览
- 问题导向简介（2-3 段）
- 3 个核心业务问题（示例：效果、成本、稳定性）
- 统一业务主线映射（当前：电商客服智能助理）

## 🎯 学习目标
- 3-6 条能力目标

## ✅ 完成本模块后的可交付产出
- 2-3 个可验证产出（模型、报告、评估结果等）

## ⏱️ 预计学习时间
- 总学习时长范围

## 📈 学习曲线设计
- 分段说明（如：基础理解 -> 工程实现 -> 优化与扩展）

## 🧭 每章建议阅读顺序
`问题 -> 直觉 -> 最小实现 -> 公式 -> 评估 -> 取舍`

## 📖 Notebooks（详细介绍）

### X.1 Notebook 标题 (filename.ipynb)
**评分**: XX/100 ⭐⭐⭐⭐⭐

**核心内容**：
- 主要知识点列表（5-8 个）
- 业务问题映射（至少 1 条）

**N 个微实践**：
1. 实践 1 - 简短描述
2. 实践 2 - 简短描述
...

**关键技术**：
- 技术栈列表

**适用场景**：
- 应用场景列表

---

（重复每个 notebook）

## 🗺️ 学习路径

### 路径 1：角色名称（推荐新手）
```
流程图展示学习顺序
```
**时间**: X-Y 小时
**产出**: 具体成果
**最低完成标准**: 可检查的验收条件

---

（2-3 条不同路径）

## 💡 实践项目建议

### 项目 1：项目名称
**难度**: ⭐⭐⭐
**时间**: X-Y 天

**功能**：
- 功能列表

**技术栈**：
- 技术列表

**学习重点**：
- 重点列表

---

（2-3 个项目）

## 🧠 知识图谱
```
模块知识结构的树形图
```

## 📚 相关资源

### 论文
- [论文标题](链接) (年份)

### 开源项目
- [项目名](链接) - 简短描述

### 工具和库
- [工具名](链接) - 简短描述

## ❓ 常见问题

### Q1: 问题？
**A**: 回答...

---

（5-8 个问题）

## ✅ 学习检查清单

### 主题 1
- [ ] 检查项 1
- [ ] 检查项 2
...

## 📊 模块质量

根据详细质量报告，Module X 的整体评分为 **XX/100** 🏆

### 各 Notebook 评分
| Notebook | 评分 | 状态 |
|----------|------|------|
| ... | ... | ... |

### 优势
- ✅ 优势 1
- ✅ 优势 2

### 改进空间
- 改进建议 1
- 改进建议 2

## 🎯 下一步

完成 Module X 后，你已经掌握了：
- ✅ 技能 1
- ✅ 技能 2

**继续学习**：
- **Module Y** - 描述
- **实践项目** - 描述

---

**模块完成日期**: YYYY-MM-DD
**质量评估**: XX/100 ⭐⭐⭐⭐⭐
**推荐指数**: ⭐⭐⭐⭐⭐
```

### README 创建清单

- [ ] 模块概览是否以“真实问题”开场
- [ ] 是否明确映射统一业务主线
- [ ] 学习目标清晰（3-6 条）
- [ ] 可交付产出明确（2-3 条）
- [ ] 学习曲线设计清晰（分阶段）
- [ ] 每章建议阅读顺序已提供
- [ ] 每个 notebook 有详细介绍
- [ ] 每个 notebook 包含业务问题映射
- [ ] 至少 2-3 条学习路径
- [ ] 每条路径有最低完成标准
- [ ] 至少 2-3 个实践项目
- [ ] 知识图谱清晰
- [ ] 相关资源丰富（论文、项目、工具）
- [ ] 5-8 个常见问题
- [ ] 学习检查清单完整
- [ ] 包含质量评分（如有）
- [ ] 总长度建议 220+ 行（按内容复杂度可增加）

---

## 学习指南文档规范

### 学习指南结构

每个模块应包含完整的 LEARNING_GUIDE.md 文档（400-500+ 行），提供详细的学习指导：

```markdown
# Module X: 模块名称 - 学习指南

## 📋 文档质量检查报告

### ✅ 已完成内容

**Notebook 01_filename.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和理论
- ✅ 完整的理论讲解
- ✅ N 个 Micro Practice 实践练习
- ✅ 可视化
- ✅ 完整实现
- ✅ 工程实践
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

**Notebook 02_filename.ipynb** - 完整且高质量
...

### 📊 质量评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **内容完整性** | X/10 | 说明 |
| **理论深度** | X/10 | 说明 |
| **代码质量** | X/10 | 说明 |
| **实践练习** | X/10 | 说明 |
| **可视化** | X/10 | 说明 |
| **工程实践** | X/10 | 说明 |

**总体评分：X.X/10** - 评价

---

## 🎯 学习指南

### 学习路径

```
第1周：主题名称
  ├─ 理解概念
  ├─ 掌握技术
  ├─ 学习方法
  └─ 实现功能

第2周：主题名称
...

第3周：主题名称
...

第4周：综合项目
  ├─ 选择方向
  ├─ 实现想法
  ├─ 实验和评估
  └─ 撰写报告
```

### 前置知识检查

在开始学习前，确保你已经掌握：

- [ ] 知识点 1
- [ ] 知识点 2
...

### 学习建议

#### 1. 理论学习（40%时间）

**必读材料**：
- 📄 [论文标题](链接) - 简短描述
...

**学习方法**：
1. 学习步骤
2. 学习步骤
...

#### 2. 代码实践（50%时间）

**实践步骤**：

**Week 1: 主题名称**
```python
# 练习1：实现功能
class ClassName:
    def __init__(self):
        pass
    
    def forward(self, x):
        pass

# 练习2：实现功能
...
```

**Week 2: 主题名称**
...

**Week 3: 主题名称**
...

**Week 4: 完整项目**
...

#### 3. 项目实战（10%时间）

**推荐项目**：

1. **项目名称**（难度）
   - 实现步骤
   - 目标

2. **项目名称**（难度）
...

### 常见问题解答

#### Q1: 问题？

**A**: 回答...

#### Q2: 问题？

**A**: 回答...

### 调试技巧

#### 1. 问题类型

```python
# 调试代码
def debug_function():
    # 监控代码
    pass
```

#### 2. 问题类型

...

### 性能优化

#### 1. 优化类型

```python
# 优化代码
def optimized_function():
    pass
```

#### 2. 优化类型

...

### 扩展阅读

#### 进阶主题

1. **主题**
   - 描述

2. **主题**
...

#### 推荐资源

**视频课程**：
- 课程名称

**代码库**：
- 代码库名称: 描述

**论文列表**：
- 论文标题 (年份)

### 评估标准

完成本模块后，你应该能够：

- [ ] 技能 1
- [ ] 技能 2
...

### 下一步

完成 Module X 后，建议：

1. **巩固基础**：...
2. **项目实践**：...
3. **阅读论文**：...
4. **参与研究**：...

---

## 📝 学习检查清单

### Week 1: 主题
- [ ] 检查项 1
- [ ] 检查项 2
...

### Week 2: 主题
...

### Week 3: 主题
...

### Week 4: 实战
...

---

**祝学习顺利！** 🚀

如有问题，请参考：
- 📖 Notebook 中的 FAQ 部分
- 💬 课程讨论区
- 🔍 Stack Overflow
- 📧 联系助教
```

### 学习指南创建清单

- [ ] 文档质量检查报告完整
- [ ] 学习路径清晰（4周计划）
- [ ] 前置知识检查完整
- [ ] 学习建议详细（理论+实践+项目）
- [ ] 代码实践包含所有练习
- [ ] 常见问题解答（5-8个问题）
- [ ] 调试技巧实用
- [ ] 性能优化有效
- [ ] 扩展阅读丰富
- [ ] 评估标准明确
- [ ] 学习检查清单完整
- [ ] 总长度 400-500+ 行

---

## 项目状态与质量

### 完成情况

**总计**: 29/29 notebooks (100%) ✅

| 模块 | Notebooks | 平均分 | 状态 |
|------|-----------|--------|------|
| Module 1: Foundation | 5 个 | 待评估 | ✅ |
| Module 2: Evolution | 3 个 | 91/100 | ✅ |
| Module 3: Transformer | 3 个 | 待评估 | ✅ |
| Module 4: PreTraining | 3 个 | 91+/100 | ✅ |
| Module 5: FineTuning | 3 个 | 93+/100 | ✅ 🏆 |
| Module 6: AdvancedTraining | 3 个 | 95/100 | ✅ 🏆 |
| Module 7: Deployment | 3 个 | 92/100 | ✅ |
| Module 8: Applications | 3 个 | 95.3/100 | ✅ 🏆 |
| Module 9: Frontiers | 3 个 | 96.0/100 | ✅ 🏆 |

**已评估平均分**: 93.7/100 ⭐⭐⭐⭐⭐

### 文档完成情况

**README 文档** (9/9 完成):
- ✅ Module01_Foundation/README.md
- ✅ Module02_Evolution/README.md
- ✅ Module03_Transformer/README.md
- ✅ Module04_PreTraining/README.md
- ✅ Module05_FineTuning/README.md
- ✅ Module06_AdvancedTraining/README.md
- ✅ Module07_Deployment/README.md
- ✅ Module08_Applications/README.md
- ✅ Module09_Frontiers/README.md

**学习指南文档** (9/9 完成):
- ✅ Module01_Foundation/LEARNING_GUIDE.md
- ✅ Module02_Evolution/LEARNING_GUIDE.md
- ✅ Module03_Transformer/LEARNING_GUIDE.md
- ✅ Module04_PreTraining/LEARNING_GUIDE.md
- ✅ Module05_FineTuning/LEARNING_GUIDE.md
- ✅ Module06_AdvancedTraining/LEARNING_GUIDE.md
- ✅ Module07_Deployment/LEARNING_GUIDE.md
- ✅ Module08_Applications/LEARNING_GUIDE.md
- ✅ Module09_Frontiers/LEARNING_GUIDE.md

### 质量报告

已完成的质量报告文档：
- ✅ module02_quality_report.md
- ✅ module04_quality_report.md
- ✅ module05_quality_report.md
- ✅ module05_notebook03_improvement_report.md
- ✅ module05_summary.md
- ✅ module06_advanced_optimization.md
- ✅ module06-09_quality_report.md
- ✅ module08-09_detailed_report.md
- ✅ project_completion_report.md

---

## 常见问题

### Q1: 内容放主线还是附录？

**判断标准：**
- **主线**：理解核心概念必需的内容
  - 基本定义和概念
  - 核心算法推导
  - 关键实现步骤
  - 必要的数学证明
- **附录**：深入理解或特殊情况
  - 严格的数学证明
  - 多种推导方法对比
  - 边界情况讨论
  - 理论性质分析

**示例**：
- 主线：Attention 的基本计算公式和推导
- 附录：Attention 的梯度推导、复杂度分析、变体对比

### Q2: 微型实践应该多小？

**指导原则：**
- **代码量**：10-30 行
- **运行时间**：< 10 秒
- **学习时间**：5-10 分钟
- **目标**：验证一个具体概念

**好的微型实践**：
```python
# ✅ 验证 softmax 的数值稳定性
scores = np.array([1000, 1001, 1002])
stable_softmax = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
```

**不好的微型实践**：
```python
# ❌ 太复杂，包含多个概念
# 实现完整的 Transformer 层（应该是综合项目）
```

### Q3: 综合项目应该多大？

**指导原则：**
- **代码量**：100-300 行
- **运行时间**：5-30 分钟
- **学习时间**：1-3 小时
- **目标**：整合模块所有知识

**项目范围**：
- 包含数据准备、模型实现、训练、评估
- 有基础版本（必做）和进阶版本（选做）
- 提供完整的代码和详细注释
- 包含结果分析和可视化

### Q4: 如何创建模块 README？

**步骤：**
1. 参考高质量示例：Module 1-9 的最新 README（2026-03-02 后版本）
2. 使用 [README 文档规范](#readme-文档规范) 模板
3. 确保包含所有必需部分（10+ 个部分）
4. 模块概览先写“真实问题”，再写技术内容
5. 每个 notebook 添加业务问题映射（至少 1 条）
6. 提供 2-3 条学习路径，并补全最低完成标准
7. 包含 2-3 个实践项目建议
8. 总长度建议 220+ 行（按复杂度调整）
9. 运行质量检查清单

**必需部分**：
- 模块概览
- 学习目标
- 可交付产出
- 学习曲线设计
- 每章建议阅读顺序
- Notebooks 详细介绍
- 学习路径
- 实践项目建议
- 知识图谱
- 相关资源
- 常见问题
- 学习检查清单
- 模块质量评分
- 下一步建议

### Q5: 如何创建学习指南？

**步骤：**
1. 参考已完成的 LEARNING_GUIDE.md 格式
2. 使用 [学习指南文档规范](#学习指南文档规范) 模板
3. 确保包含所有必需部分（12+ 个部分）
4. 提供 4 周学习计划
5. 包含代码实践示例
6. 总长度达到 400-500+ 行
7. 运行质量检查清单

**必需部分**：
- 文档质量检查报告
- 学习路径（4 周计划）
- 前置知识检查
- 学习建议（理论+实践+项目）
- 代码实践（每周练习）
- 常见问题解答
- 调试技巧
- 性能优化
- 扩展阅读
- 评估标准
- 学习检查清单

### Q6: Notebook 文件命名规则是什么？

**当前规则**：`XX_topic_name.ipynb`

- `XX`：章节编号（01-03）
- `topic_name`：主题名称（小写+下划线）

**示例**：
- ✅ `01_self_attention.ipynb`
- ✅ `02_transformer_encoder.ipynb`
- ✅ `03_research_frontiers.ipynb`
- ❌ `Module03_01_self_attention.ipynb`（旧格式，不再使用）

### Q7: 模块目录命名有什么规则？

**当前规则**：`ModuleXX_ModuleName`

**实际使用的 9 个模块**：
1. `Module01_Foundation` - 基础知识
2. `Module02_Evolution` - 模型演进
3. `Module03_Transformer` - Transformer 架构
4. `Module04_PreTraining` - 预训练（注意：大写 T）
5. `Module05_FineTuning` - 微调技术（注意：驼峰命名）
6. `Module06_AdvancedTraining` - 高级训练
7. `Module07_Deployment` - 部署与优化
8. `Module08_Applications` - 实际应用
9. `Module09_Frontiers` - 前沿探索

**注意事项**：
- 保持现有命名一致性
- 新增内容遵循相同规则
- 不要随意修改已有目录名

### Q8: 如何确保代码质量？

**代码质量检查清单**：
- [ ] 所有单元格可按顺序运行
- [ ] 输出结果符合预期
- [ ] 代码有详细英文注释
- [ ] 变量命名清晰有意义
- [ ] 运行时间合理（< 30 分钟）
- [ ] 使用类型提示（推荐）
- [ ] 遵循 PEP 8 风格
- [ ] 包含错误处理
- [ ] 有单元测试或验证代码

**测试方法**：
```bash
# 清除输出并重新运行
jupyter nbconvert --clear-output --inplace notebook.ipynb
jupyter nbconvert --execute --inplace notebook.ipynb

# 检查是否有错误
echo $?  # 应该返回 0
```

### Q9: 如何添加新的 Notebook？

**步骤**：
1. 确定模块和主题
2. 按照命名规范创建文件
3. 使用 [Notebook 标准结构](#notebook-标准结构)
4. 填充 8 个部分的内容
5. 添加 5-10 个微型实践
6. 创建 1 个综合项目
7. 运行质量检查清单
8. 测试所有代码
9. 更新模块 README 和 LEARNING_GUIDE
10. 提交代码

### Q10: 项目的质量标准是什么？

**评分维度**（满分 100）：
- **内容完整性**（20 分）：是否包含所有必需部分
- **理论深度**（20 分）：数学推导是否严谨完整
- **代码质量**（20 分）：代码是否清晰、可运行、有注释
- **实践练习**（15 分）：微型实践和综合项目的质量
- **可视化**（10 分）：图表是否清晰、有助于理解
- **工程实践**（15 分）：是否包含最佳实践和优化技巧

**质量等级**：
- **95-100 分**：优秀 ⭐⭐⭐⭐⭐（Module 6, 8, 9）
- **90-94 分**：良好 ⭐⭐⭐⭐（Module 4, 5, 7）
- **85-89 分**：合格 ⭐⭐⭐
- **< 85 分**：需要改进

**当前项目平均分**：93.7/100 ⭐⭐⭐⭐⭐

---

## 项目结构

```
llm-learning-guide/
├── README.md                           # 项目总览
├── CODEBASE.md                         # 内容创建规范（本文档）
├── SETUP.md                            # 环境配置指南
├── requirements.txt                    # Python 依赖
│
├── docs/                               # 文档目录
│   ├── SUMMARY.md                      # 项目总结
│   ├── plans/                          # 设计文档（30+ 个规划文档）
│   └── quality_reports/                # 质量报告（9 个报告文档）
│
├── notebooks/                          # 学习 Notebooks（29 个）
│   ├── Module01_Foundation/            # 基础知识（5 个 notebooks）
│   │   ├── README.md
│   │   ├── LEARNING_GUIDE.md
│   │   ├── 01_math_review.ipynb
│   │   ├── 02_neural_networks_basics.ipynb
│   │   ├── 03_backpropagation.ipynb
│   │   ├── 04_pytorch_basics.ipynb
│   │   └── 05_deep_learning_intro.ipynb
│   │
│   ├── Module02_Evolution/             # 模型演进（3 个 notebooks）
│   │   ├── README.md
│   │   ├── LEARNING_GUIDE.md
│   │   ├── 01_rnn_lstm.ipynb
│   │   ├── 02_attention_mechanism.ipynb
│   │   └── 03_seq2seq.ipynb
│   │
│   ├── Module03_Transformer/           # Transformer 架构（3 个 notebooks）
│   │   ├── README.md
│   │   ├── LEARNING_GUIDE.md
│   │   ├── 01_self_attention.ipynb
│   │   ├── 02_transformer_encoder.ipynb
│   │   └── 03_transformer_decoder.ipynb
│   │
│   ├── Module04_PreTraining/           # 预训练（3 个 notebooks）
│   │   ├── README.md
│   │   ├── LEARNING_GUIDE.md
│   │   ├── 01_language_modeling.ipynb
│   │   ├── 02_bert_architecture.ipynb
│   │   └── 03_gpt_architecture.ipynb
│   │
│   ├── Module05_FineTuning/            # 微调技术（3 个 notebooks）🏆
│   │   ├── README.md
│   │   ├── LEARNING_GUIDE.md
│   │   ├── 01_transfer_learning.ipynb
│   │   ├── 02_parameter_efficient_finetuning.ipynb  # ⭐ 100/100
│   │   └── 03_domain_adaptation.ipynb
│   │
│   ├── Module06_AdvancedTraining/      # 高级训练（3 个 notebooks）🏆
│   │   ├── README.md
│   │   ├── LEARNING_GUIDE.md
│   │   ├── 01_advanced_optimization.ipynb
│   │   ├── 02_distributed_training.ipynb
│   │   └── 03_efficient_training.ipynb
│   │
│   ├── Module07_Deployment/            # 部署与优化（3 个 notebooks）
│   │   ├── README.md
│   │   ├── LEARNING_GUIDE.md
│   │   ├── 01_inference_optimization.ipynb
│   │   ├── 02_model_serving.ipynb
│   │   └── 03_production_best_practices.ipynb
│   │
│   ├── Module08_Applications/          # 实际应用（3 个 notebooks）🏆
│   │   ├── README.md
│   │   ├── LEARNING_GUIDE.md
│   │   ├── 01_rag_systems.ipynb        # ⭐ 97/100
│   │   ├── 02_agent_systems_mcp.ipynb
│   │   └── 03_frontend_integration.ipynb
│   │
│   └── Module09_Frontiers/             # 前沿探索（3 个 notebooks）🏆
│       ├── README.md
│       ├── LEARNING_GUIDE.md
│       ├── 01_emerging_architectures.ipynb
│       ├── 02_advanced_training.ipynb
│       └── 03_research_frontiers.ipynb  # ⭐ 97/100
│
├── src/                                # 可复用代码库
│   ├── __init__.py
│   ├── data/                           # 数据处理模块
│   │   └── __init__.py
│   ├── models/                         # 模型实现
│   │   └── __init__.py
│   ├── inference/                      # 推理工具
│   │   └── __init__.py
│   └── utils/                          # 工具函数
│       └── __init__.py
│
├── scripts/                            # 实用脚本
│   ├── test_environment.py             # 环境测试
│   └── download_datasets.py            # 数据集下载
│
└── datasets/                           # 示例数据集
    └── README.md
```

## 参考资源

### 核心文档
- **项目总览**：[README.md](../README.md)
- **环境配置**：[SETUP.md](../SETUP.md)
- **项目总结**：[docs/SUMMARY.md](docs/SUMMARY.md)
- **完整设计文档**：[docs/plans/](docs/plans/) 目录

### 质量报告
- [Module 2 质量报告](docs/quality_reports/module02_quality_report.md)
- [Module 4 质量报告](docs/quality_reports/module04_quality_report.md)
- [Module 5 质量报告](docs/quality_reports/module05_quality_report.md)
- [Module 5 Notebook 3 改进报告](docs/quality_reports/module05_notebook03_improvement_report.md)
- [Module 5 总结](docs/quality_reports/module05_summary.md)
- [Module 6 高级优化](docs/quality_reports/module06_advanced_optimization.md)
- [Module 6-9 质量报告](docs/quality_reports/module06-09_quality_report.md)
- [Module 8-9 详细报告](docs/quality_reports/module08-09_detailed_report.md)
- [项目完成度报告](docs/quality_reports/project_completion_report.md)

### README 示例（高质量参考）
- [Module 1 README](notebooks/Module01_Foundation/README.md) - 基础模块问题导向写法
- [Module 5 README](notebooks/Module05_FineTuning/README.md) - 微调选型与业务约束
- [Module 8 README](notebooks/Module08_Applications/README.md) - 应用闭环与双指标视角
- [Module 9 README](notebooks/Module09_Frontiers/README.md) - 前沿判断框架

### 实用脚本
- **环境测试**：`scripts/test_environment.py`
- **数据集下载**：`scripts/download_datasets.py`

---

## 联系方式

- **GitHub Issues**: [报告问题或提出建议](../../issues)
- **GitHub Discussions**: [讨论和交流](../../discussions)
- **项目主页**: [README.md](../README.md)

---

## 更新日志

### 2026-03-02
- ✅ 新增“文档叙事基线（2026-03-02）”
- ✅ README 模板升级为问题驱动 + 业务映射 + 可交付验收结构
- ✅ 质量检查清单新增“文档叙事”维度
- ✅ README 创建清单与 FAQ 同步更新到新规范

### 2025-02-11
- ✅ 更新 CODEBASE.md 规范文件
- ✅ 反映项目实际结构和命名规范
- ✅ 完善代码规范和技术栈指南
- ✅ 添加详细的常见问题解答
- ✅ 更新项目完成状态（29/29 notebooks, 100%）

### 2025-02-10
- ✅ 完成所有 29 个 notebooks (100%)
- ✅ 创建 Module 5/8/9 完整 README
- ✅ 完成全面质量评估报告
- ✅ 建立内容创建规范和质量标准

---

**遵循本规范，确保内容质量和一致性！** ✅

**项目状态**：29/29 notebooks 已完成 | 平均质量评分 93.7/100 ⭐⭐⭐⭐⭐
