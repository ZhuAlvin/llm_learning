# Module 5 改进实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标**: 将 Module 5 从 75 分提升到 88+ 分，完成度从 66.7% 提升到 100%

**当前状态**:
- ❌ 01_transfer_learning.ipynb: 缺失
- ✅ 02_parameter_efficient_finetuning.ipynb: 95/100
- ⚠️ 03_domain_adaptation.ipynb: 62/100

**改进目标**:
- ✅ 01_transfer_learning.ipynb: 创建，目标 90+
- ✅ 02_parameter_efficient_finetuning.ipynb: 优化到 97+
- ✅ 03_domain_adaptation.ipynb: 改进到 88+
- ✅ Module README: 创建

---

## Task 1: 创建 01_transfer_learning.ipynb

**优先级**: 🔴 最高

**Files:**
- Create: `notebooks/Module05_Finetuning/01_transfer_learning.ipynb`
- Reference: `docs/plans/2025-02-10-module05-finetuning-basics.md`

**目标评分**: 90+

### Step 1: 创建 notebook 框架

创建包含 8 个主要章节的 notebook：

```markdown
# Module 5.1: 迁移学习与微调基础

## 1. 本章概览
## 2. 迁移学习理论
## 3. 预训练-微调范式
## 4. 特征提取 vs 微调
## 5. 学习率调度策略
## 6. 超参数选择
## 7. 完整微调流程
## 8. 常见问题与调试
```

### Step 2: 添加概览和导入

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
torch.manual_seed(42)
print("✓ Libraries imported")
```

### Step 3: 实现迁移学习理论部分

添加内容：
- 迁移学习定义和动机
- 源域和目标域概念
- 迁移学习的三种场景
- 为什么预训练模型有效

包含可视化：迁移学习流程图

### Step 4: 实现预训练-微调范式

添加微实践：
```python
# 🔬 Micro Practice: 预训练-微调流程演示
# 展示从预训练模型到微调的完整流程
```

### Step 5: 实现特征提取 vs 微调对比

添加微实践：
```python
# 🔬 Micro Practice: 特征提取 vs 微调对比
# 对比两种方法的性能和训练时间
```

包含可视化：性能对比图

### Step 6: 实现学习率调度

添加内容：
- Warmup 策略
- 学习率衰减
- 余弦退火
- 判别式学习率

添加微实践：
```python
# 🔬 Micro Practice: 学习率调度器实现
# 实现和可视化不同的学习率调度策略
```

### Step 7: 实现超参数选择

添加内容：
- 批次大小选择
- 学习率范围
- 训练轮数
- 权重衰减
- Dropout

添加微实践：
```python
# 🔬 Micro Practice: 超参数敏感性分析
# 展示不同超参数对性能的影响
```

### Step 8: 实现完整微调流程

添加完整的文本分类示例：
```python
# 🚀 Capstone: 完整的文本分类微调
# 从数据加载到模型评估的端到端流程
```

包含：
- 数据准备
- 模型加载
- 训练循环
- 验证和测试
- 结果可视化

### Step 9: 添加常见问题和调试

添加 FAQ：
- 如何选择预训练模型？
- 微调时如何避免过拟合？
- 如何处理类别不平衡？
- 训练不收敛怎么办？

### Step 10: 添加总结和思考题

总结核心要点，添加 5 个思考题

### Step 11: Commit

```bash
git add notebooks/Module05_Finetuning/01_transfer_learning.ipynb
git commit -m "feat(module05): create transfer learning basics notebook"
```

---

## Task 2: 大幅改进 03_domain_adaptation.ipynb

**优先级**: 🔴 最高

**Files:**
- Modify: `notebooks/Module05_Finetuning/03_domain_adaptation.ipynb`

**目标**: 从 62 分提升到 88+ 分

### Step 1: 补充完整的 DAPT 实现

替换简化代码为完整实现：

```python
class DAPTTrainer:
    """完整的领域自适应预训练实现"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prepare_mlm_data(self, texts, mlm_probability=0.15):
        """准备 MLM 训练数据"""
        # 实现 token masking
        pass

    def train(self, domain_corpus, epochs=3, lr=1e-5, batch_size=16):
        """完整的训练循环"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for batch in domain_corpus:
                # MLM training
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(domain_corpus)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return self.model
```

### Step 2: 添加领域偏移检测微实践

```python
# 🔬 Micro Practice: 领域偏移检测
# 目标：可视化源域和目标域的词汇分布差异
# 预期结果：理解领域偏移的具体表现
```

实现：
- 词频统计
- 词汇分布对比
- 词云可视化

### Step 3: 添加 DAPT 训练演示

```python
# 🔬 Micro Practice: DAPT 训练演示
# 目标：在简化数据上演示 DAPT 流程
# 预期结果：理解 DAPT 如何改进领域性能
```

实现：
- 创建模拟的通用和领域数据
- 训练前后性能对比
- 训练曲线可视化

### Step 4: 添加 TAPT vs DAPT 对比

```python
# 🔬 Micro Practice: TAPT vs DAPT 对比
# 目标：对比两种方法的效果
# 预期结果：理解何时使用哪种方法
```

实现：
- 并行训练 DAPT 和 TAPT
- 性能对比
- 数据效率分析

### Step 5: 实现完整的 EWC

```python
class EWC:
    """完整的 Elastic Weight Consolidation 实现"""

    def __init__(self, model, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_dict = {}
        self.optpar_dict = {}

    def compute_fisher(self, dataloader):
        """计算 Fisher 信息矩阵"""
        self.model.eval()

        for name, param in self.model.named_parameters():
            self.fisher_dict[name] = torch.zeros_like(param)

        for batch in dataloader:
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_dict[name] += param.grad.pow(2)

        # Average over dataset
        for name in self.fisher_dict:
            self.fisher_dict[name] /= len(dataloader)

        # Store optimal parameters
        for name, param in self.model.named_parameters():
            self.optpar_dict[name] = param.data.clone()

    def penalty(self):
        """计算 EWC 惩罚项"""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optpar = self.optpar_dict[name]
                loss += (fisher * (param - optpar).pow(2)).sum()
        return self.lambda_ewc * loss
```

### Step 6: 添加灾难性遗忘演示

```python
# 🔬 Micro Practice: 灾难性遗忘演示
# 目标：展示顺序学习两个任务时的遗忘现象
# 预期结果：理解为什么需要持续学习策略
```

实现：
- 训练任务 A
- 训练任务 B
- 测试任务 A 性能下降
- 可视化遗忘曲线

### Step 7: 添加 EWC 效果验证

```python
# 🔬 Micro Practice: EWC 防止遗忘
# 目标：验证 EWC 如何缓解灾难性遗忘
# 预期结果：对比有无 EWC 的性能差异
```

实现：
- 对比普通训练 vs EWC 训练
- 性能对比图
- Fisher 信息可视化

### Step 8: 实现 Experience Replay

```python
class ExperienceReplay:
    """经验回放缓冲区"""

    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, samples):
        """添加样本到缓冲区"""
        for sample in samples:
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)
            self.buffer.append(sample)

    def sample(self, batch_size):
        """从缓冲区采样"""
        indices = np.random.choice(len(self.buffer),
                                   min(batch_size, len(self.buffer)),
                                   replace=False)
        return [self.buffer[i] for i in indices]
```

### Step 9: 添加多领域学习实验

```python
# 🔬 Micro Practice: 多领域学习
# 目标：使用 Adapter 或 MoE 处理多个领域
# 预期结果：理解多领域学习的架构设计
```

### Step 10: 添加可视化

添加 5 个关键可视化：

1. **领域词汇分布对比**
```python
# 词云或直方图展示源域和目标域的词汇差异
```

2. **DAPT 训练曲线**
```python
# Loss vs epochs，展示训练过程
```

3. **灾难性遗忘曲线**
```python
# 任务 A 性能随任务 B 训练的变化
```

4. **Fisher 信息热图**
```python
# 展示哪些参数对旧任务最重要
```

5. **多领域性能雷达图**
```python
# 展示模型在不同领域的性能
```

### Step 11: 添加实际案例

使用医学文本数据集的完整案例：

```python
# 🚀 Capstone: 医学领域适应完整案例
# 从通用 BERT 到医学 BERT 的完整流程
```

包含：
- 数据准备（通用文本 + 医学文本）
- DAPT 训练
- 下游任务微调
- 性能对比（通用 vs 领域适应）

### Step 12: 更新 FAQ 和总结

扩充 FAQ，添加更多实用建议

### Step 13: Commit

```bash
git add notebooks/Module05_Finetuning/03_domain_adaptation.ipynb
git commit -m "feat(module05): significantly improve domain adaptation notebook"
```

---

## Task 3: 优化 02_parameter_efficient_finetuning.ipynb

**优先级**: 🟡 中等

**Files:**
- Modify: `notebooks/Module05_Finetuning/02_parameter_efficient_finetuning.ipynb`

**目标**: 从 95 分提升到 97+ 分

### Step 1: 添加类型提示

为所有函数添加类型提示：

```python
from typing import Optional, Tuple

class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0
    ) -> None:
        super(LoRALayer, self).__init__()
        # ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T @ self.B.T) * self.scaling
```

### Step 2: 添加 Hugging Face PEFT 库示例

```python
# 🔬 Micro Practice: 使用 Hugging Face PEFT 库
# 目标：学习在实际项目中使用 PEFT
# 预期结果：掌握 PEFT 库的基本用法

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# 加载基础模型
base_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 配置 LoRA
config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

# 应用 LoRA
model = get_peft_model(base_model, config)
model.print_trainable_parameters()

# 训练（与普通训练相同）
# ...

# 保存和加载
model.save_pretrained("lora_checkpoint")
```

### Step 3: 添加实际性能对比

添加真实的训练时间和内存使用数据：

```python
# 🔬 Micro Practice: 实际性能对比
# 目标：测量不同方法的实际开销
# 预期结果：理解 PEFT 的实际效率提升

import time
import psutil

def benchmark_training(model, dataloader, method_name):
    """测量训练时间和内存使用"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024**2

    # 训练一个 epoch
    for batch in dataloader:
        # Training step
        pass

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024**2

    return {
        'method': method_name,
        'time': end_time - start_time,
        'memory': end_memory - start_memory
    }
```

### Step 4: 添加 LoRA 权重分解可视化

```python
# 可视化 LoRA 的低秩分解
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 原始权重
W = torch.randn(256, 256)
axes[0].imshow(W, cmap='RdBu', aspect='auto')
axes[0].set_title('Original Weight W (256×256)')

# LoRA A 矩阵
A = torch.randn(4, 256)
axes[1].imshow(A, cmap='RdBu', aspect='auto')
axes[1].set_title('LoRA A (4×256)')

# LoRA B 矩阵
B = torch.randn(256, 4)
axes[2].imshow(B, cmap='RdBu', aspect='auto')
axes[2].set_title('LoRA B (256×4)')

plt.tight_layout()
plt.show()
```

### Step 5: 扩充实际应用案例

添加更多实际场景：
- 情感分析微调
- 命名实体识别微调
- 问答系统微调

### Step 6: Commit

```bash
git add notebooks/Module05_Finetuning/02_parameter_efficient_finetuning.ipynb
git commit -m "feat(module05): optimize PEFT notebook with types and examples"
```

---

## Task 4: 创建 Module 5 README

**优先级**: 🟢 低

**Files:**
- Create: `notebooks/Module05_Finetuning/README.md`

### Step 1: 创建 README 内容

```markdown
# Module 5: 微调技术 (Fine-tuning)

## 📚 模块概览

本模块深入讲解大语言模型的微调技术，从基础的迁移学习到前沿的参数高效微调方法。

### 学习目标

- 掌握迁移学习和微调的基本原理
- 理解并实现参数高效微调（PEFT）方法
- 学习领域适应和持续学习技术
- 能够在实际项目中应用微调技术

### 预计学习时间

**总计**: 11-13 小时

---

## 📖 Notebooks

### 5.1 迁移学习与微调基础
**文件**: `01_transfer_learning.ipynb`
**时长**: 3-4 小时
**难度**: ⭐⭐⭐

**内容**:
- 迁移学习理论
- 预训练-微调范式
- 特征提取 vs 微调
- 学习率调度策略
- 超参数选择
- 完整微调流程

**关键概念**:
- Transfer Learning
- Fine-tuning
- Learning Rate Scheduling
- Hyperparameter Tuning

---

### 5.2 参数高效微调 (PEFT)
**文件**: `02_parameter_efficient_finetuning.ipynb`
**时长**: 4-5 小时
**难度**: ⭐⭐⭐⭐

**内容**:
- LoRA (Low-Rank Adaptation)
- Adapter Layers
- Prefix-Tuning
- Prompt-Tuning
- QLoRA, (IA)³, BitFit
- 多任务 PEFT 部署

**关键概念**:
- Parameter Efficiency
- Low-Rank Decomposition
- Adapter Architecture
- Soft Prompts

**亮点**: ⭐⭐⭐⭐⭐
- 从零实现所有主要 PEFT 方法
- 详细的参数效率分析
- 实用的方法选择指南

---

### 5.3 领域适应与持续学习
**文件**: `03_domain_adaptation.ipynb`
**时长**: 4-5 小时
**难度**: ⭐⭐⭐⭐

**内容**:
- 领域偏移检测
- DAPT (Domain-Adaptive Pre-training)
- TAPT (Task-Adaptive Pre-training)
- 灾难性遗忘
- EWC (Elastic Weight Consolidation)
- Experience Replay
- 多领域学习

**关键概念**:
- Domain Shift
- Continual Learning
- Catastrophic Forgetting
- Multi-domain Learning

---

## 🎯 学习路径

### 初学者路径
```
01 迁移学习基础 → 02 PEFT (LoRA部分) → 实践项目
```

### 进阶路径
```
01 迁移学习基础 → 02 PEFT (完整) → 03 领域适应 → 综合项目
```

### 研究者路径
```
完整学习所有内容 → 深入研究 PEFT 变体 → 探索新方法
```

---

## 🛠️ 实践项目建议

### 项目 1: 情感分析微调
**难度**: ⭐⭐⭐
**技术**: 基础微调 + LoRA
**数据集**: IMDb, SST-2

### 项目 2: 医学文本分类
**难度**: ⭐⭐⭐⭐
**技术**: DAPT + TAPT + PEFT
**数据集**: PubMed, MIMIC

### 项目 3: 多任务学习系统
**难度**: ⭐⭐⭐⭐⭐
**技术**: 多任务 PEFT + 持续学习
**数据集**: GLUE benchmark

---

## 📊 知识图谱

```
微调技术
├── 基础微调
│   ├── 迁移学习
│   ├── 特征提取
│   └── 端到端微调
├── 参数高效微调 (PEFT)
│   ├── LoRA
│   ├── Adapter
│   ├── Prefix/Prompt Tuning
│   └── 高级方法 (QLoRA, IA³)
└── 领域适应
    ├── DAPT/TAPT
    ├── 持续学习
    └── 多领域学习
```

---

## 🔗 相关资源

### 论文
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Adapter**: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- **Prefix-Tuning**: [Prefix-Tuning: Optimizing Continuous Prompts](https://arxiv.org/abs/2101.00190)
- **EWC**: [Overcoming Catastrophic Forgetting](https://arxiv.org/abs/1612.00796)

### 代码库
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers)

### 博客
- [Hugging Face PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 详解](https://huggingface.co/blog/lora)

---

## ❓ 常见问题

**Q1: 什么时候应该使用 PEFT 而不是全量微调？**

A: 当满足以下条件时优先考虑 PEFT：
- 计算资源有限（GPU 内存不足）
- 需要部署多个任务模型
- 模型参数量很大（>1B）
- 训练数据相对较少

**Q2: LoRA 和 Adapter 哪个更好？**

A: 取决于场景：
- **推理速度优先** → LoRA（可合并权重）
- **多任务切换** → Adapter（模块化）
- **参数效率** → LoRA（更少参数）
- **训练稳定性** → Adapter（更稳定）

**Q3: 如何选择 LoRA 的 rank？**

A: 经验法则：
- 简单任务：r=4
- 一般任务：r=8（推荐默认值）
- 复杂任务：r=16-32
- 大模型：r=64

**Q4: DAPT 需要多少数据？**

A: 建议：
- 最少：100K-1M tokens
- 推荐：10M+ tokens
- 理想：100M+ tokens

---

## 🎓 学习检查清单

完成本模块后，你应该能够：

- [ ] 解释迁移学习和微调的原理
- [ ] 实现基础的模型微调流程
- [ ] 从零实现 LoRA 和 Adapter
- [ ] 使用 Hugging Face PEFT 库
- [ ] 理解并应用领域适应技术
- [ ] 处理灾难性遗忘问题
- [ ] 设计多任务学习系统
- [ ] 在实际项目中选择合适的微调方法

---

## 📈 下一步

完成本模块后，建议继续学习：

- **Module 6**: 高级训练技术
- **Module 7**: 模型部署与优化
- **Module 8**: 实际应用（RAG, Agent）

---

**模块维护者**: AI Learning Team
**最后更新**: 2025-02-11
**反馈**: 欢迎提出改进建议
```

### Step 2: Commit

```bash
git add notebooks/Module05_Finetuning/README.md
git commit -m "docs(module05): create module README"
```

---

## Task 5: 最终验证和文档

**优先级**: 🟢 低

### Step 1: 运行所有 notebooks

确保所有代码可以正常运行

### Step 2: 更新质量报告

重新评估所有 notebooks，更新质量报告

### Step 3: 创建改进总结

记录改进前后的对比数据

### Step 4: Final commit

```bash
git add docs/quality_reports/module05_quality_report.md
git commit -m "docs(module05): update quality report after improvements"
```

---

## 执行建议

### 方案 A: 顺序执行（推荐新手）

1. Task 1: 创建 01_transfer_learning.ipynb (4-6 小时)
2. Task 2: 改进 03_domain_adaptation.ipynb (6-8 小时)
3. Task 3: 优化 02_parameter_efficient_finetuning.ipynb (2-3 小时)
4. Task 4: 创建 README (1 小时)
5. Task 5: 最终验证 (1-2 小时)

**总时间**: 14-20 小时

### 方案 B: 并行执行（推荐有经验者）

**会话 1**: Task 1 (创建 01)
**会话 2**: Task 2 (改进 03)
**会话 3**: Task 3 (优化 02)

然后合并后执行 Task 4 和 5

**总时间**: 8-10 小时（并行）

---

## 预期结果

### 改进前
- 完成度: 66.7% (2/3)
- 平均分: 75/100
- 最低分: 62 (03_domain_adaptation)

### 改进后
- 完成度: 100% (3/3 + README)
- 平均分: 90+/100
- 最低分: 88+ (03_domain_adaptation)

### 提升
- 完成度: +33.3%
- 平均分: +15 分
- 最低分: +26 分

---

## 验收标准

- [ ] 01_transfer_learning.ipynb 创建完成，评分 ≥ 90
- [ ] 03_domain_adaptation.ipynb 改进完成，评分 ≥ 88
- [ ] 02_parameter_efficient_finetuning.ipynb 优化完成，评分 ≥ 97
- [ ] Module README 创建完成
- [ ] 所有代码可运行
- [ ] 所有可视化正常显示
- [ ] 模块平均分 ≥ 90

---

**计划创建时间**: 2025-02-11
**预计完成时间**: 2025-02-12 (顺序) 或 2025-02-11 (并行)
