# 大模型学习指南 (LLM Learning Guide)

> 从零开始，系统掌握大语言模型的理论与实践

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![Quality](https://img.shields.io/badge/quality-93.7%2F100-brightgreen.svg)](docs/quality_reports/)

## 📚 项目简介

这是一套完整的大语言模型学习指南，涵盖从基础数学到前沿研究的全部内容。通过 **29 个精心设计的 Jupyter Notebooks**，你将系统掌握 LLM 的理论基础、工程实践和实际应用。

### 🎯 核心特色

- ✅ **理论与实践结合**：每个概念都配有代码实现和可视化
- ✅ **循序渐进**：从 NumPy 到 PyTorch，再到 Transformers 库
- ✅ **动手实践**：100+ 个微实践 + 29 个综合项目
- ✅ **高质量内容**：平均质量评分 93.7/100
- ✅ **完整覆盖**：从基础到前沿，9 个模块 29 个主题

### 👥 适合人群

- 学习者：希望系统掌握 LLM 原理并能独立实现关键模块
- 工程师：希望尽快落地 RAG、Agent、部署与前端集成
- 研究者：希望理解训练范式、前沿架构与研究问题

### 🧭 贯穿式学习主线（业务案例）

本课程使用一条统一主线帮助你降低抽象感：`电商客服智能助理`。

- Module 1-3：让模型“听懂上下文”和“抓住关键信息”
- Module 4-5：让模型“学会公司知识和领域表达”
- Module 6-7：让系统“训得动、跑得稳、可观测可回滚”
- Module 8-9：让应用“可检索、可协作、可持续演进”

### 🧩 知识结构（先修关系）

```text
基础层: 数学 + 神经网络 + PyTorch
  -> 架构层: RNN/Attention -> Transformer
    -> 训练层: Pre-training -> Fine-tuning -> Advanced Training
      -> 工程层: Deployment -> Applications
        -> 前沿层: Frontiers
```

### 📈 学习曲线设计

- 阶段 1（Module 1-3）：建立直觉与最小实现能力
- 阶段 2（Module 4-6）：建立训练与适配能力
- 阶段 3（Module 7-8）：建立生产落地能力
- 阶段 4（Module 9）：建立技术判断与研究视野

每章建议按固定顺序学习：
`真实问题 -> 直觉解释 -> 最小实验 -> 公式 -> 工程实现 -> 常见坑`

---

## 📖 课程大纲

### Module 1: 基础知识 (Foundations)
**5 个 Notebooks | 预计 12-15 小时**

- 01 - 数学基础回顾（线性代数、微积分、概率论）
- 02 - 神经网络基础（感知机、激活函数、损失函数）
- 03 - 反向传播算法（梯度计算、链式法则）
- 04 - PyTorch 基础（张量操作、自动微分、模型训练）
- 05 - 深度学习介绍（MLP/CNN/RNN/Transformer 概览、正则化与 BN）

**学习目标**：掌握深度学习的数学基础和 PyTorch 编程

---

### Module 2: 模型演进 (Evolution)
**3 个 Notebooks | 预计 9-12 小时**

- 01 - RNN 与 LSTM（循环神经网络、长短期记忆）
- 02 - 注意力机制（Attention 原理、多种注意力变体）
- 03 - Seq2Seq 模型（编码器-解码器架构）

**学习目标**：理解序列建模和注意力机制的演进

---

### Module 3: Transformer 架构 (Transformer)
**3 个 Notebooks | 预计 9-12 小时**

- 01 - 自注意力机制（Self-Attention、Query-Key-Value）
- 02 - Transformer 编码器（多头注意力、前馈网络、残差连接）
- 03 - Transformer 解码器（掩码注意力、交叉注意力）

**学习目标**：深入理解 Transformer 架构的每个组件

---

### Module 4: 预训练 (Pre-training)
**3 个 Notebooks | 预计 9-12 小时 | 平均分 91+/100**

- 01 - 语言建模（自回归、掩码语言模型）
- 02 - BERT 架构（双向编码器、预训练任务）
- 03 - GPT 架构（单向解码器、生成式预训练）

**学习目标**：掌握预训练语言模型的核心思想

---

### Module 5: 微调技术 (Fine-tuning) 🏆
**3 个 Notebooks | 预计 11-14 小时 | 平均分 93+/100**

- 01 - 迁移学习与微调基础（特征提取、端到端微调）
- 02 - 参数高效微调 (PEFT)（LoRA、Adapter、Prefix-Tuning）⭐ 100/100
- 03 - 领域适应与持续学习（DAPT、TAPT、EWC）

**学习目标**：掌握高效微调大模型的各种技术

📖 [完整模块指南](notebooks/Module05_FineTuning/README.md)

---

### Module 6: 高级训练 (Advanced Training)
**3 个 Notebooks | 预计 9-12 小时 | 平均分 95/100**

- 01 - 高级优化技术（学习率调度、梯度裁剪、混合精度）
- 02 - 分布式训练（数据并行、模型并行、ZeRO）
- 03 - 高效训练技术（梯度累积、梯度检查点、Flash Attention）

**学习目标**：掌握大规模模型训练的工程技术

---

### Module 7: 部署与优化 (Deployment)
**3 个 Notebooks | 预计 9-12 小时 | 平均分 92/100**

- 01 - 推理优化（量化、剪枝、蒸馏）
- 02 - 模型服务（vLLM、TGI、FastAPI）
- 03 - 生产最佳实践（监控、日志、A/B 测试）

**学习目标**：将模型部署到生产环境

---

### Module 8: 实际应用 (Applications) 🏆
**3 个 Notebooks | 预计 9-12 小时 | 平均分 95.3/100**

- 01 - RAG 系统（检索增强生成、向量数据库）⭐ 97/100
- 02 - AI 智能体系统（ReAct、MCP 协议、多智能体）
- 03 - 前端集成（SSE、WebSocket、Streamlit、React）

**学习目标**：构建生产级 LLM 应用

📖 [完整模块指南](notebooks/Module08_Applications/README.md)

---

### Module 9: 前沿探索 (Frontiers) 🏆
**3 个 Notebooks | 预计 10-13 小时 | 平均分 96.0/100**

- 01 - 新兴架构（MoE、SSM、长上下文技术）
- 02 - 高级训练技术（RLHF、DPO、Constitutional AI）
- 03 - 研究前沿（缩放定律、可解释性、推理增强）⭐ 97/100

**学习目标**：了解 LLM 领域的最新进展

📖 [完整模块指南](notebooks/Module09_Frontiers/README.md)

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.8+ (可选，用于 GPU 加速)
- 8GB+ RAM (16GB+ 推荐)

### 安装步骤

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd llm-learning-guide

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动 Jupyter
jupyter notebook
```

### 开始学习

```bash
# 从 Module 1 开始
cd notebooks/Module01_Foundation
jupyter notebook 01_math_review.ipynb
```

---

## 📊 学习路径

### 路径 1：快速入门（推荐初学者）
**时间**: 40-50 小时

```
Module 1 → Module 2 → Module 3 → Module 4 → Module 5 (LoRA)
```

**目标**: 理解 LLM 基础并能进行简单微调

---

### 路径 2：工程实践（推荐工程师）
**时间**: 60-80 小时

```
Module 1-4 (快速浏览) → Module 5 → Module 6 → Module 7 → Module 8
```

**目标**: 掌握 LLM 工程化和应用开发

---

### 路径 3：完整学习（推荐研究者）
**时间**: 100-120 小时

```
Module 1 → Module 2 → Module 3 → Module 4 → Module 5 →
Module 6 → Module 7 → Module 8 → Module 9
```

**目标**: 全面掌握 LLM 理论与实践

### 路径 4：应用优先（推荐产品/后端）
**时间**: 45-60 小时

```
Module 1 (必学基础) → Module 3 (Attention/Transformer核心) →
Module 5 (PEFT重点) → Module 7 → Module 8
```

**目标**: 在保留必要原理的前提下，尽快完成可上线应用

---

## 🎯 学习建议

### 对于初学者

1. **按顺序学习**：Module 1-4 是基础，不要跳过
2. **动手实践**：每个微实践都要运行并理解
3. **完成项目**：每个模块的综合项目必做
4. **循序渐进**：不要急于求成，理解比速度重要

### 对于有经验者

1. **选择性学习**：根据需求选择模块
2. **深入研究**：阅读附录中的深度推导
3. **扩展实践**：尝试项目的进阶挑战
4. **参与贡献**：发现问题请提 Issue 或 PR

---

## 📈 项目质量

### 完成情况

- ✅ **29/29 notebooks** (100% 完成)
- ✅ **100+ 微实践**
- ✅ **29 个综合项目**
- ✅ **平均质量评分**: 93.7/100

### 质量分布

| 评分范围 | Notebooks 数量 | 百分比 |
|---------|---------------|--------|
| 95-100 分 | 9 个 | 50% |
| 90-94 分 | 7 个 | 38.9% |
| 85-89 分 | 2 个 | 11.1% |

### 最佳 Notebooks Top 5

1. 🥇 **02_parameter_efficient_finetuning.ipynb** (Module 5) - 100/100
2. 🥈 **01_rag_systems.ipynb** (Module 8) - 97/100
3. 🥈 **03_research_frontiers.ipynb** (Module 9) - 97/100
4. 🥉 **01_emerging_architectures.ipynb** (Module 9) - 96/100
5. **所有 Module 6 notebooks** - 95/100

📊 [查看完整质量报告](docs/quality_reports/)

---

## 🛠️ 技术栈

### 核心库

- **PyTorch** 2.0+ - 深度学习框架（Intel Mac 最高 2.2.2，详见 [SETUP.md](SETUP.md#71-intel-mac-x86_64-重要限制)）
- **Transformers** 4.35+ (< 5.0) - Hugging Face 模型库
- **NumPy** 1.24+ (< 2.0) - 数值计算（需与 PyTorch 版本匹配）
- **Matplotlib/Seaborn** - 可视化

### 应用开发

- **FastAPI** - Web 服务
- **Streamlit/Gradio** - 快速原型
- **FAISS** - 向量检索
- **vLLM** - 推理加速

### 完整依赖

查看 [requirements.txt](requirements.txt)

---

## 📂 项目结构

```
llm-learning-guide/
├── README.md                    # 项目总览
├── CODEBASE.md                  # 内容创建规范
├── requirements.txt             # 依赖列表
├── docs/                        # 文档
│   ├── plans/                   # 设计文档
│   └── quality_reports/         # 质量报告
├── notebooks/                   # 学习 Notebook
│   ├── Module01_Foundation/     # 基础知识
│   ├── Module02_Evolution/      # 模型演进
│   ├── Module03_Transformer/    # Transformer
│   ├── Module04_PreTraining/    # 预训练
│   ├── Module05_FineTuning/     # 微调技术 🏆
│   ├── Module06_AdvancedTraining/ # 高级训练
│   ├── Module07_Deployment/     # 部署优化
│   ├── Module08_Applications/   # 实际应用 🏆
│   └── Module09_Frontiers/      # 前沿探索 🏆
├── src/                         # 可复用代码
└── datasets/                    # 示例数据集
```

---

## 📚 相关资源

### 官方文档

- [PyTorch 文档](https://pytorch.org/docs/)
- [Transformers 文档](https://huggingface.co/docs/transformers/)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/)

### 推荐论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [BERT](https://arxiv.org/abs/1810.04805) - 双向预训练
- [GPT-3](https://arxiv.org/abs/2005.14165) - 大规模语言模型
- [LoRA](https://arxiv.org/abs/2106.09685) - 参数高效微调

### 推荐课程

- [Stanford CS224N](http://web.stanford.edu/class/cs224n/) - NLP with Deep Learning
- [DeepLearning.AI](https://www.deeplearning.ai/) - Andrew Ng 的深度学习课程
- [Fast.ai](https://www.fast.ai/) - 实用深度学习

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. **报告问题**：发现错误或有改进建议，请提 [Issue](../../issues)
2. **修复错误**：Fork 项目，修复后提交 Pull Request
3. **改进内容**：优化讲解、添加示例、完善文档
4. **分享经验**：在 [Discussions](../../discussions) 分享学习心得

### 贡献规范

- 遵循 [CODEBASE.md](CODEBASE.md) 中的内容创建规范
- 代码风格遵循 PEP 8
- 提交信息使用约定式提交（Conventional Commits）
- 所有代码必须通过测试

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢

### 灵感来源

- [Andrej Karpathy](https://karpathy.ai/) - Neural Networks: Zero to Hero
- [Sebastian Raschka](https://sebastianraschka.com/) - Build a Large Language Model (From Scratch)
- [Hugging Face](https://huggingface.co/) - Transformers 库和社区

### 参考资料

- Stanford CS224N, CS229, CS231N
- DeepLearning.AI 课程系列
- 各大顶会论文（NeurIPS, ICML, ACL, EMNLP）

---

## 📞 联系方式

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Email**: 请通过 GitHub Issues 联系

---

## 📅 更新日志

### 2025-02-11

- ✅ 完成所有 29 个 notebooks (100%)
- ✅ 创建 Module 5/8/9 完整 README
- ✅ 完成全面质量评估报告
- ✅ 更新项目整体 README 和 CODEBASE

### 2025-02-10

- ✅ 完成 Module 1-7 内容创建
- ✅ 建立内容创建规范和质量标准

---

**开始你的 LLM 学习之旅吧！** 🚀
