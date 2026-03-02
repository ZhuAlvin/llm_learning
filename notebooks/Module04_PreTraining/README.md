# Module 4: 预训练 (Pretraining)

## 📚 模块概览

本模块深入学习预训练语言模型，包括语言建模基础、BERT和GPT架构，掌握现代自然语言处理的核心技术。预训练语言模型是大语言模型的前身，理解其工作原理对于后续的微调技术和实际应用至关重要。

### 🎯 学习目标

- 掌握语言建模的基本概念和评估方法
- 理解BERT的双向编码器架构和掩码语言建模
- 理解GPT的自回归解码器架构和生成能力
- 掌握预训练和微调的完整流程
- 理解Few-shot Learning和In-context Learning的原理
- 对比不同预训练模型的优劣和适用场景

### ⏱️ 预计学习时间

**总计**: 9-12 小时

---

## 📖 Notebooks

### 4.1 语言建模基础 ⭐⭐⭐⭐
**文件**: `01_language_modeling.ipynb`
**时长**: 3-4 小时
**难度**: ⭐⭐⭐
**评分**: 91+/100

**内容**:
- N-gram语言模型：原理、评估、局限性
- 神经语言模型：从RNN到Transformer
- 困惑度（Perplexity）评估指标
- 自回归语言建模（CLM）
- 掩码语言建模（MLM）
- 语言模型的应用场景

**亮点**:
- ✅ 6 个微实践（含n-gram实现、困惑度计算）
- ✅ 12+ 个高质量可视化
- ✅ 完整的语言模型评估流程
- ✅ 不同语言模型的性能对比

**关键概念**: Language Modeling, Perplexity, Autoregressive LM, Masked LM, N-gram Model

---

### 4.2 BERT 架构 ⭐⭐⭐⭐⭐
**文件**: `02_bert_architecture.ipynb`
**时长**: 3-4 小时
**难度**: ⭐⭐⭐⭐
**评分**: 92+/100

**内容**:
- Encoder-only架构设计
- 掩码语言建模（MLM）原理与实现
- 下一句预测（NSP）任务
- BERT的预训练和微调流程
- BERT家族变体（RoBERTa, ALBERT等）
- BERT的应用场景和局限性

**亮点**:
- ✅ 7 个微实践（含MLM实现、BERT微调）
- ✅ 15+ 个可视化
- ✅ 从零实现简化版BERT
- ✅ 详细的预训练数据处理流程

**关键概念**: BERT, Encoder-only, Masked Language Modeling, Next Sentence Prediction, Fine-tuning

---

### 4.3 GPT 架构 ⭐⭐⭐⭐⭐
**文件**: `03_gpt_architecture.ipynb`
**时长**: 3-4 小时
**难度**: ⭐⭐⭐⭐
**评分**: 93+/100

**内容**:
- Decoder-only架构设计
- 自回归语言建模原理
- 因果注意力机制实现
- 文本生成策略（贪婪搜索、束搜索、采样）
- Few-shot Learning和In-context Learning
- GPT家族演化（GPT-1到GPT-4）
- GPT的应用场景和局限性

**亮点**:
- ✅ 8 个微实践（含文本生成、few-shot演示）
- ✅ 18+ 个可视化
- ✅ 从零实现简化版GPT
- ✅ 详细的生成策略对比

**关键概念**: GPT, Decoder-only, Autoregressive LM, Causal Attention, Few-shot Learning, In-context Learning

---

## 🎯 学习路径

### 初学者路径
```
01 语言建模基础 → 02 BERT架构基础 → 03 GPT架构基础 → 实践项目
```
**时间**: 6-8 小时
**目标**: 掌握预训练模型基础概念

### 进阶路径
```
01 语言建模基础 → 02 BERT架构完整实现 → 03 GPT架构完整实现 → 高级项目
```
**时间**: 9-12 小时
**目标**: 深入理解预训练模型原理与应用

### 研究者路径
```
完整学习所有内容 → 实现预训练模型变体 → 探索新的预训练目标
```
**时间**: 12+ 小时
**目标**: 创新预训练模型技术

---

## 🛠️ 实践项目建议

### 项目 1: 情感分析
**难度**: ⭐⭐⭐
**技术**: BERT + 分类头
**数据集**: IMDb电影评论
**时间**: 3-4 小时

### 项目 2: 问答系统
**难度**: ⭐⭐⭐⭐
**技术**: BERT + QA头
**数据集**: SQuAD
**时间**: 4-5 小时

### 项目 3: 文本生成
**难度**: ⭐⭐⭐⭐
**技术**: GPT + 提示工程
**数据集**: 自定义数据集
**时间**: 4-5 小时

---

## 📊 知识图谱

```
预训练语言模型
├── 语言建模基础
│   ├── N-gram模型
│   ├── 神经语言模型
│   ├── 自回归语言建模 (CLM)
│   ├── 掩码语言建模 (MLM)
│   └── 困惑度评估
├── BERT架构
│   ├── Encoder-only
│   ├── 掩码语言建模
│   ├── 下一句预测
│   ├── 预训练流程
│   ├── 微调流程
│   └── BERT变体
│       ├── RoBERTa
│       ├── ALBERT
│       └── DistilBERT
└── GPT架构
    ├── Decoder-only
    ├── 自回归语言建模
    ├── 因果注意力
    ├── 文本生成策略
    │   ├── 贪婪搜索
    │   ├── 束搜索
    │   └── 采样方法
    ├── Few-shot Learning
    ├── In-context Learning
    └── GPT家族
        ├── GPT-1
        ├── GPT-2
        ├── GPT-3
        └── GPT-4
```

---

## 🔗 相关资源

### 📄 核心论文

**BERT系列**:
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (2018)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) (2019)
- [ALBERT: A Lite BERT for Self-supervised Learning](https://arxiv.org/abs/1909.11942) (2019)

**GPT系列**:
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1, 2018)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020)

### 💻 代码库

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [BERT GitHub](https://github.com/google-research/bert)
- [GPT-2 GitHub](https://github.com/openai/gpt-2)

### 📚 博客文章

- [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [How GPT3 Works](http://jalammar.github.io/how-gpt3-works-visualizations-animations/)
- [Language Models are Few-Shot Learners: An Illustrated Overview](https://towardsdatascience.com/language-models-are-few-shot-learners-an-illustrated-overview-364f4a8f161a)

---

## ❓ 常见问题

### Q1: BERT 和 GPT 的主要区别是什么？

**A**: 核心区别：
- **架构**：BERT是Encoder-only（双向），GPT是Decoder-only（单向）
- **训练目标**：BERT使用MLM，GPT使用自回归语言建模
- **擅长任务**：BERT擅长理解任务（分类、NER、QA），GPT擅长生成任务（文本生成、对话）
- **使用方式**：BERT需要微调，GPT可以few-shot学习

### Q2: 为什么预训练模型如此有效？

**A**: 预训练的优势：
- 利用大规模无标注数据学习通用语言表示
- 迁移学习减少下游任务的数据需求
- 捕获语言的复杂模式和知识
- 降低下游任务的训练难度和计算成本

### Q3: 如何评估语言模型的性能？

**A**: 常用评估指标：
- **困惑度（Perplexity）**：衡量模型预测的不确定性
- **准确率**：针对特定任务的性能
- **BLEU/ROUGE**：文本生成任务
- **人类评估**：主观质量评估
- **Few-shot性能**：大模型的泛化能力

### Q4: 预训练需要多少数据和计算资源？

**A**: 资源需求：
- **数据**：基础模型需要数GB文本，大模型需要TB级数据
- **计算**：基础模型需要单个GPU几天，大模型需要数百GPU/TPU数月
- **替代方案**：使用公开预训练模型，然后微调

### Q5: In-context Learning 和 Few-shot Learning 有什么区别？

**A**: 概念区别：
- **Few-shot Learning**：通过少量示例学习新任务
- **In-context Learning**：在推理时通过上下文示例指导模型
- **关系**：In-context Learning是Few-shot Learning的一种实现方式，主要用于大语言模型

---

## 🎓 学习检查清单

完成本模块后，你应该能够：

- [ ] 解释语言建模的基本原理
- [ ] 计算和理解困惑度指标
- [ ] 实现掩码语言建模和自回归语言建模
- [ ] 构建和训练简化版BERT和GPT
- [ ] 使用预训练模型进行微调
- [ ] 设计有效的提示进行few-shot学习
- [ ] 评估语言模型的性能
- [ ] 选择适合特定任务的预训练模型

---

## 📈 模块质量

| Notebook | 评分 | 状态 |
|----------|------|------|
| 01_language_modeling | 91+/100 | ⭐⭐⭐⭐ |
| 02_bert_architecture | 92+/100 | ⭐⭐⭐⭐ |
| 03_gpt_architecture | 93+/100 | ⭐⭐⭐⭐⭐ |

**模块平均分**: 92+/100 ⭐⭐⭐⭐

---

## 🚀 下一步

完成本模块后，建议继续学习：

- **Module 5**: 微调技术（PEFT、领域适应）
- **Module 6**: 高级训练（分布式训练、混合精度）
- **Module 8**: 实际应用（RAG系统、Agent开发）

---

**模块维护者**: AI Learning Team
**最后更新**: 2025-02-11
**版本**: 2.0 (大幅改进版)
**反馈**: 欢迎提出改进建议