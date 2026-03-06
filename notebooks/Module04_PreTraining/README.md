# Module 4: 预训练 (Pretraining)

## 📚 模块概览

本模块回答一个关键问题：为什么“大模型没见过你的业务数据”也能先表现不错？

你将理解预训练的核心逻辑：先在海量通用数据上学语言规律，再在具体任务中适配。
这也是从 Transformer 机制走向大模型能力的核心桥梁。

生活化主线沿用 `电商客服智能助理`：
- 为什么模型能先“读懂大多数用户表达”
- 为什么某些专业规则仍然答不准
- 为什么后续微调能显著提升业务效果

### 🎯 学习目标

- 掌握语言建模的基本概念和评估方法
- 理解 BERT (Bidirectional Encoder Representations from Transformers) 的双向编码器架构和掩码语言建模
- 理解 GPT (Generative Pre-trained Transformer) 的自回归解码器架构和生成能力
- 掌握预训练和微调的完整流程
- 理解 Few-shot Learning (少样本学习) 和 In-context Learning (上下文学习) 的原理
- 对比不同预训练模型的优劣和适用场景

### ✅ 完成本模块后的可交付产出

- 一份语言建模评估报告（含困惑度与生成质量对比）
- 一个可运行的 BERT 任务原型或 GPT 生成原型
- 一份 BERT/GPT 选型建议（按任务、成本、时效性）

### ⏱️ 预计学习时间

**总计**: 9-12 小时

### 📈 学习曲线设计

- 第 1 段（4.1）：先理解“模型如何学习语言分布”
- 第 2 段（4.2）：再理解“BERT 如何提升理解任务表现”
- 第 3 段（4.3）：最后理解“GPT 如何支持生成与上下文学习”

### 🧭 每章建议阅读顺序

`业务问题 -> 任务定义 -> 最小实现 -> 关键公式 -> 能力边界 -> 选型建议`

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
- 自回归语言建模 CLM (Causal Language Modeling, 因果语言建模)
- 掩码语言建模 MLM (Masked Language Modeling, 掩码语言建模)
- 语言模型的应用场景
- 业务映射：用“客服常见问法预测”理解语言建模的价值与局限
**业务问题映射**：
- “模型为什么能预测用户下一句可能说什么？” -> 语言模型概率建模
- “预测准确度怎么量化？” -> 困惑度评估与业务可用阈值


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
- 下一句预测 NSP (Next Sentence Prediction, 下一句预测) 任务
- BERT的预训练和微调流程
- BERT家族变体（RoBERTa, ALBERT等）
- BERT的应用场景和局限性
- 业务映射：用“意图识别与文本分类”理解 BERT 的理解优势
**业务问题映射**：
- “客服消息意图分类效果如何快速提升？” -> BERT 预训练迁移到分类任务
- “同一句话在不同上下文中含义不同？” -> 双向编码的上下文理解优势


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
- 业务映射：用“自动生成客服回复草稿”理解 GPT 的生成优势
**业务问题映射**：
- “如何让模型自动草拟客服回复？” -> GPT 自回归生成能力
- “给几个示例就能做新任务？” -> Few-shot / In-context Learning


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
**最低完成标准**: 能解释 CLM 与 MLM 差异，并完成一个小型实验

### 进阶路径
```
01 语言建模基础 → 02 BERT架构完整实现 → 03 GPT架构完整实现 → 高级项目
```
**时间**: 9-12 小时
**目标**: 深入理解预训练模型原理与应用
**最低完成标准**: 完成 BERT/GPT 对比并给出任务选型结论

### 研究者路径
```
完整学习所有内容 → 实现预训练模型变体 → 探索新的预训练目标
```
**时间**: 12+ 小时
**目标**: 创新预训练模型技术
**最低完成标准**: 设计一个预训练目标或数据策略改动并评估效果

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
- **擅长任务**：BERT擅长理解任务（分类、NER (Named Entity Recognition, 命名实体识别)、QA (Question Answering, 问答)），GPT擅长生成任务（文本生成、对话）
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
- **BLEU (Bilingual Evaluation Understudy, 双语评估指标) / ROUGE (Recall-Oriented Understudy for Gisting Evaluation, 摘要评估指标)**：文本生成任务
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
