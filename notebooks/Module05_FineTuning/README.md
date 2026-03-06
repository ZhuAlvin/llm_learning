# Module 5: 微调技术 (Fine-tuning)

## 📚 模块概览

本模块聚焦“模型如何真正适配业务”。你将回答 3 个工程问题：

- 何时应做全量微调，何时应选 PEFT (Parameter-Efficient Fine-Tuning, 参数高效微调)？
- 数据有限且领域变化快时，如何稳定提升效果？
- 新任务持续加入时，如何避免模型遗忘旧能力？

生活化主线沿用 `电商客服智能助理`：
- 新业务规则上线后，如何快速让模型学会并上线
- 成本受限时，如何优先选择高性价比微调路线
- 多任务并存时，如何保持历史能力不退化

### 🎯 学习目标

- 掌握迁移学习和微调的基本原理
- 理解并实现参数高效微调（PEFT）方法
- 学习领域适应和持续学习技术
- 能够在实际项目中应用微调技术

### ✅ 完成本模块后的可交付产出

- 一个可复用的微调实验模板（含训练与评估）
- 一份 PEFT 选型记录（LoRA/Adapter/Prompt 等）
- 一份持续学习策略草案（遗忘评估 + 缓解方案）

### ⏱️ 预计学习时间

**总计**: 11-14 小时

### 📈 学习曲线设计

- 第 1 段（5.1）：先掌握标准微调流程与调参逻辑
- 第 2 段（5.2）：再掌握参数高效微调与部署权衡
- 第 3 段（5.3）：最后处理领域迁移与持续学习问题

### 🧭 每章建议阅读顺序

`业务目标 -> 方法对比 -> 最小实现 -> 成本评估 -> 效果评估 -> 上线建议`

---

## 📖 Notebooks

### 5.1 迁移学习与微调基础 ⭐⭐⭐⭐⭐
**文件**: `01_transfer_learning.ipynb`
**时长**: 3-4 小时
**难度**: ⭐⭐⭐
**评分**: 92-94/100

**内容**:
- 迁移学习理论与数学表示
- 预训练-微调范式
- 特征提取 vs 微调对比
- 学习率调度策略（7种实现）
- 超参数选择指南
- 完整的文本分类微调流程
- 业务映射：用“新增客服标签分类”理解全量微调流程
**业务问题映射**：
- “新增业务标签后，模型要重新训练多久？” -> 全量微调 vs 冻结层策略
- “学习率设多少既有效又不破坏原有能力？” -> 分层学习率与 Warmup


**亮点**:
- ✅ 5 个微实践 + 1 个 Capstone 项目
- ✅ 6+ 个高质量可视化
- ✅ 完整的训练器类实现
- ✅ 8 个详细 FAQ

**关键概念**: Transfer Learning, Fine-tuning, Learning Rate Scheduling, Hyperparameter Tuning

---

### 5.2 参数高效微调 (PEFT) ⭐⭐⭐⭐⭐
**文件**: `02_parameter_efficient_finetuning.ipynb`
**时长**: 4-5 小时
**难度**: ⭐⭐⭐⭐
**评分**: 100/100 🏆

**内容**:
- LoRA (Low-Rank Adaptation, 低秩适配)
- Adapter Layers
- Prefix-Tuning & Prompt-Tuning
- QLoRA (Quantized LoRA, 量化低秩适配)、(IA)³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)、BitFit (Bias-terms Fine-Tuning, 偏置项微调)
- 多任务 PEFT 部署
- Hugging Face PEFT 库完整教程
- 业务映射：用“有限显存下快速适配新活动规则”理解 PEFT 价值
**业务问题映射**：
- “显存只够 8GB，怎么微调 7B 模型？” -> LoRA 低秩适配
- “多个活动规则同时上线如何切换？” -> 多任务 PEFT 部署


**亮点**:
- ✅ 从零实现所有主要 PEFT 方法
- ✅ 11 个微实践（含性能 Benchmark）
- ✅ 8+ 个精美可视化（含 LoRA 权重分解）
- ✅ 3 个实际应用案例（情感分析、NER (Named Entity Recognition, 命名实体识别)、QA (Question Answering, 问答)）
- ✅ 完整类型提示和实际性能数据

**关键概念**: Parameter Efficiency, Low-Rank Decomposition, Adapter Architecture, Soft Prompts

---

### 5.3 领域适应与持续学习 ⭐⭐⭐⭐⭐
**文件**: `03_domain_adaptation.ipynb`
**时长**: 4-5 小时
**难度**: ⭐⭐⭐⭐
**评分**: 88+/100

**内容**:
- 领域偏移检测与分析
- DAPT (Domain-Adaptive Pre-training, 领域自适应预训练)
- TAPT (Task-Adaptive Pre-training, 任务自适应预训练)
- 灾难性遗忘问题
- EWC (Elastic Weight Consolidation, 弹性权重巩固)
- Experience Replay
- 多领域学习架构
- 业务映射：用“售后任务新旧政策并存”理解持续学习挑战
**业务问题映射**：
- “售后新政策上线后旧知识全忘了？” -> 灾难性遗忘与 EWC
- “跨品类迁移效果差？” -> DAPT + TAPT 领域适应


**亮点**:
- ✅ 7 个完整微实践（含完整 DAPT/EWC 实现）
- ✅ 20+ 个可视化（含 Fisher 信息热图）
- ✅ 800+ 行可运行代码
- ✅ 完整的方法选择指南

**关键概念**: Domain Shift, Continual Learning, Catastrophic Forgetting, Multi-domain Learning

---

## 🎯 学习路径

### 初学者路径
```
01 迁移学习基础 → 02 PEFT (LoRA部分) → 实践项目
```
**时间**: 6-8 小时
**目标**: 掌握基础微调和 LoRA
**最低完成标准**: 完成一个 LoRA 微调实验并给出基线对比

### 进阶路径
```
01 迁移学习基础 → 02 PEFT (完整) → 03 领域适应 → 综合项目
```
**时间**: 11-14 小时
**目标**: 掌握所有微调技术
**最低完成标准**: 给出完整选型建议（效果、显存、训练时间三维）

### 研究者路径
```
完整学习所有内容 → 深入研究 PEFT 变体 → 探索新方法
```
**时间**: 15+ 小时
**目标**: 深入理解并能创新
**最低完成标准**: 设计并验证一个 PEFT 或持续学习改进点

---

## 🛠️ 实践项目建议

### 项目 1: 情感分析微调
**难度**: ⭐⭐⭐
**技术**: 基础微调 + LoRA
**数据集**: IMDb, SST-2
**时间**: 2-3 小时

### 项目 2: 医学文本分类
**难度**: ⭐⭐⭐⭐
**技术**: DAPT + TAPT + PEFT
**数据集**: PubMed, MIMIC
**时间**: 4-6 小时

### 项目 3: 多任务学习系统
**难度**: ⭐⭐⭐⭐⭐
**技术**: 多任务 PEFT + 持续学习
**数据集**: GLUE (General Language Understanding Evaluation, 通用语言理解评估) benchmark
**时间**: 8-10 小时

---

## 📊 知识图谱

```
微调技术
├── 基础微调
│   ├── 迁移学习理论
│   ├── 特征提取
│   ├── 端到端微调
│   └── 学习率调度
├── 参数高效微调 (PEFT)
│   ├── LoRA (低秩适配)
│   ├── Adapter (瓶颈适配器)
│   ├── Prefix/Prompt Tuning
│   └── 高级方法 (QLoRA, IA³, BitFit)
└── 领域适应
    ├── DAPT/TAPT
    ├── 持续学习
    │   ├── EWC
    │   └── Experience Replay
    └── 多领域学习
```

---

## 🔗 相关资源

### 📄 核心论文

**LoRA**:
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

**Adapter**:
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)

**Prefix-Tuning**:
- [Prefix-Tuning: Optimizing Continuous Prompts](https://arxiv.org/abs/2101.00190)

**EWC**:
- [Overcoming Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1612.00796)

**DAPT/TAPT**:
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)

### 💻 代码库

- [Hugging Face PEFT](https://github.com/huggingface/peft) - 官方 PEFT 库
- [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers) - Adapter 实现

### 📚 博客文章

- [Hugging Face PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 详解](https://huggingface.co/blog/lora)
- [Parameter-Efficient Fine-Tuning](https://magazine.sebastianraschka.com/p/understanding-parameter-efficient)

---

## ❓ 常见问题

### Q1: 什么时候应该使用 PEFT 而不是全量微调？

**A**: 当满足以下条件时优先考虑 PEFT：
- 计算资源有限（GPU 内存不足）
- 需要部署多个任务模型
- 模型参数量很大（>1B）
- 训练数据相对较少

### Q2: LoRA 和 Adapter 哪个更好？

**A**: 取决于场景：
- **推理速度优先** → LoRA（可合并权重，无额外延迟）
- **多任务切换** → Adapter（模块化，易于切换）
- **参数效率** → LoRA（更少参数，0.1-1%）
- **训练稳定性** → Adapter（更稳定）

### Q3: 如何选择 LoRA 的 rank？

**A**: 经验法则：
- 简单任务：r=4
- 一般任务：r=8（推荐默认值）
- 复杂任务：r=16-32
- 大模型：r=64

### Q4: DAPT 需要多少数据？

**A**: 建议：
- 最少：100K-1M tokens
- 推荐：10M+ tokens
- 理想：100M+ tokens

### Q5: 如何避免灾难性遗忘？

**A**: 三种主要方法：
- **正则化**: EWC, L2
- **重放**: Experience Replay
- **架构**: Adapter per task

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

## 📈 模块质量

| Notebook | 评分 | 状态 |
|----------|------|------|
| 01_transfer_learning | 92-94/100 | ⭐⭐⭐⭐⭐ |
| 02_parameter_efficient_finetuning | 100/100 | ⭐⭐⭐⭐⭐ |
| 03_domain_adaptation | 88+/100 | ⭐⭐⭐⭐⭐ |

**模块平均分**: 93+/100 🏆

---

## 🚀 下一步

完成本模块后，建议继续学习：

- **Module 6**: 高级训练技术（分布式训练、混合精度）
- **Module 7**: 模型部署与优化（推理加速、模型压缩）
- **Module 8**: 实际应用（RAG 系统、Agent 开发）

---

**模块维护者**: AI Learning Team
**最后更新**: 2025-02-11
**版本**: 2.0 (大幅改进版)
**反馈**: 欢迎提出改进建议
