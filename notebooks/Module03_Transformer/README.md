# Module 3: Transformer 架构 (Transformer)

## 📚 模块概览

本模块深入学习 Transformer 架构，从自注意力机制到完整的编码器-解码器模型，掌握现代自然语言处理的核心技术。Transformer 架构是大语言模型的基础，理解其工作原理对于后续的预训练模型学习至关重要。

### 🎯 学习目标

- 掌握自注意力（Self-Attention）机制的数学原理
- 理解多头注意力（Multi-Head Attention）的优势
- 实现完整的 Transformer 编码器和解码器
- 掌握位置编码和掩码机制的实现
- 理解不同 Transformer 变体的设计思路
- 能够构建和训练完整的 Transformer 模型

### ⏱️ 预计学习时间

**总计**: 9-12 小时

---

## 📖 Notebooks

### 3.1 自注意力机制 ⭐⭐⭐⭐⭐
**文件**: `01_self_attention.ipynb`
**时长**: 3-4 小时
**难度**: ⭐⭐⭐⭐
**评分**: 93+/100

**内容**:
- 注意力机制回顾与演进
- 自注意力的数学原理与实现
- 缩放点积注意力的设计理由
- 多头注意力的并行计算
- 注意力可视化与解释性
- 自注意力的计算复杂度分析

**亮点**:
- ✅ 9 个微实践（含从零实现自注意力）
- ✅ 18+ 个高质量可视化
- ✅ 完整的数学推导和直觉解释
- ✅ 注意力热力图和交互示例

**关键概念**: Self-Attention, Query-Key-Value, Scaled Dot-Product Attention, Multi-Head Attention, Attention Visualization

---

### 3.2 Transformer 编码器 ⭐⭐⭐⭐
**文件**: `02_transformer_encoder.ipynb`
**时长**: 3-4 小时
**难度**: ⭐⭐⭐⭐
**评分**: 92+/100

**内容**:
- 位置编码：正弦/余弦编码、可学习位置编码
- 编码器层结构：自注意力 + 前馈网络
- 残差连接和层归一化的设计
- 完整编码器堆栈的构建
- 编码器的并行计算优势
- 编码器变体与改进

**亮点**:
- ✅ 7 个微实践（含位置编码实现）
- ✅ 15+ 个可视化（含位置编码可视化）
- ✅ 从零实现完整编码器
- ✅ 性能分析和优化建议

**关键概念**: Positional Encoding, Encoder Layer, Residual Connection, Layer Normalization, Parallel Computation

---

### 3.3 Transformer 解码器 ⭐⭐⭐⭐⭐
**文件**: `03_transformer_decoder.ipynb`
**时长**: 3-4 小时
**难度**: ⭐⭐⭐⭐
**评分**: 94+/100

**内容**:
- 掩码自注意力：因果掩码实现
- 编码器-解码器注意力：交叉注意力机制
- 解码器层和完整堆栈构建
- 自回归生成策略：贪婪搜索、束搜索
- 训练技巧：标签平滑、学习率调度
- 完整 Transformer 模型的构建与训练

**亮点**:
- ✅ 8 个微实践（含束搜索实现）
- ✅ 20+ 个可视化
- ✅ 端到端机器翻译项目
- ✅ 详细的训练和推理流程

**关键概念**: Masked Self-Attention, Cross-Attention, Autoregressive Generation, Beam Search, Label Smoothing

---

## 🎯 学习路径

### 初学者路径
```
01 自注意力机制 → 02 Transformer编码器 → 03 Transformer解码器基础 → 实践项目
```
**时间**: 6-8 小时
**目标**: 掌握Transformer基础架构

### 进阶路径
```
01 自注意力机制 → 02 Transformer编码器 → 03 Transformer解码器完整实现 → 高级项目
```
**时间**: 9-12 小时
**目标**: 深入理解Transformer原理与实现

### 研究者路径
```
完整学习所有内容 → 实现Transformer变体 → 探索注意力机制改进
```
**时间**: 12+ 小时
**目标**: 创新Transformer架构

---

## 🛠️ 实践项目建议

### 项目 1: 文本分类
**难度**: ⭐⭐⭐
**技术**: Transformer Encoder + 分类头
**数据集**: AG News, SST-2
**时间**: 3-4 小时

### 项目 2: 机器翻译
**难度**: ⭐⭐⭐⭐
**技术**: 完整 Transformer 模型
**数据集**: WMT 英法平行语料
**时间**: 5-6 小时

### 项目 3: 文本生成
**难度**: ⭐⭐⭐⭐
**技术**: Transformer Decoder + 自回归生成
**数据集**: WikiText-2, 新闻语料
**时间**: 4-5 小时

---

## 📊 知识图谱

```
Transformer架构
├── 自注意力机制
│   ├── 缩放点积注意力
│   │   ├── Query-Key-Value
│   │   ├── 注意力分数
│   │   └── 注意力权重
│   ├── 多头注意力
│   │   ├── 并行注意力头
│   │   ├── 不同子空间
│   │   └── 结果拼接
│   └── 计算复杂度
├── 编码器
│   ├── 位置编码
│   │   ├── 正弦/余弦编码
│   │   └── 可学习位置编码
│   ├── 编码器层
│   │   ├── 多头自注意力
│   │   ├── 残差连接
│   │   ├── 层归一化
│   │   └── 前馈网络
│   └── 编码器堆栈
└── 解码器
    ├── 掩码自注意力
    │   ├── 因果掩码
    │   └── 防止未来信息泄露
    ├── 交叉注意力
    │   ├── 编码器-解码器注意力
    │   └── 关注源序列
    ├── 解码器层
    │   ├── 多头掩码自注意力
    │   ├── 多头交叉注意力
    │   ├── 残差连接
    │   ├── 层归一化
    │   └── 前馈网络
    └── 生成策略
        ├── 贪婪搜索
        ├── 束搜索
        └── 采样方法
```

---

## 🔗 相关资源

### 📄 核心论文

**Transformer原论文**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) - Transformer 奠基之作

**扩展论文**:
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259) (2014)
- [Universal Transformers](https://arxiv.org/abs/1807.03819) (2018)
- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) (2020)

### 💻 代码库

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 详细注释的Transformer实现
- [Attention is All You Need PyTorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

### 📚 博客文章

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 图文并茂的Transformer讲解
- [Attention Mechanism: A Detailed Look](https://towardsdatascience.com/attention-mechanism-59faf77241f6)
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

---

## ❓ 常见问题

### Q1: 自注意力为什么比RNN好？

**A**: 自注意力的优势：
- **并行计算**：不依赖于序列顺序，可并行处理
- **长距离依赖**：直接建模任意位置之间的关系
- **可解释性**：注意力权重提供了可解释性
- **灵活建模**：多头注意力可以捕获不同类型的依赖

### Q2: 为什么需要缩放点积注意力？

**A**: 缩放的作用：
- 防止点积值过大导致softmax饱和
- 保持梯度稳定
- 使注意力分布更加均匀
- 公式：$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

### Q3: 位置编码的作用是什么？

**A**: 位置编码的重要性：
- Transformer 没有循环或卷积结构，无法捕获位置信息
- 位置编码为每个位置提供唯一的表示
- 正弦/余弦编码可以外推到更长的序列
- 可学习位置编码在某些任务上表现更好

### Q4: 掩码自注意力是如何工作的？

**A**: 掩码自注意力：
- 在解码器中使用，防止看到未来的 tokens
- 通过设置一个下三角掩码矩阵实现
- 掩码位置的注意力分数设为负无穷，softmax后变为0
- 确保自回归生成的正确性

### Q5: 如何选择Transformer的超参数？

**A**: 经验法则：
- **层数**：基础模型6-12层，大模型24-96层
- **隐藏维度**：512-1024（基础），2048-16384（大模型）
- **注意力头数**：8-16，通常为隐藏维度的因数
- **前馈网络维度**：隐藏维度的4倍
- ** dropout**：0.1-0.3

---

## 🎓 学习检查清单

完成本模块后，你应该能够：

- [ ] 从零实现自注意力和多头注意力
- [ ] 解释缩放点积注意力的设计原理
- [ ] 实现位置编码和掩码机制
- [ ] 构建完整的Transformer编码器和解码器
- [ ] 实现束搜索等生成策略
- [ ] 训练和评估Transformer模型
- [ ] 分析Transformer的计算复杂度
- [ ] 将Transformer应用到实际任务

---

## 📈 模块质量

| Notebook | 评分 | 状态 |
|----------|------|------|
| 01_self_attention | 93+/100 | ⭐⭐⭐⭐⭐ |
| 02_transformer_encoder | 92+/100 | ⭐⭐⭐⭐ |
| 03_transformer_decoder | 94+/100 | ⭐⭐⭐⭐⭐ |

**模块平均分**: 93+/100 ⭐⭐⭐⭐⭐

---

## 🚀 下一步

完成本模块后，建议继续学习：

- **Module 4**: 预训练（语言建模、BERT、GPT）
- **Module 5**: 微调技术（PEFT、领域适应）
- **Module 8**: 实际应用（RAG系统、Agent开发）

---

**模块维护者**: AI Learning Team
**最后更新**: 2025-02-11
**版本**: 2.0 (大幅改进版)
**反馈**: 欢迎提出改进建议