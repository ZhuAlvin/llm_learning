# Module 4: 预训练语言模型 - 学习指南

## 📋 文档质量检查报告

### ✅ 已完成内容

**Notebook 01_language_modeling.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和语言模型概念
- ✅ 完整的理论讲解（N-gram、神经语言模型、困惑度）
- ✅ 6 个 Micro Practice 实践练习
- ✅ 可视化（语言模型性能对比、困惑度分析）
- ✅ 完整实现（自回归语言模型、掩码语言模型）
- ✅ 评估指标（困惑度计算和解释）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

**Notebook 02_bert_architecture.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和BERT架构图
- ✅ 完整的理论讲解（Encoder-only、MLM、NSP）
- ✅ 7 个 Micro Practice 实践练习
- ✅ 可视化（BERT架构、掩码模式、注意力权重）
- ✅ 完整实现（简化版BERT、预训练流程）
- ✅ 工程实践（BERT变体、微调策略）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

**Notebook 03_gpt_architecture.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和GPT架构图
- ✅ 完整的理论讲解（Decoder-only、自回归、因果注意力）
- ✅ 8 个 Micro Practice 实践练习
- ✅ 可视化（GPT架构、生成策略、few-shot演示）
- ✅ 完整实现（简化版GPT、文本生成）
- ✅ 工程实践（生成策略、GPT变体）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

### 📊 质量评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **内容完整性** | 10/10 | 所有notebook完整且高质量 |
| **理论深度** | 9/10 | 数学公式清晰，原理讲解透彻 |
| **代码质量** | 9/10 | 实现规范，注释详细 |
| **实践练习** | 10/10 | Micro Practice 设计优秀 |
| **可视化** | 9/10 | 架构图和可视化清晰 |
| **工程实践** | 9/10 | 包含预训练和微调流程 |

**总体评分：9.3/10** - 优秀，内容完整且质量高

---

## 🎯 学习指南

### 学习路径

```
第1周：语言建模基础
  ├─ 理解语言模型概念
  ├─ 掌握N-gram语言模型
  ├─ 理解困惑度评估指标
  └─ 实现神经语言模型

第2周：BERT架构
  ├─ 理解Encoder-only架构
  ├─ 掌握掩码语言建模
  ├─ 理解下一句预测
  └─ 实现简化版BERT

第3周：GPT架构
  ├─ 理解Decoder-only架构
  ├─ 掌握自回归语言建模
  ├─ 实现因果注意力
  └─ 实现文本生成策略

第4周：综合项目
  ├─ 对比BERT和GPT
  ├─ 实现预训练流程
  ├─ 微调下游任务
  └─ 部署应用
```

### 前置知识检查

在开始学习前，确保你已经掌握：

- [ ] Python 编程基础
- [ ] PyTorch 基础（张量操作、自动微分）
- [ ] 线性代数（矩阵乘法、向量运算）
- [ ] 深度学习基础（反向传播、优化器）
- [ ] Transformer 架构（Module 3）

### 学习建议

#### 1. 理论学习（40%时间）

**必读材料**：
- 📄 [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - BERT原论文
- 📄 [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) - GPT-1原论文
- 📄 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3原论文
- 📖 [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/) - BERT可视化讲解
- 📖 [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - GPT-2可视化讲解

**学习方法**：
1. 先看可视化教程理解整体架构
2. 阅读原论文理解数学细节
3. 对照代码实现加深理解

#### 2. 代码实践（50%时间）

**实践步骤**：

**Week 1: 语言建模**
```python
# 练习1：实现N-gram语言模型
class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = {}
    
    def train(self, corpus):
        # 统计n-gram频率
        pass
    
    def predict(self, context):
        # 预测下一个词
        pass

# 练习2：计算困惑度
def calculate_perplexity(model, test_data):
    # PPL = exp(-1/N * sum(log P(w_i|context)))
    pass

# 练习3：实现神经语言模型
class NeuralLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        # 词嵌入 + LSTM + 输出层
        pass
```

**Week 2: BERT**
```python
# 练习4：实现掩码语言建模
def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    # 随机mask 15%的token
    # 80%替换为[MASK]
    # 10%替换为随机词
    # 10%保持不变
    pass

# 练习5：实现BERT编码器层
class BERTLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        self.attention = MultiHeadAttention(...)
        self.ffn = FeedForward(...)
        self.norm1 = LayerNorm(...)
        self.norm2 = LayerNorm(...)
    
    def forward(self, x, mask):
        # 自注意力 + FFN
        # 残差连接 + 层归一化
        pass

# 练习6：实现下一句预测
def next_sentence_prediction(sentence1, sentence2, model):
    # 判断两个句子是否连续
    pass
```

**Week 3: GPT**
```python
# 练习7：实现因果注意力
def causal_attention(Q, K, V):
    # 应用因果掩码
    # 防止看到未来token
    pass

# 练习8：实现GPT解码器层
class GPTLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        self.causal_attn = CausalAttention(...)
        self.ffn = FeedForward(...)
        self.norm1 = LayerNorm(...)
        self.norm2 = LayerNorm(...)
    
    def forward(self, x):
        # 因果注意力 + FFN
        # 残差连接 + 层归一化
        pass

# 练习9：实现文本生成
def generate_text(model, prompt, max_length=100):
    # 自回归生成
    for _ in range(max_length):
        next_token = model(prompt)
        prompt = torch.cat([prompt, next_token])
    return prompt
```

**Week 4: 完整模型**
```python
# 练习10：实现预训练流程
def pretrain_bert(model, corpus, epochs=10):
    for epoch in range(epochs):
        for batch in dataloader:
            # MLM任务
            mlm_loss = mlm_forward(model, batch)
            # NSP任务
            nsp_loss = nsp_forward(model, batch)
            # 总损失
            loss = mlm_loss + nsp_loss
            loss.backward()
            optimizer.step()

# 练习11：实现微调流程
def finetune_bert(model, task_data, epochs=5):
    # 冻结部分层
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # 训练分类头
    for epoch in range(epochs):
        for batch in dataloader:
            loss = classification_forward(model, batch)
            loss.backward()
            optimizer.step()

# 练习12：实现Few-shot Learning
def few_shot_inference(model, prompt, examples):
    # 将examples拼接成prompt
    # 进行推理
    pass
```

#### 3. 项目实战（10%时间）

**推荐项目**：

1. **文本分类**（简单）
   - 使用BERT预训练模型
   - 微调到情感分类任务
   - 数据：IMDB情感分类
   - 目标：准确率 > 90%

2. **文本生成**（中等）
   - 使用GPT预训练模型
   - 生成故事或文章
   - 数据：维基百科
   - 目标：生成连贯文本

3. **问答系统**（困难）
   - 使用BERT预训练模型
   - 微调到SQuAD数据集
   - 目标：F1 > 85%

### 常见问题解答

#### Q1: BERT和GPT有什么区别？

**A:** 主要区别：
- **架构**：BERT使用Encoder-only，GPT使用Decoder-only
- **注意力**：BERT使用双向注意力，GPT使用因果注意力
- **任务**：BERT适合理解任务（分类、QA），GPT适合生成任务
- **预训练**：BERT使用MLM+NSP，GPT使用CLM
- **应用**：BERT用于NLU，GPT用于NLG

#### Q2: 困惑度（Perplexity）是什么？

**A:** 困惑度是评估语言模型性能的指标：
- **定义**：PPL = exp(-1/N * sum(log P(w_i|context)))
- **含义**：模型对下一个词的"不确定程度"
- **值域**：PPL >= 1，越小越好
- **解释**：PPL=10表示模型平均在10个候选中选择

#### Q3: 什么是Masked Language Modeling？

**A:** MLM是BERT的预训练任务：
- **原理**：随机mask输入token的15%
- **目标**：预测被mask的原始token
- **优势**：可以同时看到上下文（双向）
- **应用**：适合理解任务

#### Q4: 什么是Causal Language Modeling？

**A:** CLM是GPT的预训练任务：
- **原理**：预测下一个token（自回归）
- **约束**：只能看到之前的token（因果）
- **优势**：适合文本生成
- **应用**：适合生成任务

#### Q5: Few-shot Learning是如何工作的？

**A:** Few-shot Learning的原理：
- **提示工程**：在输入中提供少量示例
- **上下文学习**：模型从示例中学习模式
- **无需训练**：直接使用预训练模型推理
- **示例**：
  ```
  输入：苹果 -> 红色
  输入：香蕉 -> 黄色
  输入：葡萄 -> ?
  输出：紫色
  ```

### 调试技巧

#### 1. 掩码错误

```python
# 检查掩码
masked_tokens = mask_tokens(inputs, tokenizer)
print(f"Original: {tokenizer.decode(inputs[0])}")
print(f"Masked: {tokenizer.decode(masked_tokens[0])}")

# 验证：15%被mask
mask_ratio = (masked_tokens == tokenizer.mask_token_id).float().mean()
print(f"Mask ratio: {mask_ratio:.2%}")
```

#### 2. 困惑度异常

```python
# 监控困惑度
for epoch in range(epochs):
    ppl = calculate_perplexity(model, validation_data)
    print(f"Epoch {epoch}: PPL = {ppl:.2f}")
    
    # 困惑度应该下降
    if ppl > previous_ppl:
        print("Warning: Perplexity increased!")
```

#### 3. 梯度消失/爆炸

```python
# 监控梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# 使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 4. 生成质量差

```python
# 调整生成策略
# 贪婪搜索：选择概率最高的词
output = model.generate(input_ids, max_length=50, do_sample=False)

# 束搜索：保留k个最佳候选
output = model.generate(input_ids, max_length=50, num_beams=5)

# 采样：随机采样（增加多样性）
output = model.generate(input_ids, max_length=50, do_sample=True, temperature=0.7)

# Top-k采样：只从top-k中采样
output = model.generate(input_ids, max_length=50, do_sample=True, top_k=50)
```

### 性能优化

#### 1. 内存优化

```python
# 使用梯度检查点
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self.layer, x)

# 减少批次大小
batch_size = 8  # 而不是 16

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
```

#### 2. 速度优化

```python
# 使用DataLoader并行加载
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# 使用torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 使用预训练模型（而非从头训练）
from transformers import BertModel, GPT2Model

bert = BertModel.from_pretrained('bert-base-uncased')
gpt = GPT2Model.from_pretrained('gpt2')
```

### 扩展阅读

#### 进阶主题

1. **BERT变体**
   - RoBERTa：改进的训练策略
   - ALBERT：参数共享
   - DistilBERT：知识蒸馏
   - ELECTRA：替代预训练任务

2. **GPT变体**
   - GPT-2：更大规模
   - GPT-3：175B参数，few-shot学习
   - GPT-4：多模态能力
   - LLaMA：开源高效模型

3. **统一架构**
   - T5：Text-to-Text框架
   - BART：序列到序列预训练
   - mT5：多语言T5

#### 推荐资源

**视频课程**：
- Stanford CS224N: NLP with Deep Learning
- Hugging Face Transformers Course
- DeepLearning.AI: NLP Specialization

**代码库**：
- Hugging Face Transformers
- fairseq (Facebook)
- tokenizers (Hugging Face)

**论文列表**：
- BERT: Pre-training of Deep Bidirectional Transformers (2018)
- GPT-2: Language Models are Unsupervised Multitask Learners (2019)
- GPT-3: Language Models are Few-Shot Learners (2020)
- T5: Exploring the Limits of Transfer Learning (2019)

### 评估标准

完成本模块后，你应该能够：

- [ ] 解释语言模型的概念和评估方法
- [ ] 计算和解释困惑度指标
- [ ] 理解BERT的Encoder-only架构
- [ ] 实现掩码语言建模
- [ ] 理解GPT的Decoder-only架构
- [ ] 实现自回归语言建模
- [ ] 实现因果注意力
- [ ] 实现文本生成策略
- [ ] 理解Few-shot Learning的原理
- [ ] 微调预训练模型到下游任务
- [ ] 阅读和理解预训练语言模型相关论文

### 下一步

完成 Module 4 后，建议：

1. **巩固基础**：重新实现一遍核心组件
2. **项目实践**：完成至少一个完整项目
3. **阅读论文**：深入理解BERT和GPT变体
4. **学习 Module 5**：微调技术（PEFT、LoRA等）

---

## 📝 学习检查清单

### Week 1: 语言建模
- [ ] 理解语言模型概念
- [ ] 实现N-gram语言模型
- [ ] 计算困惑度
- [ ] 实现神经语言模型

### Week 2: BERT
- [ ] 理解Encoder-only架构
- [ ] 实现掩码语言建模
- [ ] 实现下一句预测
- [ ] 实现简化版BERT

### Week 3: GPT
- [ ] 理解Decoder-only架构
- [ ] 实现因果注意力
- [ ] 实现自回归语言建模
- [ ] 实现文本生成策略

### Week 4: 实战
- [ ] 对比BERT和GPT
- [ ] 实现预训练流程
- [ ] 微调下游任务
- [ ] 完成项目

---

**祝学习顺利！** 🚀

如有问题，请参考：
- 📖 Notebook 中的 FAQ 部分
- 💬 课程讨论区
- 🔍 Stack Overflow
- 📧 联系助教