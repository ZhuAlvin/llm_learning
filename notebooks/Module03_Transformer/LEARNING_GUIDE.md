# Module 3: Transformer 架构 - 学习指南

## 📋 文档质量检查报告

### ✅ 已完成内容

**Notebook 03_transformer_decoder.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和架构图
- ✅ 完整的理论讲解（掩码注意力、交叉注意力）
- ✅ 9 个 Micro Practice 实践练习
- ✅ 可视化（掩码模式、注意力权重）
- ✅ 完整实现（解码器层、完整 Transformer）
- ✅ 工程实践（标签平滑、学习率调度）
- ✅ 生成策略（贪婪解码、束搜索）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

### ⚠️ 待完成内容

**Notebook 01_self_attention.ipynb** - 未创建
- 需要：自注意力机制详解
- 需要：多头注意力实现
- 需要：注意力可视化

**Notebook 02_transformer_encoder.ipynb** - 未创建
- 需要：位置编码详解
- 需要：编码器层实现
- 需要：完整编码器堆栈

### 📊 质量评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **内容完整性** | 7/10 | 解码器完整，缺少编码器和自注意力 |
| **理论深度** | 9/10 | 数学公式清晰，原理讲解透彻 |
| **代码质量** | 9/10 | 实现规范，注释详细 |
| **实践练习** | 10/10 | Micro Practice 设计优秀 |
| **可视化** | 9/10 | 掩码和注意力可视化清晰 |
| **工程实践** | 8/10 | 包含训练技巧，可补充更多 |

**总体评分：8.5/10** - 已完成部分质量优秀，需补充前两个 notebook

---

## 🎯 学习指南

### 学习路径

```
第1周：自注意力机制
  ├─ 理解注意力机制回顾
  ├─ 掌握 Q、K、V 概念
  ├─ 实现缩放点积注意力
  └─ 实现多头注意力

第2周：Transformer 编码器
  ├─ 理解位置编码
  ├─ 实现编码器层
  ├─ 构建编码器堆栈
  └─ 应用到文本分类

第3周：Transformer 解码器
  ├─ 理解掩码注意力
  ├─ 实现交叉注意力
  ├─ 构建完整 Transformer
  └─ 实现生成策略

第4周：综合项目
  ├─ 机器翻译任务
  ├─ 训练和评估
  ├─ 优化和调试
  └─ 部署应用
```

### 前置知识检查

在开始学习前，确保你已经掌握：

- [ ] Python 编程基础
- [ ] PyTorch 基础（张量操作、自动微分）
- [ ] 线性代数（矩阵乘法、向量运算）
- [ ] 深度学习基础（反向传播、优化器）
- [ ] RNN 和 Seq2Seq 模型（Module 2）

### 学习建议

#### 1. 理论学习（40%时间）

**必读材料**：
- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原论文
- 📖 [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化讲解
- 📖 [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 带注释的实现

**学习方法**：
1. 先看可视化教程理解整体架构
2. 阅读原论文理解数学细节
3. 对照代码实现加深理解

#### 2. 代码实践（50%时间）

**实践步骤**：

**Week 1: 自注意力**
```python
# 练习1：实现缩放点积注意力
def scaled_dot_product_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

# 练习2：实现多头注意力
class MultiHeadAttention(nn.Module):
    # 将注意力分成多个头
    # 并行计算，最后拼接
```

**Week 2: 编码器**
```python
# 练习3：实现位置编码
class PositionalEncoding(nn.Module):
    # 使用正弦/余弦函数
    # 注入位置信息

# 练习4：实现编码器层
class EncoderLayer(nn.Module):
    # 自注意力 + FFN
    # 残差连接 + 层归一化
```

**Week 3: 解码器**
```python
# 练习5：实现掩码注意力
def masked_attention(Q, K, V, mask):
    # 应用因果掩码
    # 防止看到未来

# 练习6：实现交叉注意力
def cross_attention(Q_dec, K_enc, V_enc):
    # Q 来自解码器
    # K, V 来自编码器
```

**Week 4: 完整模型**
```python
# 练习7：构建完整 Transformer
class Transformer(nn.Module):
    def __init__(self):
        self.encoder = TransformerEncoder(...)
        self.decoder = TransformerDecoder(...)

# 练习8：实现束搜索
def beam_search(model, src, beam_size=5):
    # 保留 k 个最佳候选
    # 选择总体得分最高的序列
```

#### 3. 项目实战（10%时间）

**推荐项目**：

1. **文本分类**（简单）
   - 使用 Transformer Encoder
   - 数据：IMDB 情感分类
   - 目标：准确率 > 85%

2. **机器翻译**（中等）
   - 使用完整 Transformer
   - 数据：WMT14 英德翻译
   - 目标：BLEU > 25

3. **文本生成**（困难）
   - 使用 Transformer Decoder
   - 数据：维基百科
   - 目标：生成连贯文本

### 常见问题解答

#### Q1: 为什么 Transformer 比 RNN 更好？

**A:** 主要优势：
- **并行化**：可以并行处理整个序列，RNN 必须顺序处理
- **长程依赖**：注意力机制直接连接任意位置，RNN 需要多步传递
- **训练速度**：GPU 利用率高，训练快 10-100 倍
- **性能**：在大多数 NLP 任务上超越 RNN

#### Q2: 自注意力和交叉注意力有什么区别？

**A:**
- **自注意力**：Q、K、V 来自同一序列，用于序列内部的关系建模
- **交叉注意力**：Q 来自一个序列，K、V 来自另一个序列，用于序列间的对齐

#### Q3: 为什么需要多头注意力？

**A:**
- 不同的头可以关注不同的模式（语法、语义、位置等）
- 增加模型的表达能力
- 类似于 CNN 中的多个卷积核

#### Q4: 位置编码为什么使用正弦/余弦函数？

**A:**
- 可以外推到训练时未见过的长度
- 相对位置关系可以通过线性变换表示
- 不需要学习参数

#### Q5: 如何选择超参数？

**A:** 常见配置：
- **小模型**：d_model=256, num_heads=4, num_layers=4
- **基础模型**：d_model=512, num_heads=8, num_layers=6
- **大模型**：d_model=1024, num_heads=16, num_layers=12

### 调试技巧

#### 1. 形状不匹配

```python
# 常见错误
RuntimeError: mat1 and mat2 shapes cannot be multiplied

# 调试方法
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"Expected: Q @ K.T = ({Q.shape[0]}, {Q.shape[1]}) @ ({K.shape[1]}, {K.shape[0]})")
```

#### 2. 掩码错误

```python
# 检查掩码
mask = create_causal_mask(seq_len)
plt.imshow(mask)
plt.show()

# 验证：位置 i 只能看到 j <= i
assert mask[i, j] == 0 for all j > i
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

#### 4. 注意力权重异常

```python
# 可视化注意力
attn_weights = model.get_attention_weights()
plt.imshow(attn_weights[0].detach())
plt.colorbar()
plt.show()

# 检查：每行和应该为 1
assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))
```

### 性能优化

#### 1. 内存优化

```python
# 使用梯度检查点
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self.layer, x)

# 减少批次大小
batch_size = 16  # 而不是 32

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
```

#### 2. 速度优化

```python
# 使用 DataLoader 并行加载
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# 使用 torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 使用 Flash Attention
# pip install flash-attn
from flash_attn import flash_attn_func
```

### 扩展阅读

#### 进阶主题

1. **Transformer 变体**
   - Reformer：降低内存复杂度
   - Linformer：线性复杂度注意力
   - Performer：快速注意力近似

2. **预训练技术**
   - BERT：掩码语言模型
   - GPT：因果语言模型
   - T5：统一的文本到文本框架

3. **高效 Transformer**
   - Sparse Attention：稀疏注意力模式
   - Low-rank Attention：低秩分解
   - Kernel Methods：核方法近似

#### 推荐资源

**视频课程**：
- Stanford CS224N: NLP with Deep Learning
- Hugging Face Transformers Course
- DeepLearning.AI: NLP Specialization

**代码库**：
- Hugging Face Transformers
- fairseq (Facebook)
- tensor2tensor (Google)

**论文列表**：
- Attention Is All You Need (2017)
- BERT (2018)
- GPT-2 (2019)
- T5 (2019)
- GPT-3 (2020)

### 评估标准

完成本模块后，你应该能够：

- [ ] 解释自注意力机制的工作原理
- [ ] 实现多头注意力模块
- [ ] 构建完整的 Transformer 编码器
- [ ] 构建完整的 Transformer 解码器
- [ ] 理解掩码注意力和交叉注意力的区别
- [ ] 实现位置编码
- [ ] 实现贪婪解码和束搜索
- [ ] 训练 Transformer 模型完成序列到序列任务
- [ ] 调试和优化 Transformer 模型
- [ ] 阅读和理解 Transformer 相关论文

### 下一步

完成 Module 3 后，建议：

1. **巩固基础**：重新实现一遍核心组件
2. **项目实践**：完成至少一个完整项目
3. **阅读论文**：深入理解 Transformer 变体
4. **学习 Module 4**：预训练语言模型（BERT, GPT）

---

## 📝 学习检查清单

### Week 1: 自注意力
- [ ] 理解 Q、K、V 的含义
- [ ] 实现缩放点积注意力
- [ ] 实现多头注意力
- [ ] 可视化注意力权重

### Week 2: 编码器
- [ ] 理解位置编码原理
- [ ] 实现位置编码
- [ ] 实现编码器层
- [ ] 构建编码器堆栈

### Week 3: 解码器
- [ ] 理解因果掩码
- [ ] 实现掩码注意力
- [ ] 实现交叉注意力
- [ ] 构建完整 Transformer

### Week 4: 实战
- [ ] 实现训练循环
- [ ] 实现束搜索
- [ ] 完成翻译项目
- [ ] 优化模型性能

---

**祝学习顺利！** 🚀

如有问题，请参考：
- 📖 Notebook 中的 FAQ 部分
- 💬 课程讨论区
- 🔍 Stack Overflow
- 📧 联系助教
