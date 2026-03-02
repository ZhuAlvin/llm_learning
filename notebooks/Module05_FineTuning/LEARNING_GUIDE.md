# Module 5: 微调技术 - 学习指南

## 📋 文档质量检查报告

### ✅ 已完成内容

**Notebook 01_transfer_learning.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和迁移学习理论
- ✅ 完整的理论讲解（迁移学习、微调、学习率调度）
- ✅ 5 个 Micro Practice 实践练习
- ✅ 可视化（迁移学习效果、学习率调度）
- ✅ 完整实现（文本分类微调、训练器类）
- ✅ 工程实践（超参数选择、性能优化）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

**Notebook 02_parameter_efficient_finetuning.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和PEFT理论
- ✅ 完整的理论讲解（LoRA、Adapter、Prefix-Tuning）
- ✅ 11 个 Micro Practice 实践练习
- ✅ 可视化（PEFT方法对比、LoRA权重分解）
- ✅ 完整实现（所有主要PEFT方法）
- ✅ 工程实践（Hugging Face PEFT库、多任务部署）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

**Notebook 03_domain_adaptation.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和领域适应理论
- ✅ 完整的理论讲解（DAPT、TAPT、EWC、Experience Replay）
- ✅ 7 个 Micro Practice 实践练习
- ✅ 可视化（领域偏移、Fisher信息热图）
- ✅ 完整实现（DAPT/EWC、多领域学习）
- ✅ 工程实践（灾难性遗忘、持续学习）
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
| **工程实践** | 9/10 | 包含完整的微调流程 |

**总体评分：9.3/10** - 优秀，内容完整且质量高

---

## 🎯 学习指南

### 学习路径

```
第1周：迁移学习基础
  ├─ 理解迁移学习原理
  ├─ 掌握预训练-微调范式
  ├─ 学习学习率调度策略
  └─ 实现文本分类微调

第2周：参数高效微调
  ├─ 理解PEFT原理
  ├─ 掌握LoRA方法
  ├─ 学习Adapter和Prefix-Tuning
  └─ 实现多种PEFT方法

第3周：领域适应
  ├─ 理解领域偏移问题
  ├─ 掌握DAPT和TAPT
  ├─ 学习持续学习技术
  └─ 实现多领域学习

第4周：综合项目
  ├─ 选择合适的微调方法
  ├─ 实现完整的微调流程
  ├─ 评估和优化性能
  └─ 部署应用
```

### 前置知识检查

在开始学习前，确保你已经掌握：

- [ ] Python 编程基础
- [ ] PyTorch 基础（张量操作、自动微分）
- [ ] 线性代数（矩阵乘法、向量运算）
- [ ] 深度学习基础（反向传播、优化器）
- [ ] Transformer 架构（Module 3）
- [ ] 预训练语言模型（Module 4）

### 学习建议

#### 1. 理论学习（40%时间）

**必读材料**：
- 📄 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - LoRA原论文
- 📄 [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) - Adapter原论文
- 📄 [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) - Prompt-Tuning原论文
- 📖 [Hugging Face PEFT Guide](https://huggingface.co/docs/peft) - PEFT官方文档
- 📖 [A Gentle Introduction to Transfer Learning](https://ruder.io/transfer-learning/) - 迁移学习教程

**学习方法**：
1. 先理解迁移学习的基本概念
2. 深入学习PEFT方法的数学原理
3. 对比不同方法的优缺点
4. 阅读原论文理解细节

#### 2. 代码实践（50%时间）

**实践步骤**：

**Week 1: 迁移学习**
```python
# 练习1：实现迁移学习
class TransferLearningModel(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        self.backbone = pretrained_model
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# 练习2：实现学习率调度
def create_scheduler(optimizer, scheduler_type='linear'):
    if scheduler_type == 'linear':
        return get_linear_schedule_with_warmup(...)
    elif scheduler_type == 'cosine':
        return get_cosine_schedule_with_warmup(...)

# 练习3：实现微调训练器
class FineTuningTrainer:
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
    
    def train(self, epochs, learning_rate):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = create_scheduler(optimizer)
        
        for epoch in range(epochs):
            for batch in self.train_data:
                loss = self.model(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
```

**Week 2: PEFT**
```python
# 练习4：实现LoRA
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1.0 / rank
    
    def forward(self, x):
        return x @ (self.lora_A @ self.lora_B) * self.scaling

# 练习5：实现Adapter
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + x

# 练习6：实现Prefix-Tuning
class PrefixTuning(nn.Module):
    def __init__(self, num_layers, hidden_size, prefix_length=10):
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_layers, prefix_length, hidden_size)
        )
    
    def get_prefix(self):
        return self.prefix_embeddings

# 练习7：对比PEFT方法
def compare_peft_methods(model, methods=['lora', 'adapter', 'prefix']):
    results = {}
    for method in methods:
        peft_model = apply_peft(model, method)
        results[method] = evaluate(peft_model)
    return results
```

**Week 3: 领域适应**
```python
# 练习8：实现DAPT
def domain_adaptive_pretraining(model, domain_corpus, epochs=3):
    # 在领域数据上继续预训练
    for epoch in range(epochs):
        for batch in domain_corpus:
            loss = mlm_forward(model, batch)
            loss.backward()
            optimizer.step()

# 练习9：实现EWC
class EWC:
    def __init__(self, model, importance=1000):
        self.model = model
        self.importance = importance
        self.fisher = {}
    
    def compute_fisher(self, data):
        # 计算Fisher信息矩阵
        for name, param in self.model.named_parameters():
            self.fisher[name] = param.grad ** 2
    
    def penalty(self):
        # 计算EWC惩罚项
        penalty = 0
        for name, param in self.model.named_parameters():
            penalty += self.importance * self.fisher[name] * (param - self.old_params[name]) ** 2
        return penalty

# 练习10：实现Experience Replay
class ExperienceReplay:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add_experience(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# 练习11：实现多领域学习
def multi_domain_learning(model, domain_data):
    for domain, data in domain_data.items():
        # 为每个领域创建适配器
        adapter = AdapterLayer(hidden_size)
        model.add_adapter(domain, adapter)
        
        # 训练领域适配器
        for batch in data:
            loss = model(batch, domain=domain)
            loss.backward()
            optimizer.step()
```

**Week 4: 完整项目**
```python
# 练习12：实现完整的微调流程
def complete_finetuning_pipeline(model, task_data, method='lora'):
    # 1. 选择微调方法
    if method == 'lora':
        model = apply_lora(model)
    elif method == 'adapter':
        model = apply_adapter(model)
    
    # 2. 准备数据
    train_loader, val_loader = prepare_data(task_data)
    
    # 3. 训练模型
    trainer = FineTuningTrainer(model, train_loader, val_loader)
    trainer.train(epochs=5, learning_rate=2e-5)
    
    # 4. 评估模型
    metrics = evaluate(model, val_loader)
    
    return model, metrics

# 练习13：实现多任务PEFT
def multi_task_peft(model, tasks):
    # 为每个任务创建独立的LoRA适配器
    adapters = {}
    for task in tasks:
        adapters[task] = LoRALayer(hidden_size, hidden_size)
    
    # 训练多任务模型
    for task, data in tasks.items():
        model.set_adapter(adapters[task])
        for batch in data:
            loss = model(batch)
            loss.backward()
            optimizer.step()
```

#### 3. 项目实战（10%时间）

**推荐项目**：

1. **情感分析**（简单）
   - 使用LoRA微调BERT
   - 数据：IMDB情感分类
   - 目标：准确率 > 92%

2. **命名实体识别**（中等）
   - 使用Adapter微调BERT
   - 数据：CoNLL-2003
   - 目标：F1 > 88%

3. **多领域文本分类**（困难）
   - 使用多领域学习
   - 数据：多个领域的文本
   - 目标：所有领域F1 > 85%

### 常见问题解答

#### Q1: 迁移学习和微调有什么区别？

**A:** 
- **迁移学习**：广义概念，指将一个任务学到的知识应用到另一个任务
- **微调**：迁移学习的一种具体方法，在预训练模型基础上继续训练
- **关系**：微调是迁移学习的一种实现方式

#### Q2: 为什么需要PEFT？

**A:** PEFT的优势：
- **参数效率**：只训练少量参数（1-5%）
- **存储效率**：每个任务只需存储适配器权重
- **切换灵活**：可以快速切换不同任务的适配器
- **避免灾难性遗忘**：不修改原始模型权重

#### Q3: LoRA、Adapter、Prefix-Tuning如何选择？

**A:** 选择指南：
- **LoRA**：通用性强，适合大多数任务
- **Adapter**：适合需要保持模型结构不变的场景
- **Prefix-Tuning**：适合生成任务，需要较少参数

#### Q4: 什么是灾难性遗忘？

**A:** 灾难性遗忘：
- **定义**：学习新任务时忘记旧任务的知识
- **原因**：神经网络权重被新任务覆盖
- **解决方法**：EWC、Experience Replay、多任务学习

#### Q5: 如何选择学习率？

**A:** 学习率选择：
- **预训练模型**：通常使用较小的学习率（1e-5到5e-5）
- **从头训练**：可以使用较大的学习率（1e-4到1e-3）
- **微调策略**：使用学习率调度器（线性衰减、余弦退火）
- **经验法则**：从2e-5开始，根据验证集调整

### 调试技巧

#### 1. 过拟合

```python
# 监控训练和验证损失
for epoch in range(epochs):
    train_loss = train_epoch(model, train_data)
    val_loss = validate(model, val_data)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 如果验证损失上升，可能过拟合
    if val_loss > best_val_loss:
        print("Warning: Overfitting detected!")
        # 增加正则化
        # 减少模型复杂度
        # 使用早停
```

#### 2. 学习率不合适

```python
# 学习率范围测试
def lr_range_test(model, data, start_lr=1e-7, end_lr=1, num_iter=100):
    lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
    losses = []
    
    for lr in lrs:
        optimizer = AdamW(model.parameters(), lr=lr)
        loss = train_step(model, data, optimizer)
        losses.append(loss)
    
    # 绘制学习率-损失曲线
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.show()
```

#### 3. PEFT效果差

```python
# 检查PEFT配置
def check_peft_config(peft_model):
    # 检查可训练参数数量
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 检查适配器权重
    for name, param in peft_model.named_parameters():
        if 'lora' in name or 'adapter' in name:
            print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")
```

#### 4. 领域适应失败

```python
# 检查领域偏移
def detect_domain_shift(source_data, target_data):
    # 计算源域和目标域的分布差异
    source_dist = compute_distribution(source_data)
    target_dist = compute_distribution(target_data)
    
    # 使用KL散度衡量差异
    kl_divergence = compute_kl_divergence(source_dist, target_dist)
    print(f"Domain Shift (KL): {kl_divergence:.4f}")
    
    # 如果偏移大，需要领域适应
    if kl_divergence > threshold:
        print("Large domain shift detected, consider DAPT or TAPT")
```

### 性能优化

#### 1. 内存优化

```python
# 使用梯度检查点
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self.layer, x)

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 2. 速度优化

```python
# 使用DataLoader并行加载
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# 使用torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 使用Hugging Face PEFT库
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
```

### 扩展阅读

#### 进阶主题

1. **高级PEFT方法**
   - QLoRA：量化LoRA
   - (IA)³：可插放适配器
   - BitFit：仅微调偏置项
   - LoRA+：改进的LoRA

2. **持续学习**
   - Progressive Neural Networks
   - PackNet
   - Memory Aware Synapses
   - Gradient Episodic Memory

3. **领域适应**
   - Domain-Adversarial Training
   - Domain-Invariant Representation Learning
   - Meta-Learning for Domain Adaptation

#### 推荐资源

**视频课程**：
- Stanford CS231N: CNNs for Visual Recognition
- Hugging Face PEFT Course
- DeepLearning.AI: Fine-tuning LLMs

**代码库**：
- Hugging Face PEFT
- LoRA (Microsoft)
- AdapterHub (Humboldt University)

**论文列表**：
- LoRA: Low-Rank Adaptation of Large Language Models (2021)
- Parameter-Efficient Transfer Learning for NLP (2019)
- The Power of Scale for Parameter-Efficient Prompt Tuning (2021)
- Don't Forget the Long-Tail: Adversarial Training with Long-Tail (2020)

### 评估标准

完成本模块后，你应该能够：

- [ ] 解释迁移学习和微调的原理
- [ ] 实现完整的微调流程
- [ ] 理解PEFT方法的数学原理
- [ ] 实现LoRA、Adapter、Prefix-Tuning
- [ ] 理解领域偏移问题
- [ ] 实现DAPT和TAPT
- [ ] 理解灾难性遗忘问题
- [ ] 实现EWC和Experience Replay
- [ ] 选择合适的微调方法
- [ ] 优化微调性能
- [ ] 阅读和理解微调相关论文

### 下一步

完成 Module 5 后，建议：

1. **巩固基础**：重新实现一遍核心组件
2. **项目实践**：完成至少一个完整项目
3. **阅读论文**：深入理解PEFT和领域适应
4. **学习 Module 6**：高级训练技术

---

## 📝 学习检查清单

### Week 1: 迁移学习
- [ ] 理解迁移学习原理
- [ ] 实现预训练-微调范式
- [ ] 学习学习率调度策略
- [ ] 实现文本分类微调

### Week 2: PEFT
- [ ] 理解PEFT原理
- [ ] 实现LoRA
- [ ] 实现Adapter
- [ ] 实现Prefix-Tuning

### Week 3: 领域适应
- [ ] 理解领域偏移
- [ ] 实现DAPT
- [ ] 实现EWC
- [ ] 实现Experience Replay

### Week 4: 实战
- [ ] 选择合适的微调方法
- [ ] 实现完整的微调流程
- [ ] 评估和优化性能
- [ ] 完成项目

---

**祝学习顺利！** 🚀

如有问题，请参考：
- 📖 Notebook 中的 FAQ 部分
- 💬 课程讨论区
- 🔍 Stack Overflow
- 📧 联系助教