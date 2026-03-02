# Module 2 - 01 RNN and LSTM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a comprehensive notebook explaining RNN and LSTM architectures, demonstrating why sequence models are needed, implementing both from scratch, and showing the gradient vanishing problem.

**Architecture:** Follow CODEBASE.md structure, build from simple RNN to LSTM, use NumPy for educational implementation then PyTorch for practical use, demonstrate on sequence tasks.

**Tech Stack:** Jupyter Notebook, NumPy, PyTorch, Matplotlib

---

## Task 1: Create Notebook and Sequence Modeling Motivation

**Files:**
- Create: `notebooks/Module02_Evolution/01_rnn_lstm.ipynb`

**Step 1: Create notebook with overview**

```markdown
# Module 2.1: RNN与LSTM

## 1. 本章概览

### 📚 学习目标

1. **序列模型的必要性**：理解为什么需要处理序列数据
2. **RNN原理**：掌握循环神经网络的基本结构
3. **LSTM架构**：理解长短期记忆网络如何解决梯度消失
4. **实现与应用**：从零实现RNN和LSTM

### 🎯 核心问题

- 为什么前馈网络不适合序列数据？
- RNN如何处理变长序列？
- 梯度消失问题是什么？LSTM如何解决？

### 🗺️ 知识地图

```
前馈网络的局限
    ↓
循环神经网络(RNN)
    ↓
梯度消失问题
    ↓
LSTM/GRU
    ↓
Attention机制(下一章)
```

### ⏱️ 预计学习时间：3-4小时
```

**Step 2: Add imports**

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Libraries imported")
print(f"PyTorch version: {torch.__version__}")
```

**Step 3: Add motivation section**

```markdown
## 2. 动机与背景

### 为什么需要序列模型？

**序列数据无处不在**：
- 文本：单词序列
- 语音：音频帧序列
- 视频：图像帧序列
- 时间序列：股票价格、天气数据

**前馈网络的局限**：
1. **固定输入长度**：无法处理变长序列
2. **无记忆能力**：每个输入独立处理
3. **无法捕捉时序关系**：不考虑顺序信息

### 一个简单的例子

**任务**：预测句子的下一个单词

- 输入："The cat sat on the ___"
- 期望输出："mat" 或 "floor"

前馈网络需要固定长度输入，无法处理这种任务。
```

**Step 4: Demonstrate sequence problem**

```python
# 🔬 Micro Practice: Why Feedforward Networks Fail on Sequences
# Goal: Show the limitation of feedforward networks
# Expected outcome: Understand the need for sequence models

# Simple sequence prediction task: echo the input with delay
def generate_echo_data(seq_length=10, n_samples=100):
    """
    Generate data where output is input shifted by 1 step
    Input:  [1, 2, 3, 4, 5]
    Output: [0, 1, 2, 3, 4]
    """
    X = np.random.randint(0, 10, (n_samples, seq_length))
    y = np.zeros_like(X)
    y[:, 1:] = X[:, :-1]  # Shift by 1
    return X, y

X, y = generate_echo_data(seq_length=5, n_samples=3)

print("Echo Task Example:")
print("="*50)
for i in range(3):
    print(f"Input:  {X[i]}")
    print(f"Output: {y[i]}")
    print()

print("问题：前馈网络如何知道当前输出依赖于之前的输入？")
print("答案：需要循环结构来维护'记忆'！")
```

**Step 5: Commit**

```bash
git add notebooks/Module02_Evolution/01_rnn_lstm.ipynb
git commit -m "feat(module02): create RNN/LSTM notebook with motivation"
```

---

## Task 2: RNN Theory and Implementation

**Files:**
- Modify: `notebooks/Module02_Evolution/01_rnn_lstm.ipynb`

**Step 1: Add RNN theory**

```markdown
## 3. 理论基础

### 3.1 循环神经网络 (RNN)

**核心思想**：在处理序列时维护一个"隐藏状态"，捕捉历史信息。

**数学表示**：

对于时间步 $t$：
$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
$$y_t = W_{hy}h_t + b_y$$

其中：
- $x_t$: 时间步 $t$ 的输入
- $h_t$: 时间步 $t$ 的隐藏状态
- $y_t$: 时间步 $t$ 的输出
- $W_{hh}$: 隐藏状态到隐藏状态的权重
- $W_{xh}$: 输入到隐藏状态的权重
- $W_{hy}$: 隐藏状态到输出的权重

**关键特点**：
- 参数共享：所有时间步使用相同的权重
- 循环连接：$h_t$ 依赖于 $h_{t-1}$
- 可处理任意长度序列

**展开视图**：

```
x₁ → [RNN] → h₁ → y₁
      ↓
x₂ → [RNN] → h₂ → y₂
      ↓
x₃ → [RNN] → h₃ → y₃
```
```

**Step 2: Implement simple RNN from scratch**

```python
# 🔬 Micro Practice: Implement Simple RNN
# Goal: Build RNN from scratch using NumPy
# Expected outcome: Understand RNN computation

class SimpleRNN:
    """
    Simple RNN implementation from scratch
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize RNN parameters

        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
        """
        # Initialize weights with small random values
        self.hidden_size = hidden_size

        # Input to hidden
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01

        # Hidden to hidden (recurrent connection)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01

        # Hidden to output
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01

        # Biases
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))

    def forward(self, X):
        """
        Forward pass through sequence

        Args:
            X: Input sequence (seq_length, input_size)

        Returns:
            outputs: Output at each time step
            hidden_states: Hidden states at each time step
        """
        seq_length = X.shape[0]

        # Initialize hidden state
        h = np.zeros((1, self.hidden_size))

        # Store outputs and hidden states
        outputs = []
        hidden_states = [h]

        # Process sequence
        for t in range(seq_length):
            # Get input at time t
            x_t = X[t:t+1]  # Shape: (1, input_size)

            # Update hidden state
            h = np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)

            # Compute output
            y_t = h @ self.W_hy + self.b_y

            outputs.append(y_t)
            hidden_states.append(h)

        return np.array(outputs).squeeze(), hidden_states

# Test RNN
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)

# Create sample sequence
seq_length = 5
X = np.random.randn(seq_length, 10)

# Forward pass
outputs, hidden_states = rnn.forward(X)

print("RNN Forward Pass:")
print(f"Input shape: {X.shape}")
print(f"Output shape: {outputs.shape}")
print(f"Number of hidden states: {len(hidden_states)}")
print(f"Hidden state shape: {hidden_states[0].shape}")
print("\n✓ RNN processes sequence step by step!")
```

**Step 3: Visualize RNN computation**

```python
# 🔬 Micro Practice: Visualize RNN Unrolling
# Goal: See how RNN processes sequences
# Expected outcome: Understand temporal dynamics

# Simple sequence: sine wave
t = np.linspace(0, 4*np.pi, 50)
sequence = np.sin(t).reshape(-1, 1)

# Process with RNN
rnn_viz = SimpleRNN(input_size=1, hidden_size=5, output_size=1)
outputs, hidden_states = rnn_viz.forward(sequence)

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Input sequence
axes[0].plot(sequence, 'b-', linewidth=2, label='Input')
axes[0].set_title('Input Sequence (Sine Wave)', fontsize=14, weight='bold')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Hidden states evolution
hidden_array = np.array([h.flatten() for h in hidden_states[1:]])
for i in range(min(5, hidden_array.shape[1])):
    axes[1].plot(hidden_array[:, i], label=f'h_{i}', alpha=0.7)
axes[1].set_title('Hidden States Evolution', fontsize=14, weight='bold')
axes[1].set_ylabel('Hidden State Value')
axes[1].legend(loc='right')
axes[1].grid(True, alpha=0.3)

# Output
axes[2].plot(outputs, 'r-', linewidth=2, label='RNN Output')
axes[2].set_title('RNN Output', fontsize=14, weight='bold')
axes[2].set_xlabel('Time Step')
axes[2].set_ylabel('Value')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ RNN maintains hidden state across time steps!")
```

**Step 4: Commit**

```bash
git add notebooks/Module02_Evolution/01_rnn_lstm.ipynb
git commit -m "feat(module02): add RNN theory and NumPy implementation"
```

---

## Task 3: Gradient Vanishing Problem

**Files:**
- Modify: `notebooks/Module02_Evolution/01_rnn_lstm.ipynb`

**Step 1: Add gradient vanishing theory**

```markdown
### 3.2 梯度消失问题 (Vanishing Gradient Problem)

**问题**：在长序列中，早期时间步的梯度会指数级衰减。

**数学分析**：

反向传播时，梯度需要通过多个时间步传播：

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^T \frac{\partial h_t}{\partial h_{t-1}}$$

由于 $\frac{\partial h_t}{\partial h_{t-1}} = W_{hh} \cdot \text{diag}(\tanh'(z_t))$，如果：
- $|W_{hh}| < 1$ 且 $|\tanh'| < 1$：梯度消失
- $|W_{hh}| > 1$：梯度爆炸

**后果**：
- 无法学习长期依赖
- 训练困难
- 性能受限

**解决方案**：
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- 梯度裁剪
```

**Step 2: Demonstrate gradient vanishing**

```python
# 🔬 Micro Practice: Demonstrate Gradient Vanishing
# Goal: Show how gradients decay over time
# Expected outcome: Understand the problem

def compute_gradient_flow(seq_length, W_value):
    """
    Simulate gradient flow through RNN

    Args:
        seq_length: Length of sequence
        W_value: Value of recurrent weight

    Returns:
        Gradient magnitudes at each time step
    """
    # Assume tanh derivative ≈ 0.5 (average)
    tanh_derivative = 0.5

    # Gradient at final time step
    grad = 1.0
    gradients = [grad]

    # Backpropagate through time
    for t in range(seq_length - 1):
        grad = grad * W_value * tanh_derivative
        gradients.append(grad)

    return list(reversed(gradients))

# Test different scenarios
seq_length = 50
scenarios = [
    (0.5, "W < 1 (Vanishing)"),
    (1.0, "W = 1 (Stable)"),
    (1.5, "W > 1 (Exploding)")
]

plt.figure(figsize=(14, 5))

for i, (W, label) in enumerate(scenarios):
    grads = compute_gradient_flow(seq_length, W)

    plt.subplot(1, 3, i+1)
    plt.plot(grads, linewidth=2)
    plt.title(label, fontsize=12, weight='bold')
    plt.xlabel('Time Step (backward)')
    plt.ylabel('Gradient Magnitude')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Initial')
    plt.legend()

plt.tight_layout()
plt.show()

print("观察：")
print("- W < 1: 梯度快速衰减到0（梯度消失）")
print("- W = 1: 梯度保持稳定")
print("- W > 1: 梯度指数增长（梯度爆炸）")
print("\n这就是为什么RNN难以学习长期依赖！")
```

**Step 3: Commit**

```bash
git add notebooks/Module02_Evolution/01_rnn_lstm.ipynb
git commit -m "feat(module02): demonstrate gradient vanishing problem"
```

---

## Task 4: LSTM Architecture

**Files:**
- Modify: `notebooks/Module02_Evolution/01_rnn_lstm.ipynb`

**Step 1: Add LSTM theory**

```markdown
### 3.3 长短期记忆网络 (LSTM)

**核心思想**：使用门控机制控制信息流，解决梯度消失问题。

**LSTM单元结构**：

1. **遗忘门 (Forget Gate)**：决定丢弃哪些信息
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **输入门 (Input Gate)**：决定存储哪些新信息
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

3. **细胞状态更新 (Cell State Update)**：
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

4. **输出门 (Output Gate)**：决定输出哪些信息
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

**关键优势**：
- 细胞状态 $C_t$ 提供"高速公路"，梯度可以直接流动
- 门控机制学习何时记忆、遗忘、输出
- 有效缓解梯度消失问题
```

**Step 2: Implement LSTM from scratch**

```python
# 🔬 Micro Practice: Implement LSTM
# Goal: Build LSTM from scratch
# Expected outcome: Understand gate mechanisms

class SimpleLSTM:
    """
    LSTM implementation from scratch
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize LSTM parameters
        """
        self.hidden_size = hidden_size

        # Concatenated input size
        concat_size = input_size + hidden_size

        # Forget gate
        self.W_f = np.random.randn(concat_size, hidden_size) * 0.01
        self.b_f = np.zeros((1, hidden_size))

        # Input gate
        self.W_i = np.random.randn(concat_size, hidden_size) * 0.01
        self.b_i = np.zeros((1, hidden_size))

        # Cell gate
        self.W_c = np.random.randn(concat_size, hidden_size) * 0.01
        self.b_c = np.zeros((1, hidden_size))

        # Output gate
        self.W_o = np.random.randn(concat_size, hidden_size) * 0.01
        self.b_o = np.zeros((1, hidden_size))

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        """
        Forward pass through LSTM

        Args:
            X: Input sequence (seq_length, input_size)

        Returns:
            outputs: Hidden states at each time step
            gates_history: Gate activations for visualization
        """
        seq_length = X.shape[0]

        # Initialize states
        h = np.zeros((1, self.hidden_size))
        C = np.zeros((1, self.hidden_size))

        outputs = []
        gates_history = {'forget': [], 'input': [], 'output': []}

        for t in range(seq_length):
            x_t = X[t:t+1]

            # Concatenate input and previous hidden state
            concat = np.concatenate([h, x_t], axis=1)

            # Forget gate
            f_t = self.sigmoid(concat @ self.W_f + self.b_f)

            # Input gate
            i_t = self.sigmoid(concat @ self.W_i + self.b_i)
            C_tilde = np.tanh(concat @ self.W_c + self.b_c)

            # Update cell state
            C = f_t * C + i_t * C_tilde

            # Output gate
            o_t = self.sigmoid(concat @ self.W_o + self.b_o)
            h = o_t * np.tanh(C)

            outputs.append(h)
            gates_history['forget'].append(f_t.mean())
            gates_history['input'].append(i_t.mean())
            gates_history['output'].append(o_t.mean())

        return np.array(outputs).squeeze(), gates_history

# Test LSTM
lstm = SimpleLSTM(input_size=10, hidden_size=20)
X = np.random.randn(30, 10)
outputs, gates = lstm.forward(X)

print("LSTM Forward Pass:")
print(f"Input shape: {X.shape}")
print(f"Output shape: {outputs.shape}")
print("\n✓ LSTM successfully processes sequence with gates!")
```

**Step 3: Visualize LSTM gates**

```python
# 🔬 Micro Practice: Visualize LSTM Gates
# Goal: See how gates control information flow
# Expected outcome: Understand gate behavior

# Process sequence with LSTM
t = np.linspace(0, 4*np.pi, 100)
sequence = np.sin(t).reshape(-1, 1)

lstm_viz = SimpleLSTM(input_size=1, hidden_size=10)
outputs, gates = lstm_viz.forward(sequence)

# Visualize gates
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Input
axes[0].plot(sequence, 'b-', linewidth=2)
axes[0].set_title('Input Sequence', fontsize=14, weight='bold')
axes[0].set_ylabel('Value')
axes[0].grid(True, alpha=0.3)

# Gates
gate_names = ['Forget Gate', 'Input Gate', 'Output Gate']
gate_keys = ['forget', 'input', 'output']
colors = ['red', 'green', 'blue']

for i, (name, key, color) in enumerate(zip(gate_names, gate_keys, colors)):
    axes[i+1].plot(gates[key], color=color, linewidth=2, label=name)
    axes[i+1].set_title(name, fontsize=14, weight='bold')
    axes[i+1].set_ylabel('Gate Value')
    axes[i+1].set_ylim(-0.1, 1.1)
    axes[i+1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[i+1].legend()
    axes[i+1].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time Step')

plt.tight_layout()
plt.show()

print("观察：")
print("- 遗忘门：控制保留多少历史信息")
print("- 输入门：控制接受多少新信息")
print("- 输出门：控制输出多少信息")
print("\n✓ 门控机制让LSTM能够学习长期依赖！")
```

**Step 4: Commit**

```bash
git add notebooks/Module02_Evolution/01_rnn_lstm.ipynb
git commit -m "feat(module02): add LSTM architecture and implementation"
```

---

## Task 5: PyTorch Implementation and Summary

**Files:**
- Modify: `notebooks/Module02_Evolution/01_rnn_lstm.ipynb`

**Step 1: Add PyTorch RNN/LSTM**

```markdown
### 3.4 PyTorch实现

PyTorch提供了高效的RNN和LSTM实现。
```

```python
# 🔬 Micro Practice: PyTorch RNN and LSTM
# Goal: Use PyTorch's built-in implementations
# Expected outcome: Compare with our implementations

# Create sample data
batch_size = 2
seq_length = 10
input_size = 5
hidden_size = 20

X_torch = torch.randn(seq_length, batch_size, input_size)

print("=== PyTorch RNN ===\n")

# RNN
rnn_torch = nn.RNN(input_size, hidden_size, batch_first=False)
output_rnn, h_n = rnn_torch(X_torch)

print(f"Input shape: {X_torch.shape}")
print(f"Output shape: {output_rnn.shape}")
print(f"Final hidden state shape: {h_n.shape}\n")

print("=== PyTorch LSTM ===\n")

# LSTM
lstm_torch = nn.LSTM(input_size, hidden_size, batch_first=False)
output_lstm, (h_n, c_n) = lstm_torch(X_torch)

print(f"Input shape: {X_torch.shape}")
print(f"Output shape: {output_lstm.shape}")
print(f"Final hidden state shape: {h_n.shape}")
print(f"Final cell state shape: {c_n.shape}\n")

print("✓ PyTorch makes it easy to use RNN/LSTM!")
```

**Step 2: Add remaining sections**

```markdown
## 4-6. 实践总结

本章实现了：
- ✅ 从零实现RNN
- ✅ 理解梯度消失问题
- ✅ 从零实现LSTM
- ✅ 使用PyTorch的RNN/LSTM

## 7. 常见问题与调试

### Q1: RNN和LSTM的主要区别是什么？

**A:**
- RNN：简单循环结构，容易梯度消失
- LSTM：门控机制，细胞状态提供梯度高速公路

### Q2: 什么时候使用RNN，什么时候使用LSTM？

**A:**
- 短序列（<10步）：RNN可能足够
- 长序列：使用LSTM或GRU
- 实践中：LSTM是更安全的选择

### Q3: GRU和LSTM有什么区别？

**A:**
- GRU：更简单，只有2个门（更新门、重置门）
- LSTM：3个门，更强大但参数更多
- 性能：通常相近，GRU训练更快

## 8. 总结与展望

### 核心要点

1. **RNN**：通过循环连接处理序列
2. **梯度消失**：限制了RNN学习长期依赖
3. **LSTM**：门控机制有效解决梯度消失
4. **应用**：序列建模的基础

### 下一章预告

**Module 2.2: Attention机制**
- Attention的动机
- Attention的数学原理
- 实现和可视化

### 💡 思考题

1. 为什么LSTM的细胞状态能缓解梯度消失？
2. 如何选择隐藏层大小？
3. 双向RNN有什么优势？
```

**Step 3: Final commit**

```bash
git add notebooks/Module02_Evolution/01_rnn_lstm.ipynb
git commit -m "feat(module02): complete RNN/LSTM notebook"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module02-rnn-lstm.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.
