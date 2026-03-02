# Module 1 - 02 Neural Networks Basics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a comprehensive notebook introducing neural networks from perceptrons to multi-layer networks with activation functions and complete NumPy implementation.

**Architecture:** Follow the 8-section CODEBASE.md structure with theory in Chinese, code in English, building from single perceptron to multi-layer networks with hands-on implementations.

**Tech Stack:** Jupyter Notebook, NumPy, Matplotlib

---

## Task 1: Create Notebook Structure and Overview

**Files:**
- Create: `notebooks/Module01_Foundation/02_neural_networks_basics.ipynb`

**Step 1: Create notebook with title and overview**

```markdown
# Module 1.2: 神经网络基础

## 1. 本章概览 (Overview)

### 📚 学习目标

1. **感知机**：理解最简单的神经网络单元
2. **多层网络**：理解如何组合神经元构建深度网络
3. **激活函数**：理解非线性的重要性
4. **前向传播**：实现完整的前向传播过程

### 🎯 核心问题

- 神经网络如何学习？
- 为什么需要多层？
- 激活函数的作用是什么？

### ⏱️ 预计学习时间：2-3小时
```

**Step 2: Add imports**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 6)
print("✓ Libraries imported")
```

**Step 3: Commit**

```bash
git add notebooks/Module01_Foundation/02_neural_networks_basics.ipynb
git commit -m "feat(module01): create neural networks basics notebook structure"
```

---

## Task 2: Perceptron Theory and Implementation

**Files:**
- Modify: `notebooks/Module01_Foundation/02_neural_networks_basics.ipynb`

**Step 1: Add perceptron theory**

```markdown
## 2. 动机与背景 (Motivation)

### 从生物神经元到人工神经元

生物神经元接收多个输入信号，处理后输出信号。人工神经元（感知机）模拟这个过程。

## 3. 理论基础 (Theory)

### 3.1 感知机 (Perceptron)

**数学模型**：

$$y = f\left(\sum_{i=1}^n w_i x_i + b\right)$$

其中：
- $x_i$: 输入特征
- $w_i$: 权重
- $b$: 偏置
- $f$: 激活函数（阶跃函数）

**几何意义**：感知机在特征空间中画一条直线（或超平面）分隔数据。
```

**Step 2: Implement perceptron**

```python
# 🔬 Micro Practice: Implement Perceptron
# Goal: Build a simple perceptron from scratch
# Expected outcome: Binary classification on linearly separable data

class Perceptron:
    """
    Simple perceptron for binary classification
    """

    def __init__(self, n_features):
        """
        Initialize perceptron

        Args:
            n_features: Number of input features
        """
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predictions (0 or 1)
        """
        linear_output = X @ self.weights + self.bias
        return (linear_output >= 0).astype(int)

    def train(self, X, y, learning_rate=0.1, epochs=100):
        """
        Train perceptron using perceptron learning rule

        Args:
            X: Training features
            y: Training labels (0 or 1)
            learning_rate: Learning rate
            epochs: Number of training epochs
        """
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                prediction = self.predict(xi.reshape(1, -1))[0]
                error = yi - prediction
                self.weights += learning_rate * error * xi
                self.bias += learning_rate * error

# Test on simple dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate

perceptron = Perceptron(n_features=2)
perceptron.train(X, y, epochs=10)

print("Perceptron weights:", perceptron.weights)
print("Perceptron bias:", perceptron.bias)
print("\nPredictions:")
for xi, yi in zip(X, y):
    pred = perceptron.predict(xi.reshape(1, -1))[0]
    print(f"Input: {xi}, True: {yi}, Pred: {pred}")
```

**Step 3: Visualize decision boundary**

```python
# 🔬 Micro Practice: Visualize Decision Boundary
# Goal: See how perceptron separates data

def plot_decision_boundary(model, X, y):
    """Plot decision boundary of perceptron"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue']),
                edgecolors='black', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.show()

plot_decision_boundary(perceptron, X, y)
print("✓ 感知机成功学习了AND逻辑门！")
```

**Step 4: Commit**

```bash
git add notebooks/Module01_Foundation/02_neural_networks_basics.ipynb
git commit -m "feat(module01): add perceptron theory and implementation"
```

---

## Task 3: Activation Functions

**Files:**
- Modify: `notebooks/Module01_Foundation/02_neural_networks_basics.ipynb`

**Step 1: Add activation functions theory**

```markdown
### 3.2 激活函数 (Activation Functions)

**为什么需要激活函数？**

如果没有激活函数，多层神经网络等价于单层：
$$f(f(x)) = W_2(W_1x) = (W_2W_1)x = Wx$$

激活函数引入**非线性**，使网络能够学习复杂模式。

**常用激活函数**：

1. **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
   - 输出范围：(0, 1)
   - 用途：二分类输出层

2. **Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
   - 输出范围：(-1, 1)
   - 比Sigmoid更好（零中心）

3. **ReLU**: $\text{ReLU}(x) = \max(0, x)$
   - 最常用
   - 计算简单，缓解梯度消失

4. **Leaky ReLU**: $\text{LeakyReLU}(x) = \max(0.01x, x)$
   - 解决ReLU的"死亡"问题
```

**Step 2: Implement and visualize activation functions**

```python
# 🔬 Micro Practice: Activation Functions
# Goal: Understand different activation functions
# Expected outcome: Visualize and compare activations

def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh activation"""
    return np.tanh(x)

def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation"""
    return np.where(x > 0, x, alpha * x)

# Visualize
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(14, 10))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.title('Sigmoid: σ(x) = 1/(1+e^(-x))')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), 'g-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.title('Tanh: tanh(x)')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x, relu(x), 'r-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.title('ReLU: max(0, x)')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x), 'm-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.title('Leaky ReLU: max(0.01x, x)')
plt.xlabel('x')
plt.ylabel('Leaky ReLU(x)')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

print("✓ 激活函数对比完成")
```

**Step 3: Commit**

```bash
git add notebooks/Module01_Foundation/02_neural_networks_basics.ipynb
git commit -m "feat(module01): add activation functions theory and visualization"
```

---

## Task 4: Multi-Layer Neural Network

**Files:**
- Modify: `notebooks/Module01_Foundation/02_neural_networks_basics.ipynb`

**Step 1: Add multi-layer network theory**

```markdown
### 3.3 多层神经网络 (Multi-Layer Neural Network)

**架构**：

```
Input Layer → Hidden Layer(s) → Output Layer
```

**前向传播**：

$$\mathbf{h} = f(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{y} = g(\mathbf{W}_2\mathbf{h} + \mathbf{b}_2)$$

其中 $f$ 和 $g$ 是激活函数。

**为什么需要多层？**

- 单层只能学习线性可分的模式
- 多层可以学习复杂的非线性模式
- 深度网络可以学习层次化的特征表示
```

**Step 2: Implement multi-layer network**

```python
# 🔬 Micro Practice: Multi-Layer Neural Network
# Goal: Build a 2-layer neural network from scratch
# Expected outcome: Solve XOR problem (not linearly separable)

class NeuralNetwork:
    """
    Simple 2-layer neural network
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize network

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units
        """
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        """
        Forward propagation

        Args:
            X: Input features (n_samples, input_size)

        Returns:
            output: Network output
            cache: Intermediate values for backprop
        """
        # Hidden layer
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)  # Activation

        # Output layer
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)  # Activation

        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2, cache

    def predict(self, X):
        """Make predictions"""
        output, _ = self.forward(X)
        return (output > 0.5).astype(int)

# Test on XOR problem (not linearly separable!)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])  # XOR gate

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

print("XOR Problem (before training):")
output, _ = nn.forward(X_xor)
print("Predictions:", output.flatten())
print("True labels:", y_xor.flatten())
print("\n注意：训练前的预测是随机的")
print("下一章我们将学习如何训练这个网络（反向传播）")
```

**Step 3: Visualize network architecture**

```python
# 🔬 Micro Practice: Visualize Network Architecture
# Goal: Understand network structure

def visualize_network():
    """Visualize neural network architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Layer positions
    layer_sizes = [2, 4, 1]
    layer_names = ['Input\nLayer', 'Hidden\nLayer', 'Output\nLayer']
    v_spacing = 1.0 / max(layer_sizes)
    h_spacing = 1.0 / (len(layer_sizes) - 1)

    # Draw nodes
    for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        layer_top = v_spacing * (size - 1) / 2.0 + 0.5
        for j in range(size):
            circle = plt.Circle((i * h_spacing, layer_top - j * v_spacing),
                               v_spacing/4.0, color='lightblue', ec='black', zorder=4)
            ax.add_artist(circle)

            # Add labels
            if i == 0:
                ax.text(i * h_spacing - 0.15, layer_top - j * v_spacing,
                       f'$x_{j+1}$', fontsize=12, ha='right', va='center')
            elif i == len(layer_sizes) - 1:
                ax.text(i * h_spacing + 0.15, layer_top - j * v_spacing,
                       f'$y$', fontsize=12, ha='left', va='center')

        # Layer names
        ax.text(i * h_spacing, -0.2, name, fontsize=14, ha='center', weight='bold')

    # Draw connections
    for i in range(len(layer_sizes) - 1):
        layer_top_from = v_spacing * (layer_sizes[i] - 1) / 2.0 + 0.5
        layer_top_to = v_spacing * (layer_sizes[i+1] - 1) / 2.0 + 0.5
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                ax.plot([i * h_spacing, (i+1) * h_spacing],
                       [layer_top_from - j * v_spacing, layer_top_to - k * v_spacing],
                       'gray', alpha=0.3, linewidth=0.5)

    ax.axis('off')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.1)
    plt.title('Neural Network Architecture (2-4-1)', fontsize=16, weight='bold', pad=20)
    plt.show()

visualize_network()
print("✓ 网络结构：2个输入 → 4个隐藏神经元 → 1个输出")
```

**Step 4: Commit**

```bash
git add notebooks/Module01_Foundation/02_neural_networks_basics.ipynb
git commit -m "feat(module01): add multi-layer network implementation"
```

---

## Task 5: Summary and Remaining Sections

**Files:**
- Modify: `notebooks/Module01_Foundation/02_neural_networks_basics.ipynb`

**Step 1: Add simplified sections 4-6**

```markdown
## 4-6. 实践总结

本章我们实现了：
- ✅ 感知机（单层神经网络）
- ✅ 激活函数（引入非线性）
- ✅ 多层神经网络（前向传播）

**下一章预告**：我们将学习如何训练这些网络（反向传播算法）。
```

**Step 2: Add FAQ**

```markdown
## 7. 常见问题与调试

### Q1: 为什么感知机不能解决XOR问题？

**A:** XOR不是线性可分的。感知机只能画一条直线分隔数据，而XOR需要两条线。

### Q2: 隐藏层应该有多少个神经元？

**A:** 没有固定规则，通常：
- 从小开始（如输入维度的2倍）
- 根据任务复杂度调整
- 过多会过拟合，过少会欠拟合

### Q3: 应该选择哪个激活函数？

**A:**
- 隐藏层：ReLU（默认选择）
- 输出层：根据任务
  - 二分类：Sigmoid
  - 多分类：Softmax
  - 回归：线性（无激活）
```

**Step 3: Add summary**

```markdown
## 8. 总结与展望

### 核心要点

1. **感知机**：最简单的神经网络，但只能处理线性可分问题
2. **激活函数**：引入非线性，使网络能学习复杂模式
3. **多层网络**：组合多个层，学习层次化特征

### 与其他技术的联系

```
感知机 → 多层网络 → 深度网络
   ↓         ↓          ↓
线性分类  非线性分类  复杂模式识别
```

### 下一章预告

**Module 1.3: 反向传播**
- 如何计算梯度？
- 如何更新权重？
- 完整的训练流程

### 💡 思考题

1. 为什么需要偏置项 $b$？
2. 如果所有激活函数都是线性的会怎样？
3. 更深的网络一定更好吗？
4. 如何初始化权重？
```

**Step 4: Final commit**

```bash
git add notebooks/Module01_Foundation/02_neural_networks_basics.ipynb
git commit -m "feat(module01): complete neural networks basics notebook"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module01-neural-networks.md`.

This plan is ready for **parallel execution** in a separate session using `superpowers:executing-plans`.
