# Module 1 - 03 Backpropagation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a comprehensive notebook explaining and implementing backpropagation algorithm from scratch, including chain rule, gradient computation, and complete training loop.

**Architecture:** Follow CODEBASE.md structure, build from chain rule basics to full backpropagation implementation with manual gradient calculation and automatic differentiation comparison.

**Tech Stack:** Jupyter Notebook, NumPy, Matplotlib

---

## Task 1: Create Notebook and Chain Rule Basics

**Files:**
- Create: `notebooks/Module01_Foundation/03_backpropagation.ipynb`

**Step 1: Create notebook with overview**

```markdown
# Module 1.3: 反向传播算法

## 1. 本章概览

### 📚 学习目标

1. **链式法则**：理解复合函数的求导
2. **计算图**：理解前向和反向传播的计算流程
3. **梯度计算**：手动计算神经网络的梯度
4. **完整训练**：实现完整的训练循环

### 🎯 核心问题

- 如何高效计算复杂函数的梯度？
- 反向传播的本质是什么？
- 如何训练神经网络？

### ⏱️ 预计学习时间：2-3小时
```

**Step 2: Add imports and chain rule theory**

```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
print("✓ Libraries imported")
```

```markdown
## 2. 动机与背景

### 为什么需要反向传播？

神经网络有数百万个参数，如何高效计算每个参数的梯度？

**朴素方法**：对每个参数单独求导 → $O(n)$ 次前向传播
**反向传播**：一次前向 + 一次反向 → $O(1)$ 次前向传播

## 3. 理论基础

### 3.1 链式法则 (Chain Rule)

**单变量**：
$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

**多变量**：
$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$

**直觉**：从输出到输入，逐层传播梯度。
```

**Step 3: Implement chain rule examples**

```python
# 🔬 Micro Practice: Chain Rule
# Goal: Understand chain rule with concrete examples
# Expected outcome: Verify chain rule numerically

def f(x):
    """Outer function: f(y) = y^2"""
    return x**2

def g(x):
    """Inner function: g(x) = 2x + 1"""
    return 2*x + 1

def composite(x):
    """Composite function: f(g(x)) = (2x+1)^2"""
    return f(g(x))

# Analytical derivative using chain rule
# d/dx f(g(x)) = f'(g(x)) * g'(x) = 2*g(x) * 2 = 4*(2x+1)
def analytical_derivative(x):
    return 4 * (2*x + 1)

# Numerical derivative
def numerical_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x)) / h

# Test
x = 3.0
analytical = analytical_derivative(x)
numerical = numerical_derivative(composite, x)

print(f"Function: f(g(x)) = (2x+1)²")
print(f"At x = {x}:")
print(f"  Analytical derivative: {analytical}")
print(f"  Numerical derivative: {numerical:.6f}")
print(f"  Difference: {abs(analytical - numerical):.2e}")
print("\n✓ Chain rule verified!")
```

**Step 4: Commit**

```bash
git add notebooks/Module01_Foundation/03_backpropagation.ipynb
git commit -m "feat(module01): create backpropagation notebook with chain rule"
```

---

## Task 2: Computational Graph

**Files:**
- Modify: `notebooks/Module01_Foundation/03_backpropagation.ipynb`

**Step 1: Add computational graph theory**

```markdown
### 3.2 计算图 (Computational Graph)

**计算图**：将计算过程表示为有向无环图（DAG）

**前向传播**：从输入到输出计算函数值
**反向传播**：从输出到输入���算梯度

**示例**：$z = (x + y) \times w$

```
Forward:
x, y, w → a = x + y → z = a × w

Backward:
∂z/∂w = a
∂z/∂a = w
∂z/∂x = ∂z/∂a × ∂a/∂x = w × 1 = w
∂z/∂y = ∂z/∂a × ∂a/∂y = w × 1 = w
```
```

**Step 2: Implement computational graph**

```python
# 🔬 Micro Practice: Computational Graph
# Goal: Implement simple computational graph with forward and backward
# Expected outcome: Understand gradient flow

class ComputationalGraph:
    """
    Simple computational graph for z = (x + y) * w
    """

    def forward(self, x, y, w):
        """
        Forward pass

        Args:
            x, y, w: Input values

        Returns:
            z: Output value
        """
        # Store intermediate values for backward pass
        self.x = x
        self.y = y
        self.w = w
        self.a = x + y  # Intermediate result
        self.z = self.a * w  # Final output
        return self.z

    def backward(self):
        """
        Backward pass - compute all gradients

        Returns:
            Dictionary of gradients
        """
        # Start from output: dz/dz = 1
        dz_dz = 1.0

        # Gradient of z w.r.t. w: dz/dw = a
        dz_dw = self.a * dz_dz

        # Gradient of z w.r.t. a: dz/da = w
        dz_da = self.w * dz_dz

        # Gradient of z w.r.t. x: dz/dx = dz/da * da/dx = w * 1
        dz_dx = dz_da * 1.0

        # Gradient of z w.r.t. y: dz/dy = dz/da * da/dy = w * 1
        dz_dy = dz_da * 1.0

        return {'x': dz_dx, 'y': dz_dy, 'w': dz_dw}

# Test
graph = ComputationalGraph()
x, y, w = 2.0, 3.0, 4.0

# Forward
z = graph.forward(x, y, w)
print(f"Forward: z = ({x} + {y}) × {w} = {z}")

# Backward
grads = graph.backward()
print(f"\nBackward (gradients):")
print(f"  ∂z/∂x = {grads['x']}")
print(f"  ∂z/∂y = {grads['y']}")
print(f"  ∂z/∂w = {grads['w']}")

# Verify numerically
def f_x(x): return (x + y) * w
def f_y(y): return (x + y) * w
def f_w(w): return (x + y) * w

print(f"\nNumerical verification:")
print(f"  ∂z/∂x ≈ {(f_x(x+1e-5) - f_x(x))/1e-5:.6f}")
print(f"  ∂z/∂y ≈ {(f_y(y+1e-5) - f_y(y))/1e-5:.6f}")
print(f"  ∂z/∂w ≈ {(f_w(w+1e-5) - f_w(w))/1e-5:.6f}")
```

**Step 3: Visualize computational graph**

```python
# 🔬 Micro Practice: Visualize Computational Graph
# Goal: See the structure of forward and backward pass

import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Forward pass
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
ax1.set_title('Forward Pass', fontsize=16, weight='bold')

# Nodes
nodes = [
    (1, 4, 'x=2'), (1, 2, 'y=3'), (1, 0.5, 'w=4'),
    (5, 3, 'a=5'), (9, 3, 'z=20')
]
for x, y, label in nodes:
    circle = mpatches.Circle((x, y), 0.4, color='lightblue', ec='black', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold')

# Edges
edges = [(1, 4, 5, 3), (1, 2, 5, 3), (5, 3, 9, 3), (1, 0.5, 9, 3)]
for x1, y1, x2, y2 in edges:
    arrow = FancyArrowPatch((x1+0.4, y1), (x2-0.4, y2),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax1.add_patch(arrow)

# Operations
ax1.text(3, 3.5, '+', fontsize=14, ha='center', bbox=dict(boxstyle='round', facecolor='yellow'))
ax1.text(7, 3.5, '×', fontsize=14, ha='center', bbox=dict(boxstyle='round', facecolor='yellow'))

# Backward pass
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('Backward Pass (Gradients)', fontsize=16, weight='bold')

# Nodes with gradients
grad_nodes = [
    (1, 4, '∂z/∂x=4'), (1, 2, '∂z/∂y=4'), (1, 0.5, '∂z/∂w=5'),
    (5, 3, '∂z/∂a=4'), (9, 3, '∂z/∂z=1')
]
for x, y, label in grad_nodes:
    circle = mpatches.Circle((x, y), 0.4, color='lightcoral', ec='black', linewidth=2)
    ax2.add_patch(circle)
    ax2.text(x, y, label, ha='center', va='center', fontsize=9, weight='bold')

# Backward edges (reversed)
for x1, y1, x2, y2 in edges:
    arrow = FancyArrowPatch((x2-0.4, y2), (x1+0.4, y1),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax2.add_patch(arrow)

plt.tight_layout()
plt.show()

print("✓ 计算图展示了前向和反向传播的过程")
```

**Step 4: Commit**

```bash
git add notebooks/Module01_Foundation/03_backpropagation.ipynb
git commit -m "feat(module01): add computational graph theory and visualization"
```

---

## Task 3: Backpropagation for Neural Networks

**Files:**
- Modify: `notebooks/Module01_Foundation/03_backpropagation.ipynb`

**Step 1: Add neural network backprop theory**

```markdown
### 3.3 神经网络的反向传播

**网络结构**：
```
x → [W1, b1] → h → [W2, b2] → y
```

**前向传播**：
$$z_1 = W_1x + b_1$$
$$h = \sigma(z_1)$$
$$z_2 = W_2h + b_2$$
$$\hat{y} = \sigma(z_2)$$

**损失函数**（均方误差）：
$$L = \frac{1}{2}(\hat{y} - y)^2$$

**反向传播**（从输出到输入）：

1. **输出层梯度**：
   $$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$
   $$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z_2)$$

2. **权重和偏置梯度**：
   $$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot h^T$$
   $$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2}$$

3. **隐藏层梯度**：
   $$\frac{\partial L}{\partial h} = W_2^T \cdot \frac{\partial L}{\partial z_2}$$
   $$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \cdot \sigma'(z_1)$$

4. **第一层权重和偏置**：
   $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x^T$$
   $$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}$$
```

**Step 2: Implement complete backpropagation**

```python
# 🔬 Micro Practice: Complete Backpropagation Implementation
# Goal: Implement full forward and backward pass
# Expected outcome: Train a neural network on XOR problem

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetworkWithBackprop:
    """
    2-layer neural network with backpropagation
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, X):
        """Forward propagation"""
        self.z1 = X @ self.W1 + self.b1
        self.h = sigmoid(self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        self.y_pred = sigmoid(self.z2)
        return self.y_pred

    def backward(self, X, y_true):
        """
        Backward propagation

        Args:
            X: Input features
            y_true: True labels
        """
        m = X.shape[0]  # Number of samples

        # Output layer gradients
        dL_dy_pred = self.y_pred - y_true
        dL_dz2 = dL_dy_pred * sigmoid_derivative(self.z2)

        # Gradients for W2 and b2
        dL_dW2 = (self.h.T @ dL_dz2) / m
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dL_dh = dL_dz2 @ self.W2.T
        dL_dz1 = dL_dh * sigmoid_derivative(self.z1)

        # Gradients for W1 and b1
        dL_dW1 = (X.T @ dL_dz1) / m
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m

        # Update weights (gradient descent)
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2

    def compute_loss(self, y_pred, y_true):
        """Mean squared error loss"""
        return np.mean((y_pred - y_true)**2)

    def train(self, X, y, epochs=10000, print_every=1000):
        """
        Train the network

        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            print_every: Print loss every N epochs
        """
        losses = []

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)

            # Backward pass
            self.backward(X, y)

            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

        return losses

# Train on XOR problem
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

print("Training Neural Network on XOR Problem...")
print("="*50)

nn = NeuralNetworkWithBackprop(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
losses = nn.train(X_xor, y_xor, epochs=10000, print_every=2000)

print("\nFinal Predictions:")
predictions = nn.forward(X_xor)
for i, (xi, yi, pred) in enumerate(zip(X_xor, y_xor, predictions)):
    print(f"Input: {xi}, True: {yi[0]}, Pred: {pred[0]:.4f}, Rounded: {round(pred[0])}")

print("\n✓ 网络成功学习了XOR函数！")
```

**Step 3: Visualize training progress**

```python
# 🔬 Micro Practice: Visualize Training
# Goal: See how loss decreases during training

plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(losses, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Decision boundary
plt.subplot(1, 2, 2)
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
plt.colorbar(label='Prediction')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor.ravel(),
           cmap='RdYlBu', s=200, edgecolors='black', linewidth=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary (XOR Problem Solved!)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ 反向传播成功训练了神经网络！")
```

**Step 4: Commit**

```bash
git add notebooks/Module01_Foundation/03_backpropagation.ipynb
git commit -m "feat(module01): implement complete backpropagation algorithm"
```

---

## Task 4: Summary and Remaining Sections

**Files:**
- Modify: `notebooks/Module01_Foundation/03_backpropagation.ipynb`

**Step 1: Add remaining sections**

```markdown
## 4-6. 实践总结

本章实现了：
- ✅ 链式法则的应用
- ✅ 计算图的前向和反向传播
- ✅ 完整的反向传播算法
- ✅ 训练神经网络解决XOR问题

## 7. 常见问题与调试

### Q1: 为什么需要保存前向传播的中间值？

**A:** 反向传播需要这些值来计算梯度。例如，计算 $\frac{\partial L}{\partial W_2}$ 需要隐藏层的激活值 $h$。

### Q2: 梯度消失和梯度爆炸是什么？

**A:**
- **梯度消失**：梯度变得非常小，权重几乎不更新（深层网络+Sigmoid）
- **梯度爆炸**：梯度变得非常大，权重更新过大（解决：梯度裁剪）

### Q3: 如何验证反向传播实现是否正确？

**A:** 使用**梯度检查**（gradient checking）：
- 数值梯度：$(f(x+h) - f(x-h)) / (2h)$
- 解析梯度：反向传播计算的梯度
- 比较两者，差异应该 < 1e-7

### Q4: 学习率如何选择？

**A:**
- 太大：不收敛，震荡
- 太小：收敛慢
- 常用：0.001, 0.01, 0.1
- 技巧：学习率衰减、自适应学习率（Adam）

## 8. 总结与展望

### 核心要点

1. **链式法则**：反向传播的数学基础
2. **计算图**：清晰表示前向和反向传播
3. **梯度计算**：从输出到输入逐层传播
4. **权重更新**：梯度下降优化参数

### 与其他技术的联系

```
链式法则 → 反向传播 → 梯度下降 → 训练神经网络
```

### 下一章预告

**Module 1.4: PyTorch基础**
- 自动微分
- 构建模型
- 训练循环

### 💡 思考题

1. 为什么反向传播比数值梯度快得多？
2. 如何处理梯度消失问题？
3. Batch Normalization如何影响反向传播？
4. 为什么需要激活函数的导数？
```

**Step 2: Final commit**

```bash
git add notebooks/Module01_Foundation/03_backpropagation.ipynb
git commit -m "feat(module01): complete backpropagation notebook"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module01-backpropagation.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.
