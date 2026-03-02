# Module 1 - 04 PyTorch Basics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a comprehensive PyTorch introduction covering tensors, automatic differentiation, model building, and complete training loops.

**Architecture:** Follow CODEBASE.md structure, transition from NumPy to PyTorch, demonstrate automatic differentiation advantages, and build complete training pipeline.

**Tech Stack:** Jupyter Notebook, PyTorch, NumPy, Matplotlib

---

## Task 1: Create Notebook and Tensor Basics

**Files:**
- Create: `notebooks/Module01_Foundation/04_pytorch_basics.ipynb`

**Step 1: Create notebook with overview**

```markdown
# Module 1.4: PyTorch基础

## 1. 本章概览

### 📚 学习目标

1. **Tensor操作**：掌握PyTorch的核心数据结构
2. **自动微分**：理解autograd的工作原理
3. **构建模型**：使用nn.Module构建神经网络
4. **训练循环**：实现完整的训练和评估流程

### 🎯 核心问题

- PyTorch与NumPy有什么区别？
- 自动微分如何工作？
- 如何用PyTorch构建和训练模型？

### ⏱️ 预计学习时间：2-3小时
```

**Step 2: Add imports and tensor basics**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
np.random.seed(42)
```

```markdown
## 2. 动机与背景

### 为什么使用PyTorch？

**NumPy的局限**：
- 需要手动实现反向传播
- 不支持GPU加速
- 没有自动微分

**PyTorch的优势**：
- ✅ 自动微分（autograd）
- ✅ GPU加速
- ✅ 动态计算图
- ✅ 丰富的神经网络模块

## 3. 理论基础

### 3.1 Tensor基础

**Tensor**：PyTorch的核心数据结构，类似NumPy数组但支持GPU。

**创建Tensor的方式**：
- 从数据创建
- 随机初始化
- 特殊值（zeros, ones, eye）
```

**Step 3: Implement tensor operations**

```python
# 🔬 Micro Practice: Tensor Operations
# Goal: Master basic tensor operations
# Expected outcome: Understand tensor creation and manipulation

print("=== Creating Tensors ===\n")

# From Python list
t1 = torch.tensor([1, 2, 3, 4])
print(f"From list: {t1}")
print(f"Shape: {t1.shape}, Dtype: {t1.dtype}\n")

# From NumPy array
arr = np.array([[1, 2], [3, 4]])
t2 = torch.from_numpy(arr)
print(f"From NumPy:\n{t2}")
print(f"Shape: {t2.shape}\n")

# Random tensors
t3 = torch.randn(2, 3)  # Normal distribution
print(f"Random (normal):\n{t3}\n")

# Special tensors
t4 = torch.zeros(2, 2)
t5 = torch.ones(2, 2)
t6 = torch.eye(3)
print(f"Zeros:\n{t4}\n")
print(f"Ones:\n{t5}\n")
print(f"Identity:\n{t6}\n")

print("=== Tensor Operations ===\n")

# Basic operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}  # Element-wise")
print(f"a @ b = {a @ b}  # Dot product\n")

# Matrix operations
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B  # Matrix multiplication
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"C = A @ B shape: {C.shape}\n")

# Reshaping
x = torch.arange(12)
print(f"Original: {x}")
print(f"Reshaped (3, 4):\n{x.reshape(3, 4)}")
print(f"Reshaped (2, 6):\n{x.reshape(2, 6)}\n")

print("✓ Tensor operations completed")
```

**Step 4: Compare with NumPy**

```python
# 🔬 Micro Practice: PyTorch vs NumPy
# Goal: Understand similarities and differences
# Expected outcome: See interoperability

# NumPy array
np_array = np.array([[1, 2], [3, 4]])
print("NumPy array:")
print(np_array)
print(f"Type: {type(np_array)}\n")

# Convert to PyTorch
torch_tensor = torch.from_numpy(np_array)
print("PyTorch tensor (from NumPy):")
print(torch_tensor)
print(f"Type: {type(torch_tensor)}\n")

# Convert back to NumPy
back_to_numpy = torch_tensor.numpy()
print("Back to NumPy:")
print(back_to_numpy)
print(f"Type: {type(back_to_numpy)}\n")

# Note: They share memory!
np_array[0, 0] = 999
print("After modifying NumPy array:")
print(f"NumPy: {np_array}")
print(f"PyTorch: {torch_tensor}")
print("\n⚠️ They share memory! Modifying one affects the other.")
```

**Step 5: Commit**

```bash
git add notebooks/Module01_Foundation/04_pytorch_basics.ipynb
git commit -m "feat(module01): create pytorch basics notebook with tensor operations"
```

---

## Task 2: Automatic Differentiation (Autograd)

**Files:**
- Modify: `notebooks/Module01_Foundation/04_pytorch_basics.ipynb`

**Step 1: Add autograd theory**

```markdown
### 3.2 自动微分 (Automatic Differentiation)

**核心概念**：PyTorch自动跟踪所有操作，构建计算图，然后自动计算梯度。

**关键属性**：
- `requires_grad=True`：告诉PyTorch跟踪这个tensor的操作
- `.grad`：存储计算的梯度
- `.backward()`：触发反向传播

**优势**：
- 不需要手动推导梯度公式
- 不需要手动实现反向传播
- 支持任意复杂的计算图
```

**Step 2: Implement autograd examples**

```python
# 🔬 Micro Practice: Automatic Differentiation
# Goal: Understand how autograd works
# Expected outcome: Compute gradients automatically

print("=== Simple Example: y = x² ===\n")

# Create tensor with gradient tracking
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}\n")

# Forward pass
y = x ** 2
print(f"y = x² = {y}")
print(f"y.requires_grad = {y.requires_grad}\n")

# Backward pass (compute gradients)
y.backward()

# Check gradient: dy/dx = 2x = 2*3 = 6
print(f"dy/dx = {x.grad}")
print(f"Expected: 2*x = 2*3 = 6")
print(f"✓ Gradient computed automatically!\n")

print("=== Complex Example: z = (x + y)² ===\n")

# Multiple inputs
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward
z = (x + y) ** 2
print(f"x = {x.item()}, y = {y.item()}")
print(f"z = (x + y)² = {z.item()}\n")

# Backward
z.backward()

# Check gradients
# dz/dx = 2(x+y) = 2*5 = 10
# dz/dy = 2(x+y) = 2*5 = 10
print(f"dz/dx = {x.grad.item()}")
print(f"dz/dy = {y.grad.item()}")
print(f"Expected: 2(x+y) = 2*5 = 10")
print(f"✓ Multiple gradients computed!\n")
```

**Step 3: Visualize computational graph**

```python
# 🔬 Micro Practice: Visualize Autograd
# Goal: See how PyTorch tracks operations
# Expected outcome: Understand computational graph

# Create a more complex computation
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Forward: y = w*x + b, loss = (y - target)²
y = w * x + b
target = torch.tensor(5.0)
loss = (y - target) ** 2

print("Computation:")
print(f"  x = {x.item()}")
print(f"  w = {w.item()}")
print(f"  b = {b.item()}")
print(f"  y = w*x + b = {y.item()}")
print(f"  target = {target.item()}")
print(f"  loss = (y - target)² = {loss.item()}\n")

# Backward
loss.backward()

print("Gradients:")
print(f"  ∂loss/∂w = {w.grad.item():.4f}")
print(f"  ∂loss/∂b = {b.grad.item():.4f}")
print(f"  ∂loss/∂x = {x.grad.item():.4f}\n")

# Manual verification
print("Manual calculation:")
print(f"  ∂loss/∂w = 2(y-target) * x = 2*({y.item()}-{target.item()}) * {x.item()} = {2*(y.item()-target.item())*x.item():.4f}")
print(f"  ∂loss/∂b = 2(y-target) * 1 = 2*({y.item()}-{target.item()}) = {2*(y.item()-target.item()):.4f}")
print("\n✓ Autograd matches manual calculation!")
```

**Step 4: Commit**

```bash
git add notebooks/Module01_Foundation/04_pytorch_basics.ipynb
git commit -m "feat(module01): add automatic differentiation with autograd"
```

---

## Task 3: Building Models with nn.Module

**Files:**
- Modify: `notebooks/Module01_Foundation/04_pytorch_basics.ipynb`

**Step 1: Add nn.Module theory**

```markdown
### 3.3 构建模型 (Building Models)

**nn.Module**：PyTorch中所有神经网络的基类

**关键方法**：
- `__init__()`: 定义层和参数
- `forward()`: 定义前向传播逻辑

**常用层**：
- `nn.Linear`: 全连接层
- `nn.ReLU`, `nn.Sigmoid`: 激活函数
- `nn.Sequential`: 顺序容器
```

**Step 2: Implement simple model**

```python
# 🔬 Micro Practice: Build Neural Network with nn.Module
# Goal: Create a 2-layer neural network
# Expected outcome: Understand model structure

class SimpleNN(nn.Module):
    """
    Simple 2-layer neural network
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize layers

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units
        """
        super(SimpleNN, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Layer 1
        x = self.fc1(x)
        x = self.relu(x)

        # Layer 2
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

# Create model
model = SimpleNN(input_size=2, hidden_size=4, output_size=1)

print("Model architecture:")
print(model)
print()

# Check parameters
print("Model parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape {param.shape}")
print()

# Test forward pass
x_test = torch.randn(1, 2)  # Batch size 1, 2 features
output = model(x_test)
print(f"Input shape: {x_test.shape}")
print(f"Output shape: {output.shape}")
print(f"Output value: {output.item():.4f}")
print("\n✓ Model created and tested!")
```

**Step 3: Alternative: Sequential model**

```python
# 🔬 Micro Practice: Sequential Model
# Goal: Build model using nn.Sequential
# Expected outcome: Simpler model definition

# Same model using Sequential
model_seq = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

print("Sequential model:")
print(model_seq)
print()

# Test
output_seq = model_seq(x_test)
print(f"Output: {output_seq.item():.4f}")
print("\n✓ Sequential model is more concise for simple architectures!")
```

**Step 4: Commit**

```bash
git add notebooks/Module01_Foundation/04_pytorch_basics.ipynb
git commit -m "feat(module01): add model building with nn.Module"
```

---

## Task 4: Complete Training Loop

**Files:**
- Modify: `notebooks/Module01_Foundation/04_pytorch_basics.ipynb`

**Step 1: Add training loop theory**

```markdown
### 3.4 训练循环 (Training Loop)

**完整训练流程**：

1. **前向传播**：计算预测值
2. **计算损失**：比较预测与真实值
3. **反向传播**：计算梯度
4. **更新参数**：使用优化器

**关键组件**：
- **Loss Function**: `nn.MSELoss()`, `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.SGD()`, `optim.Adam()`
```

**Step 2: Implement complete training**

```python
# 🔬 Micro Practice: Complete Training Loop
# Goal: Train neural network on XOR problem
# Expected outcome: Successfully learn XOR function

# Prepare data
X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print("Training data (XOR):")
for x, y in zip(X_train, y_train):
    print(f"  Input: {x.numpy()}, Target: {y.item()}")
print()

# Create model
model = SimpleNN(input_size=2, hidden_size=8, output_size=1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Training loop
num_epochs = 5000
losses = []

print("Training...")
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters

    # Record loss
    losses.append(loss.item())

    # Print progress
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

print("\nTraining completed!")
print()

# Test the model
print("Final predictions:")
with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(X_train)
    for x, y_true, y_pred in zip(X_train, y_train, predictions):
        print(f"Input: {x.numpy()}, True: {y_true.item()}, "
              f"Pred: {y_pred.item():.4f}, Rounded: {round(y_pred.item())}")

print("\n✓ Model successfully learned XOR function!")
```

**Step 3: Visualize training**

```python
# 🔬 Micro Practice: Visualize Training Progress
# Goal: See loss curve and decision boundary

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(losses, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Decision boundary
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict on grid
with torch.no_grad():
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    Z = model(grid_tensor).numpy()
    Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train.numpy().ravel(),
           cmap='RdYlBu', s=200, edgecolors='black', linewidth=2)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Decision Boundary (XOR Solved!)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ PyTorch successfully trained the model!")
```

**Step 4: Commit**

```bash
git add notebooks/Module01_Foundation/04_pytorch_basics.ipynb
git commit -m "feat(module01): add complete training loop implementation"
```

---

## Task 5: Summary and Capstone Project

**Files:**
- Modify: `notebooks/Module01_Foundation/04_pytorch_basics.ipynb`

**Step 1: Add capstone project**

```markdown
## 6. 综合项目：完整的分类任务

### 项目目标

使用PyTorch实现一个完整的二分类任务，包括：
- 数据准备
- 模型定义
- 训练循环
- 评估和可视化
```

```python
# 🚀 Capstone Project: Binary Classification
# Goal: Complete end-to-end PyTorch workflow
# Expected outcome: Working classification pipeline

# Generate synthetic dataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Create dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

print(f"Training set: {X_train_tensor.shape}")
print(f"Test set: {X_test_tensor.shape}\n")

# Define model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Initialize
model = Classifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 1000
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    train_pred = model(X_train_tensor)
    train_loss = criterion(train_pred, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    test_pred = model(X_test_tensor)
    test_pred_class = (test_pred > 0.5).float()
    accuracy = (test_pred_class == y_test_tensor).float().mean()
    print(f"\nTest Accuracy: {accuracy.item():.2%}")

print("\n✓ Capstone project completed!")
```

**Step 2: Add remaining sections**

```markdown
## 7. 常见问题与调试

### Q1: 什么时候使用.detach()或torch.no_grad()？

**A:**
- `torch.no_grad()`: 推理时，不需要计算梯度
- `.detach()`: 从计算图中分离tensor

### Q2: 为什么要调用optimizer.zero_grad()？

**A:** PyTorch默认累积梯度。每次迭代前需要清零，否则梯度会叠加。

### Q3: model.train()和model.eval()有什么区别？

**A:**
- `model.train()`: 训练模式（Dropout、BatchNorm等会生效）
- `model.eval()`: 评估模式（Dropout关闭、BatchNorm使用统计值）

## 8. 总结与展望

### 核心要点

1. **Tensor**: PyTorch的核心数据结构
2. **Autograd**: 自动微分，无需手动推导梯度
3. **nn.Module**: 构建模型的标准方式
4. **训练循环**: 前向→损失→反向→更新

### Module 1 完成！

恭喜！你已经完成了Module 1的所有内容：
- ✅ 数学基础
- ✅ 神经网络原理
- ✅ 反向传播算法
- ✅ PyTorch实践

### 下一步：Module 2 - 从RNN到Attention

准备学习序列模型和注意力机制！
```

**Step 3: Final commit**

```bash
git add notebooks/Module01_Foundation/04_pytorch_basics.ipynb
git commit -m "feat(module01): complete pytorch basics with capstone project"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module01-pytorch-basics.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.
