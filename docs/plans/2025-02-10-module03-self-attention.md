# Module 3 - 01 Self-Attention Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive notebook on Self-Attention mechanism with Query-Key-Value formulation, multi-head attention, and positional encoding.

**Architecture:** Follow CODEBASE.md structure, build from basic attention to scaled dot-product attention to multi-head attention, implement from scratch with NumPy then PyTorch, visualize attention patterns.

**Tech Stack:** Jupyter Notebook, NumPy, PyTorch, Matplotlib, Seaborn

---

## Task 1: Overview and Motivation

**Files:**
- Create: `notebooks/Module03_Transformer/01_self_attention.ipynb`

**Step 1: Create notebook with overview**

Add overview section explaining self-attention's role in Transformers, learning objectives (understand Q-K-V mechanism, implement multi-head attention, understand positional encoding), and knowledge map.

**Step 2: Add imports and setup**

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
torch.manual_seed(42)
print("✓ Libraries imported")
```

**Step 3: Add motivation section**

Explain why self-attention is needed: RNN processes sequentially (slow), attention in Seq2Seq only looks at encoder outputs, self-attention allows parallel processing and direct connections between all positions.

**Step 4: Demonstrate the problem**

Create micro-practice showing limitation of sequential processing vs parallel attention.

**Step 5: Commit**

```bash
git add notebooks/Module03_Transformer/01_self_attention.ipynb
git commit -m "feat(module03): create self-attention notebook with motivation"
```

---

## Task 2: Query-Key-Value Mechanism

**Files:**
- Modify: `notebooks/Module03_Transformer/01_self_attention.ipynb`

**Step 1: Add Q-K-V theory**

Explain the Query-Key-Value formulation: Query (what I'm looking for), Key (what I have), Value (what I return). Include mathematical formulation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step 2: Implement basic self-attention from scratch**

Build NumPy implementation with clear steps: linear projections to Q/K/V, compute attention scores, apply softmax, weighted sum of values.

**Step 3: Visualize attention weights**

Create heatmap showing which positions attend to which other positions.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/01_self_attention.ipynb
git commit -m "feat(module03): add Q-K-V mechanism and basic self-attention"
```

---

## Task 3: Scaled Dot-Product Attention

**Files:**
- Modify: `notebooks/Module03_Transformer/01_self_attention.ipynb`

**Step 1: Add scaling theory**

Explain why we divide by sqrt(d_k): prevents dot products from growing too large, keeps gradients stable. Show mathematical analysis of variance.

**Step 2: Implement scaled dot-product attention**

Complete NumPy implementation with proper scaling, demonstrate effect of scaling with and without.

**Step 3: Add masking support**

Implement attention masking for padding and causal (autoregressive) attention.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/01_self_attention.ipynb
git commit -m "feat(module03): implement scaled dot-product attention with masking"
```

---

## Task 4: Multi-Head Attention

**Files:**
- Modify: `notebooks/Module03_Transformer/01_self_attention.ipynb`

**Step 1: Add multi-head theory**

Explain motivation: multiple attention heads can focus on different aspects (syntax, semantics, etc.). Include formulation:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**Step 2: Implement multi-head attention from scratch**

Build complete NumPy implementation with multiple heads, concatenation, and output projection.

**Step 3: Visualize multiple heads**

Create visualization showing what different heads attend to.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/01_self_attention.ipynb
git commit -m "feat(module03): implement multi-head attention"
```

---

## Task 5: Positional Encoding

**Files:**
- Modify: `notebooks/Module03_Transformer/01_self_attention.ipynb`

**Step 1: Add positional encoding theory**

Explain why needed: self-attention is permutation invariant, need to inject position information. Introduce sinusoidal encoding:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**Step 2: Implement positional encoding**

Build both sinusoidal and learned positional encodings.

**Step 3: Visualize positional encodings**

Create heatmap showing positional encoding patterns, demonstrate relative position properties.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/01_self_attention.ipynb
git commit -m "feat(module03): add positional encoding"
```

---

## Task 6: PyTorch Implementation

**Files:**
- Modify: `notebooks/Module03_Transformer/01_self_attention.ipynb`

**Step 1: Implement PyTorch multi-head attention**

Build complete PyTorch nn.Module implementation with proper batching and GPU support.

**Step 2: Compare with torch.nn.MultiheadAttention**

Show how to use PyTorch's built-in implementation, compare outputs with custom implementation.

**Step 3: Performance comparison**

Benchmark NumPy vs PyTorch vs torch.nn implementations.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/01_self_attention.ipynb
git commit -m "feat(module03): add PyTorch self-attention implementation"
```

---

## Task 7: Comprehensive Example and Summary

**Files:**
- Modify: `notebooks/Module03_Transformer/01_self_attention.ipynb`

**Step 1: Create comprehensive example**

Build complete example processing a sentence, showing attention patterns at each step.

**Step 2: Add FAQ section**

Include common questions: Why Q-K-V instead of just one matrix? Why multi-head? How to choose number of heads? What's the computational complexity?

**Step 3: Add summary and thinking questions**

Complete summary section with key takeaways, connections to Transformer architecture, and thinking questions.

**Step 4: Final commit**

```bash
git add notebooks/Module03_Transformer/01_self_attention.ipynb
git commit -m "feat(module03): complete self-attention notebook"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module03-self-attention.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.
