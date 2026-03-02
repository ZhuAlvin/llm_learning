# Module 3 - 03 Transformer Decoder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive notebook on Transformer Decoder architecture with masked self-attention, encoder-decoder attention, and complete Transformer model for sequence-to-sequence tasks.

**Architecture:** Follow CODEBASE.md structure, build from masked attention to decoder layer to full encoder-decoder Transformer, implement complete training pipeline, demonstrate on machine translation.

**Tech Stack:** Jupyter Notebook, NumPy, PyTorch, Matplotlib

---

## Task 1: Overview and Masked Self-Attention

**Files:**
- Create: `notebooks/Module03_Transformer/03_transformer_decoder.ipynb`

**Step 1: Create notebook with overview**

Add overview explaining decoder's role in Transformer, learning objectives (understand masked attention, implement decoder, build complete Transformer), and architecture diagram.

**Step 2: Add imports**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
print("✓ Libraries imported")
```

**Step 3: Add masked attention theory**

Explain why masking is needed: autoregressive generation, prevent looking at future tokens. Show causal mask pattern:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

**Step 4: Implement masked self-attention**

Build NumPy implementation with causal masking, visualize attention patterns showing the triangular mask.

**Step 5: Commit**

```bash
git add notebooks/Module03_Transformer/03_transformer_decoder.ipynb
git commit -m "feat(module03): create decoder notebook with masked attention"
```

---

## Task 2: Encoder-Decoder Attention

**Files:**
- Modify: `notebooks/Module03_Transformer/03_transformer_decoder.ipynb`

**Step 1: Add cross-attention theory**

Explain encoder-decoder attention (cross-attention): Query from decoder, Key and Value from encoder. This allows decoder to attend to encoder outputs.

$$\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}$$

**Step 2: Implement cross-attention**

Build implementation showing how decoder queries encoder memory.

**Step 3: Visualize cross-attention patterns**

Create heatmap showing which decoder positions attend to which encoder positions (alignment visualization).

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/03_transformer_decoder.ipynb
git commit -m "feat(module03): add encoder-decoder attention"
```

---

## Task 3: Single Decoder Layer

**Files:**
- Modify: `notebooks/Module03_Transformer/03_transformer_decoder.ipynb`

**Step 1: Add decoder layer architecture**

Explain complete decoder layer structure:
1. Masked multi-head self-attention
2. Add & Norm
3. Multi-head cross-attention (with encoder)
4. Add & Norm
5. Feed-forward network
6. Add & Norm

**Step 2: Implement decoder layer from scratch**

Build complete NumPy implementation combining masked self-attention, cross-attention, FFN, LayerNorm, residual connections.

**Step 3: Visualize decoder layer**

Create diagram showing information flow through decoder layer with all three sub-layers.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/03_transformer_decoder.ipynb
git commit -m "feat(module03): implement single decoder layer"
```

---

## Task 4: Full Decoder Stack

**Files:**
- Modify: `notebooks/Module03_Transformer/03_transformer_decoder.ipynb`

**Step 1: Add decoder stack theory**

Explain stacking decoder layers, how each layer attends to encoder output, autoregressive generation process.

**Step 2: Implement full decoder**

Build complete decoder with N stacked layers, output embedding, positional encoding, final linear projection and softmax.

**Step 3: Demonstrate autoregressive generation**

Show step-by-step generation process: start with <BOS>, generate one token at a time, use previous outputs as input.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/03_transformer_decoder.ipynb
git commit -m "feat(module03): implement full decoder stack"
```

---

## Task 5: Complete Transformer Model

**Files:**
- Modify: `notebooks/Module03_Transformer/03_transformer_decoder.ipynb`

**Step 1: Add complete Transformer architecture**

Explain full encoder-decoder Transformer: encoder processes source, decoder generates target attending to encoder output.

**Step 2: Implement complete Transformer**

Build full model combining encoder and decoder, add input/output embeddings, implement forward pass for training and generation.

**Step 3: Visualize complete architecture**

Create comprehensive diagram showing encoder stack, decoder stack, and connections between them.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/03_transformer_decoder.ipynb
git commit -m "feat(module03): implement complete Transformer model"
```

---

## Task 6: PyTorch Implementation and Training

**Files:**
- Modify: `notebooks/Module03_Transformer/03_transformer_decoder.ipynb`

**Step 1: Implement PyTorch Transformer**

Build complete PyTorch implementation with proper nn.Module structure, efficient batching, GPU support.

**Step 2: Add training utilities**

Implement label smoothing, learning rate scheduling (warmup + decay), gradient clipping, teacher forcing.

**Step 3: Compare with torch.nn.Transformer**

Show PyTorch's built-in Transformer, verify compatibility with custom implementation.

**Step 4: Commit**

```bash
git add notebooks/Module03_Transformer/03_transformer_decoder.ipynb
git commit -m "feat(module03): add PyTorch Transformer implementation"
```

---

## Task 7: Machine Translation Example and Summary

**Files:**
- Modify: `notebooks/Module03_Transformer/03_transformer_decoder.ipynb`

**Step 1: Create translation example**

Build complete machine translation example: prepare toy dataset (e.g., number sequences or simple phrases), train Transformer, demonstrate translation with attention visualization.

**Step 2: Add generation strategies**

Implement greedy decoding, beam search, nucleus sampling. Compare generation quality.

**Step 3: Add FAQ and summary**

Include questions: Encoder-only vs decoder-only vs encoder-decoder? How to handle different sequence lengths? What's the memory complexity? How to improve generation quality?

Complete summary with Module 3 recap and preview of Module 4 (pre-training).

**Step 4: Create module README**

Document Module 3 overview, list all notebooks, learning path.

**Step 5: Final commit**

```bash
git add notebooks/Module03_Transformer/03_transformer_decoder.ipynb
git add notebooks/Module03_Transformer/README.md
git commit -m "feat(module03): complete Transformer decoder and module 3"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module03-transformer-decoder.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.

**Module 3 Complete!** All 3 notebooks planned:
1. ✅ Self-Attention
2. ✅ Transformer Encoder
3. ✅ Transformer Decoder & Complete Architecture
