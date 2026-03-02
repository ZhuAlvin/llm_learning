# Module 4 - 03 GPT Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive notebook on GPT architecture with autoregressive language modeling, decoder-only Transformer, and generation capabilities.

**Architecture:** Follow CODEBASE.md structure, build from decoder-only Transformer to GPT-specific components, implement autoregressive pre-training, demonstrate text generation and few-shot learning.

**Tech Stack:** Jupyter Notebook, NumPy, PyTorch, Transformers, Matplotlib

---

## Task 1: Overview and Autoregressive Modeling

**Files:**
- Create: `notebooks/Module04_PreTraining/03_gpt_architecture.ipynb`

**Step 1: Create notebook with overview**

Add overview explaining GPT's approach (autoregressive pre-training), learning objectives (understand decoder-only architecture, implement GPT, explore in-context learning), and comparison with BERT.

**Step 2: Add imports**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
print("✓ Libraries imported")
```

**Step 3: Add autoregressive LM theory**

Explain autoregressive language modeling: predict next token given all previous tokens, unidirectional attention (causal masking):

$$P(x) = \prod_{i=1}^n P(x_i | x_{<i})$$

Compare with BERT's bidirectional approach.

**Step 4: Demonstrate autoregressive generation**

Create micro-practice showing step-by-step autoregressive generation process.

**Step 5: Commit**

```bash
git add notebooks/Module04_PreTraining/03_gpt_architecture.ipynb
git commit -m "feat(module04): create GPT notebook with autoregressive modeling"
```

---

## Task 2: Decoder-Only Transformer

**Files:**
- Modify: `notebooks/Module04_PreTraining/03_gpt_architecture.ipynb`

**Step 1: Add decoder-only architecture theory**

Explain GPT's architecture choice: uses only Transformer decoder (no encoder), removes cross-attention, keeps causal self-attention. Show differences from encoder-decoder:
- No encoder-decoder attention
- Causal masking in self-attention
- Simpler architecture

**Step 2: Implement causal self-attention**

Build self-attention with causal masking, visualize attention patterns showing triangular mask.

**Step 3: Implement GPT block**

Build single GPT block: causal multi-head attention, layer norm, feed-forward network, residual connections.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/03_gpt_architecture.ipynb
git commit -m "feat(module04): add decoder-only Transformer architecture"
```

---

## Task 3: Complete GPT Architecture

**Files:**
- Modify: `notebooks/Module04_PreTraining/03_gpt_architecture.ipynb`

**Step 1: Add GPT architecture overview**

Explain complete GPT structure:
- Token embeddings + Position embeddings
- Stack of GPT blocks (12/24/48 layers for GPT-2 small/medium/large)
- Language modeling head (linear projection to vocabulary)
- No special tokens except <|endoftext|>

**Step 2: Implement complete GPT model**

Build full GPT model from scratch: embeddings, stacked blocks, LM head, forward pass for training and generation.

**Step 3: Visualize GPT architecture**

Create diagram showing GPT structure, compare side-by-side with BERT architecture.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/03_gpt_architecture.ipynb
git commit -m "feat(module04): implement complete GPT architecture"
```

---

## Task 4: Pre-training and Loss Function

**Files:**
- Modify: `notebooks/Module04_PreTraining/03_gpt_architecture.ipynb`

**Step 1: Add pre-training theory**

Explain GPT pre-training: next-token prediction on large text corpus, cross-entropy loss:

$$\mathcal{L} = -\sum_{i=1}^n \log P(x_i | x_{<i})$$

Discuss training details: BPE tokenization, context window, batch size, learning rate schedule.

**Step 2: Implement pre-training loop**

Build complete pre-training pipeline on toy dataset: data preparation, loss computation, optimization, gradient accumulation.

**Step 3: Monitor training**

Visualize training metrics: loss curve, perplexity, sample generations during training.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/03_gpt_architecture.ipynb
git commit -m "feat(module04): add GPT pre-training process"
```

---

## Task 5: Text Generation with GPT

**Files:**
- Modify: `notebooks/Module04_PreTraining/03_gpt_architecture.ipynb`

**Step 1: Add generation strategies theory**

Explain different generation methods for GPT:
- Greedy decoding
- Beam search
- Top-k sampling
- Nucleus (top-p) sampling
- Temperature scaling
- Repetition penalty

**Step 2: Implement all generation strategies**

Build complete generation functions with all strategies, demonstrate on trained GPT model.

**Step 3: Compare generation quality**

Show examples with different strategies, analyze diversity, coherence, and creativity trade-offs.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/03_gpt_architecture.ipynb
git commit -m "feat(module04): add GPT text generation strategies"
```

---

## Task 6: Few-Shot Learning and Prompting

**Files:**
- Modify: `notebooks/Module04_PreTraining/03_gpt_architecture.ipynb`

**Step 1: Add in-context learning theory**

Explain GPT's emergent ability: few-shot learning without fine-tuning, learns from prompt examples. Show prompt formats:
- Zero-shot: task description only
- One-shot: one example
- Few-shot: multiple examples

**Step 2: Demonstrate few-shot learning**

Build examples showing GPT performing tasks via prompting: classification, translation, QA, reasoning.

**Step 3: Analyze prompt engineering**

Show how prompt design affects performance, demonstrate prompt sensitivity.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/03_gpt_architecture.ipynb
git commit -m "feat(module04): add few-shot learning and prompting"
```

---

## Task 7: GPT Evolution and Summary

**Files:**
- Modify: `notebooks/Module04_PreTraining/03_gpt_architecture.ipynb`

**Step 1: Add GPT evolution overview**

Explain GPT family evolution:
- GPT-1: 117M parameters, proof of concept
- GPT-2: 1.5B parameters, zero-shot capabilities
- GPT-3: 175B parameters, few-shot learning
- GPT-4: multimodal, improved reasoning

**Step 2: Use Transformers library**

Demonstrate using pre-trained GPT-2 from Hugging Face: load model, generate text, fine-tune on custom data.

**Step 3: Add FAQ and summary**

Include questions: BERT vs GPT when to use which? Why is GPT better at generation? How does scale affect capabilities? What's the cost of autoregressive generation?

Complete summary with Module 4 recap, create module README.

**Step 4: Final commit**

```bash
git add notebooks/Module04_PreTraining/03_gpt_architecture.ipynb
git add notebooks/Module04_PreTraining/README.md
git commit -m "feat(module04): complete GPT architecture and module 4"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module04-gpt-architecture.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.

**Module 4 Complete!** All 3 notebooks planned:
1. ✅ Language Modeling Fundamentals
2. ✅ BERT Architecture
3. ✅ GPT Architecture
