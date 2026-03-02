# Module 4 - 01 Language Modeling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive notebook on language modeling fundamentals, from n-gram models to neural language models, understanding perplexity and evaluation metrics.

**Architecture:** Follow CODEBASE.md structure, build from statistical language models to neural language models, implement evaluation metrics, demonstrate on real text data.

**Tech Stack:** Jupyter Notebook, NumPy, PyTorch, NLTK, Matplotlib

---

## Task 1: Overview and Statistical Language Models

**Files:**
- Create: `notebooks/Module04_PreTraining/01_language_modeling.ipynb`

**Step 1: Create notebook with overview**

Add overview explaining language modeling's role in NLP, learning objectives (understand LM fundamentals, implement n-gram models, build neural LM, evaluate with perplexity), and knowledge map.

**Step 2: Add imports**

```python
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from nltk import word_tokenize
import re

np.random.seed(42)
torch.manual_seed(42)
print("✓ Libraries imported")
```

**Step 3: Add language modeling theory**

Explain what is language modeling: predicting next word given context, probability distribution over sequences:

$$P(w_1, w_2, ..., w_n) = \prod_{i=1}^n P(w_i | w_1, ..., w_{i-1})$$

**Step 4: Implement unigram model**

Build simple unigram language model (word frequency), demonstrate generation and probability calculation.

**Step 5: Commit**

```bash
git add notebooks/Module04_PreTraining/01_language_modeling.ipynb
git commit -m "feat(module04): create language modeling notebook with unigram"
```

---

## Task 2: N-gram Language Models

**Files:**
- Modify: `notebooks/Module04_PreTraining/01_language_modeling.ipynb`

**Step 1: Add n-gram theory**

Explain Markov assumption, n-gram models (bigram, trigram), conditional probability:

$$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-n+1}, ..., w_{i-1})$$

**Step 2: Implement bigram and trigram models**

Build n-gram models with smoothing (add-k smoothing, Kneser-Ney), handle unseen n-grams.

**Step 3: Compare n-gram models**

Demonstrate generation quality with different n values, show trade-off between context and sparsity.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/01_language_modeling.ipynb
git commit -m "feat(module04): add n-gram language models"
```

---

## Task 3: Evaluation Metrics

**Files:**
- Modify: `notebooks/Module04_PreTraining/01_language_modeling.ipynb`

**Step 1: Add perplexity theory**

Explain perplexity as evaluation metric: measures how well model predicts test data, lower is better:

$$\text{Perplexity} = 2^{-\frac{1}{N}\sum_{i=1}^N \log_2 P(w_i | \text{context})}$$

Equivalently: $\text{PPL} = \exp(\text{cross-entropy})$

**Step 2: Implement perplexity calculation**

Build function to compute perplexity for any language model, demonstrate on n-gram models.

**Step 3: Add other metrics**

Implement cross-entropy, bits-per-character, compare metrics on different models.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/01_language_modeling.ipynb
git commit -m "feat(module04): add evaluation metrics and perplexity"
```

---

## Task 4: Neural Language Models

**Files:**
- Modify: `notebooks/Module04_PreTraining/01_language_modeling.ipynb`

**Step 1: Add neural LM theory**

Explain transition from n-gram to neural models: distributed representations, no sparsity problem, can handle longer context. Show basic architecture:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
$$P(w_t | w_{<t}) = \text{softmax}(W_o h_t + b_o)$$

**Step 2: Implement RNN language model**

Build simple RNN-based language model with word embeddings, train on text corpus.

**Step 3: Compare with n-gram models**

Evaluate neural LM vs n-gram models on perplexity, generation quality, context handling.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/01_language_modeling.ipynb
git commit -m "feat(module04): add neural language models"
```

---

## Task 5: Modern Neural Architectures

**Files:**
- Modify: `notebooks/Module04_PreTraining/01_language_modeling.ipynb`

**Step 1: Add LSTM language model**

Implement LSTM-based language model, compare with RNN on long-range dependencies.

**Step 2: Add Transformer language model**

Build decoder-only Transformer for language modeling (GPT-style), explain causal masking.

**Step 3: Compare architectures**

Benchmark RNN vs LSTM vs Transformer on perplexity, speed, memory usage.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/01_language_modeling.ipynb
git commit -m "feat(module04): add LSTM and Transformer language models"
```

---

## Task 6: Text Generation Strategies

**Files:**
- Modify: `notebooks/Module04_PreTraining/01_language_modeling.ipynb`

**Step 1: Add generation strategies theory**

Explain different decoding methods: greedy, beam search, sampling, top-k sampling, nucleus (top-p) sampling, temperature scaling.

**Step 2: Implement generation strategies**

Build all generation methods, demonstrate on trained language model.

**Step 3: Compare generation quality**

Show examples with different strategies, analyze diversity vs quality trade-off.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/01_language_modeling.ipynb
git commit -m "feat(module04): add text generation strategies"
```

---

## Task 7: Practical Application and Summary

**Files:**
- Modify: `notebooks/Module04_PreTraining/01_language_modeling.ipynb`

**Step 1: Create comprehensive example**

Build complete language modeling pipeline: data preprocessing, tokenization, training, evaluation, generation on real dataset (e.g., WikiText-2 subset).

**Step 2: Add FAQ section**

Include questions: Why perplexity? How to handle OOV words? What's the difference between LM and classification? How much data is needed? What's the relationship to pre-training?

**Step 3: Add summary**

Complete summary with key takeaways, evolution from n-gram to neural to Transformer, preview of next notebook (BERT architecture).

**Step 4: Final commit**

```bash
git add notebooks/Module04_PreTraining/01_language_modeling.ipynb
git commit -m "feat(module04): complete language modeling notebook"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module04-language-modeling.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.
