# Module 4 - 02 BERT Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive notebook on BERT architecture with masked language modeling, next sentence prediction, and bidirectional encoding understanding.

**Architecture:** Follow CODEBASE.md structure, build from Transformer encoder to BERT-specific components (MLM, NSP), implement pre-training objectives, demonstrate fine-tuning.

**Tech Stack:** Jupyter Notebook, NumPy, PyTorch, Transformers, Matplotlib

---

## Task 1: Overview and Bidirectional Context

**Files:**
- Create: `notebooks/Module04_PreTraining/02_bert_architecture.ipynb`

**Step 1: Create notebook with overview**

Add overview explaining BERT's innovation (bidirectional pre-training), learning objectives (understand MLM and NSP, implement BERT, fine-tune for downstream tasks), and comparison with GPT.

**Step 2: Add imports**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
torch.manual_seed(42)
print("✓ Libraries imported")
```

**Step 3: Add bidirectional encoding theory**

Explain why bidirectional context matters: left-to-right LM can't see future context, BERT uses both directions. Compare:
- GPT (autoregressive): $P(w_i | w_{<i})$
- BERT (masked): $P(w_i | w_{<i}, w_{>i})$

**Step 4: Demonstrate bidirectional advantage**

Create micro-practice showing tasks where bidirectional context helps (e.g., fill-in-the-blank, sentiment analysis).

**Step 5: Commit**

```bash
git add notebooks/Module04_PreTraining/02_bert_architecture.ipynb
git commit -m "feat(module04): create BERT notebook with bidirectional context"
```

---

## Task 2: Masked Language Modeling (MLM)

**Files:**
- Modify: `notebooks/Module04_PreTraining/02_bert_architecture.ipynb`

**Step 1: Add MLM theory**

Explain masked language modeling objective: randomly mask 15% of tokens, predict masked tokens using bidirectional context. Masking strategy:
- 80% replace with [MASK]
- 10% replace with random token
- 10% keep original

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(w_i | \text{context})$$

**Step 2: Implement MLM data preparation**

Build function to create MLM training data: random masking, special token handling, attention masks.

**Step 3: Implement MLM head**

Build MLM prediction head on top of BERT encoder, compute masked token loss.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/02_bert_architecture.ipynb
git commit -m "feat(module04): add masked language modeling"
```

---

## Task 3: Next Sentence Prediction (NSP)

**Files:**
- Modify: `notebooks/Module04_PreTraining/02_bert_architecture.ipynb`

**Step 1: Add NSP theory**

Explain next sentence prediction objective: binary classification whether sentence B follows sentence A, helps with sentence-pair tasks (QA, NLI).

$$\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} | [CLS])$$

**Step 2: Implement NSP data preparation**

Build function to create NSP training data: 50% actual next sentences, 50% random sentences.

**Step 3: Implement NSP head**

Build binary classification head using [CLS] token representation.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/02_bert_architecture.ipynb
git commit -m "feat(module04): add next sentence prediction"
```

---

## Task 4: Complete BERT Architecture

**Files:**
- Modify: `notebooks/Module04_PreTraining/02_bert_architecture.ipynb`

**Step 1: Add BERT architecture overview**

Explain complete BERT structure:
- Token embeddings + Segment embeddings + Position embeddings
- Transformer encoder stack (12 or 24 layers)
- MLM head + NSP head
- Special tokens: [CLS], [SEP], [MASK], [PAD]

**Step 2: Implement BERT from scratch**

Build complete BERT model combining all components: embeddings, encoder, MLM head, NSP head.

**Step 3: Visualize BERT architecture**

Create comprehensive diagram showing input processing, encoder stack, and dual prediction heads.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/02_bert_architecture.ipynb
git commit -m "feat(module04): implement complete BERT architecture"
```

---

## Task 5: Pre-training Process

**Files:**
- Modify: `notebooks/Module04_PreTraining/02_bert_architecture.ipynb`

**Step 1: Add pre-training theory**

Explain BERT pre-training: large corpus (Wikipedia + BookCorpus), combined MLM + NSP loss, training details (batch size, learning rate, warmup).

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

**Step 2: Implement pre-training loop**

Build complete pre-training pipeline on toy dataset: data loading, combined loss, optimization, checkpointing.

**Step 3: Monitor pre-training**

Visualize training curves: MLM accuracy, NSP accuracy, total loss over time.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/02_bert_architecture.ipynb
git commit -m "feat(module04): add BERT pre-training process"
```

---

## Task 6: Fine-tuning for Downstream Tasks

**Files:**
- Modify: `notebooks/Module04_PreTraining/02_bert_architecture.ipynb`

**Step 1: Add fine-tuning theory**

Explain fine-tuning paradigm: use pre-trained BERT, add task-specific head, train on labeled data. Show different task types:
- Classification: use [CLS] representation
- Token classification: use all token representations
- Span prediction: use token pairs

**Step 2: Implement classification fine-tuning**

Build example: sentiment classification, add classification head, fine-tune on labeled data.

**Step 3: Compare pre-training vs from-scratch**

Train same model with and without pre-training, compare performance and convergence speed.

**Step 4: Commit**

```bash
git add notebooks/Module04_PreTraining/02_bert_architecture.ipynb
git commit -m "feat(module04): add BERT fine-tuning"
```

---

## Task 7: Using Transformers Library and Summary

**Files:**
- Modify: `notebooks/Module04_PreTraining/02_bert_architecture.ipynb`

**Step 1: Demonstrate Transformers library**

Show how to use pre-trained BERT from Hugging Face: load model, tokenizer, inference, fine-tuning with Trainer API.

**Step 2: Compare custom vs library implementation**

Verify outputs match between custom BERT and transformers.BertModel.

**Step 3: Add FAQ and summary**

Include questions: Why MLM instead of standard LM? Is NSP necessary? How to choose BERT variant (base/large)? What tasks benefit most from BERT? BERT vs GPT trade-offs?

Complete summary with BERT's impact, preview of GPT architecture.

**Step 4: Final commit**

```bash
git add notebooks/Module04_PreTraining/02_bert_architecture.ipynb
git commit -m "feat(module04): complete BERT architecture notebook"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module04-bert-architecture.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.
