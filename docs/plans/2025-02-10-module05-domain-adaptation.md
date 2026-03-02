# Module 5 - 03 Domain Adaptation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive notebook on domain adaptation, continual learning, and preventing catastrophic forgetting when adapting models to new domains.

**Architecture:** Follow CODEBASE.md structure, build from domain shift understanding to adaptation techniques, implement continual learning strategies, demonstrate on domain-specific tasks.

**Tech Stack:** Jupyter Notebook, PyTorch, Transformers, Datasets, Matplotlib

---

## Task 1: Overview and Domain Shift

**Files:**
- Create: `notebooks/Module05_FineTuning/03_domain_adaptation.ipynb`

**Step 1: Create notebook with overview**

Add overview explaining domain adaptation challenges, learning objectives (understand domain shift, implement adaptation methods, handle continual learning), and real-world scenarios.

**Step 2: Add imports**

```python
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

np.random.seed(42)
torch.manual_seed(42)
print("✓ Libraries imported")
```

**Step 3: Add domain shift theory**

Explain domain shift: distribution mismatch between source and target domains. Show examples:
- General → Medical domain
- Formal → Informal text
- English → Code-switched text

Formalize as: $P_{\text{source}}(X, Y) \neq P_{\text{target}}(X, Y)$

**Step 4: Demonstrate domain shift**

Create micro-practice visualizing embeddings from different domains using t-SNE, show distribution differences.

**Step 5: Commit**

```bash
git add notebooks/Module05_FineTuning/03_domain_adaptation.ipynb
git commit -m "feat(module05): create domain adaptation notebook"
```

---

## Task 2: Continued Pre-training

**Files:**
- Modify: `notebooks/Module05_FineTuning/03_domain_adaptation.ipynb`

**Step 1: Add continued pre-training theory**

Explain domain-adaptive pre-training (DAPT): continue pre-training on domain-specific unlabeled data before task fine-tuning. Show the three-stage process:

$$\text{General Pre-training} \rightarrow \text{Domain Pre-training} \rightarrow \text{Task Fine-tuning}$$

**Step 2: Implement continued pre-training**

Build example: take pre-trained BERT, continue MLM training on domain-specific corpus (e.g., biomedical text).

**Step 3: Evaluate domain adaptation**

Compare performance with and without domain pre-training on downstream task.

**Step 4: Commit**

```bash
git add notebooks/Module05_FineTuning/03_domain_adaptation.ipynb
git commit -m "feat(module05): implement continued pre-training"
```

---

## Task 3: Task-Adaptive Pre-training

**Files:**
- Modify: `notebooks/Module05_FineTuning/03_domain_adaptation.ipynb`

**Step 1: Add TAPT theory**

Explain task-adaptive pre-training (TAPT): continue pre-training on task-specific unlabeled data, more focused than DAPT.

**Step 2: Implement TAPT**

Build example combining DAPT and TAPT, show incremental improvements.

**Step 3: Add data selection strategies**

Implement techniques for selecting relevant pre-training data: similarity-based selection, curriculum learning.

**Step 4: Commit**

```bash
git add notebooks/Module05_FineTuning/03_domain_adaptation.ipynb
git commit -m "feat(module05): implement task-adaptive pre-training"
```

---

## Task 4: Catastrophic Forgetting

**Files:**
- Modify: `notebooks/Module05_FineTuning/03_domain_adaptation.ipynb`

**Step 1: Add catastrophic forgetting theory**

Explain catastrophic forgetting: when fine-tuning on new task, model forgets previous tasks. Show the problem:

$$\text{Performance on Task A} \downarrow \text{ after training on Task B}$$

**Step 2: Demonstrate catastrophic forgetting**

Build experiment showing performance degradation on original task after domain adaptation.

**Step 3: Measure forgetting**

Implement metrics to quantify forgetting: backward transfer, forward transfer.

**Step 4: Commit**

```bash
git add notebooks/Module05_FineTuning/03_domain_adaptation.ipynb
git commit -m "feat(module05): demonstrate catastrophic forgetting"
```

---

## Task 5: Continual Learning Strategies

**Files:**
- Modify: `notebooks/Module05_FineTuning/03_domain_adaptation.ipynb`

**Step 1: Add continual learning theory**

Explain strategies to prevent forgetting:
- **Regularization-based**: EWC (Elastic Weight Consolidation), L2 regularization
- **Replay-based**: Experience replay, pseudo-rehearsal
- **Architecture-based**: Progressive networks, adapters per task

**Step 2: Implement EWC**

Build Elastic Weight Consolidation: compute Fisher information, add regularization term to preserve important weights.

**Step 3: Implement experience replay**

Build replay buffer storing examples from previous tasks, mix with new task data during training.

**Step 4: Commit**

```bash
git add notebooks/Module05_FineTuning/03_domain_adaptation.ipynb
git commit -m "feat(module05): implement continual learning strategies"
```

---

## Task 6: Multi-Domain Learning

**Files:**
- Modify: `notebooks/Module05_FineTuning/03_domain_adaptation.ipynb`

**Step 1: Add multi-domain learning theory**

Explain training on multiple domains simultaneously: shared representations, domain-specific components, domain adversarial training.

**Step 2: Implement domain-adversarial training**

Build model with domain classifier, gradient reversal layer for domain-invariant features.

**Step 3: Implement mixture-of-experts**

Build MoE architecture with domain-specific experts, gating mechanism for expert selection.

**Step 4: Commit**

```bash
git add notebooks/Module05_FineTuning/03_domain_adaptation.ipynb
git commit -m "feat(module05): implement multi-domain learning"
```

---

## Task 7: Practical Guidelines and Summary

**Files:**
- Modify: `notebooks/Module05_FineTuning/03_domain_adaptation.ipynb`

**Step 1: Create comprehensive example**

Build end-to-end domain adaptation pipeline: general model → domain pre-training → task fine-tuning with continual learning on multiple domains.

**Step 2: Add best practices**

Document guidelines: when to use DAPT vs TAPT, how to balance old and new tasks, data requirements, evaluation strategies.

**Step 3: Add FAQ and summary**

Include questions: How much domain data is needed? Can we adapt to multiple domains? How to detect domain shift? What if we don't have domain labels?

Complete summary with Module 5 recap, create module README.

**Step 4: Final commit**

```bash
git add notebooks/Module05_FineTuning/03_domain_adaptation.ipynb
git add notebooks/Module05_FineTuning/README.md
git commit -m "feat(module05): complete domain adaptation and module 5"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-02-10-module05-domain-adaptation.md`.

Ready for **parallel execution** using `superpowers:executing-plans`.

**Module 5 Complete!** All 3 notebooks planned:
1. ✅ Fine-tuning Basics
2. ✅ Parameter-Efficient Fine-Tuning (PEFT)
3. ✅ Domain Adaptation & Continual Learning
