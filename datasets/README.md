# 数据集说明

本目录用于存放课程实践使用的轻量级数据集。默认策略是：

- 提供最小可运行样例，保证 CPU 环境可完成核心实验
- 大规模数据集按需下载，不直接纳入仓库

---

## 1. 当前内置下载项

### 1.1 `tiny_shakespeare.txt`

- 用途：Module 4（语言建模与生成实验）
- 大小：约 1MB
- 来源：Andrej Karpathy `char-rnn` 示例数据
- 下载方式：

```bash
python scripts/download_datasets.py
```

或手动下载：

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O datasets/tiny_shakespeare.txt
```

---

## 2. 与脚本的一致性说明

当前下载脚本 [`scripts/download_datasets.py`](../scripts/download_datasets.py) 默认仅下载 `tiny_shakespeare.txt`。

如果你新增了其他数据集，请同时更新：

1. `scripts/download_datasets.py` 的下载列表
2. 本 README 的“数据集列表”与“统计信息”

---

## 3. 推荐扩展数据（按需）

以下数据集建议按实验需要单独下载：

- 问答微调：SQuAD、CMRC、自建 FAQ 数据
- 情感分类：IMDb、SST-2、ChnSentiCorp
- 检索增强：企业知识库文档、技术文档集合

---

## 4. 使用建议

### 4.1 学习阶段

优先使用小数据集快速验证概念，关注“流程跑通 + 指标可解释”。

### 4.2 实验阶段

逐步扩大数据规模，记录以下变化：

- 收敛速度
- 泛化能力
- 训练成本（时间/显存）

### 4.3 生产阶段

使用真实业务数据，并建立：

- 数据版本管理
- 隐私与合规检查
- 数据质量监控

---

## 5. 注意事项

- 数据文件默认不提交到 Git（见 `.gitignore`）
- 首次运行实验前，建议先执行下载脚本
- 请确保磁盘空间与网络环境满足下载需求

---

## 6. 自定义数据集接入流程

1. 将数据文件放入 `datasets/`（或外部存储后在配置中引用）
2. 更新 notebook 中的数据读取路径
3. 更新本 README 的说明与用途映射
4. 如需自动化下载，更新 `scripts/download_datasets.py`

---

准备好数据后，建议从 Module 4 或 Module 5 的相关 notebook 开始实验。
