# 数据集说明

本目录包含用于学习的轻量级数据集。所有数据集都经过精心选择，确保在CPU上也能快速运行。

---

## 📁 数据集列表

### 1. tiny_shakespeare.txt
- **用途**: Module 4 - 语言模型预训练
- **大小**: ~1MB
- **描述**: 莎士比亚作品的小型文本集
- **来源**: Andrej Karpathy's char-rnn
- **下载**: 运行 `python scripts/download_datasets.py`

### 2. simple_qa_pairs.json
- **用途**: Module 5 - 微调实践
- **大小**: ~100KB
- **描述**: 简单的问答对数据集
- **格式**: `{"question": "...", "answer": "..."}`

### 3. sentiment_samples.csv
- **用途**: Module 5 - 情感分类微调
- **大小**: ~50KB
- **描述**: 电影评论情感分类样本
- **格式**: `text,label`

---

## 🚀 下载数据集

运行下载脚本：

```bash
python scripts/download_datasets.py
```

或手动下载：

```bash
# Tiny Shakespeare
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O tiny_shakespeare.txt
```

---

## 📊 数据集统计

| 数据集 | 大小 | 样本数 | 用途 |
|--------|------|--------|------|
| tiny_shakespeare.txt | ~1MB | ~40K字符 | 语言模型预训练 |
| simple_qa_pairs.json | ~100KB | ~500对 | 问答微调 |
| sentiment_samples.csv | ~50KB | ~1000条 | 情感分类 |

---

## 💡 使用建议

1. **学习阶段**: 使用这些轻量级数据集快速验证概念
2. **实验阶段**: 可以扩展到更大的数据集
3. **生产阶段**: 使用真实的业务数据

---

## 🔗 更大数据集

如果需要更大的数据集进行实验：

- **Hugging Face Datasets**: https://huggingface.co/datasets
- **Common Crawl**: https://commoncrawl.org/
- **The Pile**: https://pile.eleuther.ai/

---

## ⚠️ 注意事项

- 数据集文件不会被提交到Git（见.gitignore）
- 首次使用前需要运行下载脚本
- 确保有足够的磁盘空间（至少2GB）

---

## 📝 自定义数据集

您可以添加自己的数据集：

1. 将数据文件放在此目录
2. 更新本README
3. 在Notebook中引用

---

**开始下载数据集，准备学习！** 🚀
