# Multimodal Platform

轻量级多模态推理平台 —— 便利店货架监控示范项目。

## 快速开始

```bash
# 1. 启动服务
docker compose up -d

# 2. 验证健康检查
curl http://localhost:8000/health

# 3. 图像分类
curl -X POST http://localhost:8000/classify \
  -F "image=@tests/fixtures/sample.jpg"

# 4. 查看指标
curl http://localhost:8000/metrics
```

## 架构

```
用户请求 → FastAPI (/classify) → PyTorch ViT-Tiny + LoRA → 返回分类结果
                                   ↓
                              ONNX Runtime (可选)
```

## 组件

| 模块 | 功能 |
|------|------|
| `inference_server.py` | FastAPI 推理 API 入口 |
| `model_registry.py` | 模型版本和 LoRA 分支管理 |
| `lora_manager.py` | 多 LoRA 加载和热切换 |
| `quantizer.py` | INT8 PTQ + ONNX 导出 |

## 测试

```bash
pytest tests/ -v
```

## 压测

```bash
python scripts/benchmark.py --concurrency 10 --requests 100
```

## 训练 LoRA

```bash
python scripts/train_lora.py --num-classes 5 --num-shots 10 --epochs 5
```
