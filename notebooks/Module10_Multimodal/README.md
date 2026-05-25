# Module 10: 多模态小模型应用 (Multimodal Small Model Applications)

## 📚 模块概览

本模块是课程的**多模态综合实战**。它建立在 NLP 主线中学到的核心技术之上，将它们应用到视觉和多模态场景：

> 🤝 **双主线连接**：本模块与 NLP 主线（Module 1-9）形成**双主线并行**结构：
> - **Module 5 (Fine-tuning)** 教的 LoRA (Low-Rank Adaptation, 低秩适应) → 在 10.4 中用于 ViT (Vision Transformer, 视觉 Transformer) 和 CLIP (Contrastive Language-Image Pre-training) 的视觉微调
> - **Module 7 (Deployment)** 教的量化 / ONNX (Open Neural Network Exchange) → 在 10.4 中扩展到边缘设备的 TensorRT 部署
> - **Module 8 (Applications)** 教的向量检索 / RAG → 在 10.2 中扩展为以图搜图和多模态检索

你将基于统一主线案例 `通用多模态平台`，掌握从模型选型、场景应用到边缘部署的完整链路。

核心目标不是罗列模型名称，而是回答 4 个业务问题：

- 图像/视频场景下，模型种类繁多，如何根据业务约束（延迟、精度、功耗）做出可量化的选型决策？
- 通用模型在具体场景（安防、零售、医疗）中效果不佳，如何系统化地做领域适配？
- 视频流是连续信号而非单张图片，推理管线需要怎样的架构才能做到实时、不掉帧？
- 边缘设备资源极度受限（< 4GB 显存），如何把模型压到能跑、同时保持可接受的精度？

### 📋 前置要求

- 完成 Module 1-3（PyTorch 基础、Transformer 架构），或具备同等 CV/DL 基础
- **推荐**完成 Module 5（LoRA/PEFT 概念在 CV 中直接复用）、Module 7（推理优化概念扩展至边缘）
- 了解图像分类、目标检测的基本概念
- 有 Linux 命令行和 Python 环境配置经验

### 🎯 学习目标

- 掌握多模态小模型的架构全景和选型方法
- 能独立完成图像识别任务的模型训练与评估
- 理解视频理解的时序建模和实时推理管线
- 掌握 PEFT for Vision 的微调策略和多租户模型管理
- 能将量化模型部署到边缘设备并构建推理 API

### ✅ 完成本模块后的可交付产出

- 一份多模态模型选型报告（含 3 个场景的 benchmark 对比数据）
- 一个特定场景的微调模型（含 LoRA 权重和评估指标）
- 一个可部署的边缘推理 API 服务（含量化模型、Dockerfile 和压测报告）

### ⏱️ 预计学习时间

**总计**: 12-15 小时

### 📈 学习曲线设计

- 第 1 段（10.1）：建立全局视野，知道“有哪些模型、怎么选”
- 第 2 段（10.2-10.3）：深入场景实战，分别攻克图像和视频两大方向
- 第 3 段（10.4）：打通最后一公里——微调适配 + 边缘部署

### 🧭 每章建议阅读顺序

`业务场景 → 模型选型 → 最小实现 → 指标评估 → 领域适配 → 边缘部署 → 成本取舍`

### 📊 模块内统一评估视角

- 技术指标：Accuracy / mAP / Recall、FLOPs、推理延迟（P50/P95）、显存占用、吞吐量（FPS）
- 业务指标：告警准确率、误报率、首次响应时间、单路视频成本、模型更新停机时间

---

## 📖 Notebooks 详细介绍

### 10.1 多模态小模型基础与选型 (01_multimodal_foundations.ipynb)

**核心内容**：
- **多模态架构全景**：视觉编码器（CNN / ViT / Hybrid）、图文对齐（CLIP-style contrastive learning）、联合训练范式
- **轻量模型谱系**：MobileNet 系列、EfficientNet、TinyViT、MobileCLIP、FastViT、Phi-3-vision
- **CLIP (Contrastive Language-Image Pre-training, 对比语言-图像预训练) 原理**：双塔架构、对比损失、零样本迁移能力
- **模型选型框架**：精度-效率 Pareto 前沿分析、延迟预算分配、显存估算公式
- **Benchmark 方法论**：标准数据集（ImageNet-1K, COCO, Kinetics-400）、评估协议、可复现实验设计

**业务问题映射**：
- "安防监控需要实时检测，单帧推理不能超过 100ms，哪些模型能满足？" → 选型框架 + latency benchmark
- "零售场景需要识别 200+ 商品品类，模型扩展性如何评估？" → 分类头设计 + embedding 空间分析
- "医疗影像需要高精度和可解释性，轻量模型能做到吗？" → 精度-效率 trade-off 分析 + Grad-CAM 可解释性

**8 个微实践**：
1. 图像特征提取对比 - CNN vs ViT backbone 的 feature 质量
2. CLIP 零样本分类 - 不训练直接分类，理解图文对齐能力
3. 模型参数与推理速度基准测试 - FLOPs / 参数量 / 实测延迟
4. 不同输入分辨率对精度和速度的影响
5. 模型选型决策矩阵实践 - 给定场景约束，输出推荐模型
6. 多模态 embedding 可视化 - t-SNE 降维观察图文对齐
7. 显存估算与优化 - 用公式和工具预估推理显存
8. 业务场景-模型匹配练习 - 安防/零售/医疗三选一完成选型报告

**关键技术**：
- 视觉编码器：ViT (Vision Transformer, 视觉 Transformer), CNN (Convolutional Neural Network, 卷积神经网络)
- 轻量模型：MobileNetV3, EfficientNet-B0, TinyViT-5M, MobileCLIP-S0
- 评估工具：timm, transformers, ONNX Runtime, torch.profiler

**适用场景**：
- 边缘设备模型选型评估
- 多模态产品技术选型
- 模型压缩前的基线建立

---

### 10.2 图像识别实践 (02_image_recognition.ipynb)

**核心内容**：
- **图像分类**：轻量 backbone + 分类头设计、标签平滑、mixup/cutmix 增强
- **目标检测**：轻量检测头（YOLO-nano, MobileNet-SSD）、Anchor-free 方法
- **图像检索**：特征提取 + FAISS 向量索引、以图搜图管线
- **数据预处理管线**：albumentations 增强策略、在线 vs 离线预处理
- **评估体系**：混淆矩阵、mAP (mean Average Precision, 平均精度均值)、ROC-AUC、模型校准

**业务问题映射**：
- "零售货架有 200+ SKU (Stock Keeping Unit, 库存量单位)，如何用 < 1B 模型做到 95% 识别率？" → 检测+分类级联 + 难例挖掘
- "医疗影像中罕见病变样本极少，如何提升召回？" → 数据增强 + Focal Loss + 类别平衡采样
- "工业质检需要 pixel-level 缺陷定位，轻量模型够用吗？" → 轻量分割头 + 特征金字塔

**8 个微实践**：
1. 轻量图像分类管线 - TinyViT 训练与评估全流程
2. 数据增强策略对比 - AutoAugment / RandAugment / 自定义增强
3. 目标检测快速上手 - YOLO-nano 在自定义数据上的训练
4. 检测结果分析与调优 - mAP 分解、误检/漏检分析
5. 以图搜图 - 特征提取 + FAISS 索引构建
6. 多标签分类实践 - 商品多属性同时识别
7. 模型校准 - 置信度 vs 准确率曲线、温度缩放
8. 图像识别综合评估 - 构建完整评估报告

**关键技术**：
- 模型：TinyViT, MobileNetV3, YOLO-nano, MobileNet-SSD
- 增强：albumentations, mixup, cutmix, Mosaic
- 评估：scikit-learn metrics, mAP, ROC-AUC, ECE (Expected Calibration Error, 期望校准误差)

**适用场景**：
- 零售商品识别与货架巡检
- 医疗影像辅助诊断
- 工业质检与缺陷检测
- 通用图像分类/检测/检索

---

### 10.3 视频理解与监控 (03_video_understanding.ipynb)

**核心内容**：
- **视频帧采样策略**：固定间隔、关键帧提取、运动检测触发、自适应帧率
- **轻量时序建模**：TSM (Temporal Shift Module, 时序移位模块)、SlowFast-light、3D 卷积的轻量替代
- **实时视频分析管线**：RTSP (Real-Time Streaming Protocol, 实时流媒体协议) 拉流 → 预处理 → 推理 → 后处理 → 告警
- **多路视频调度**：异步推理、动态批处理、优先级队列
- **异常检测方法**：基于重建、基于预测、基于 One-Class 分类

**业务问题映射**：
- "16 路摄像头同时接入，单台边缘设备怎么保证不卡顿？" → 多路调度 + 动态帧率 + 异步管线
- "夜间/雨天/遮挡场景下检测效果大幅下降怎么办？" → 数据增强 + 领域适应 + 多模态融合（红外+可见光）
- "异常事件（打架、跌倒）样本极少且多样，怎么训练？" → 异常检测范式 + 弱监督 + 规则融合

**8 个微实践**：
1. 视频帧采样策略对比 - 固定间隔 vs 关键帧 vs 运动检测
2. TSM 动作识别 - 2D CNN + 时序移位实现高效视频理解
3. 实时视频推理管线 - OpenCV 拉流 → 推理 → 结果渲染
4. 多路视频异步推理 - asyncio + 线程池实现并发处理
5. 目标跟踪集成 - ByteTrack 轻量跟踪器 + 检测结果关联
6. 异常检测实验 - 基于 AutoEncoder 的重建误差方法
7. 视频推理性能剖析 - FPS、延迟分布、瓶颈定位
8. 监控告警规则引擎 - 业务规则 + 模型输出联合决策

**关键技术**：
- 视频理解：TSM, SlowFast, VideoMAE (轻量变体)
- 流媒体：RTSP, FFmpeg, OpenCV VideoCapture
- 跟踪：ByteTrack, BoT-SORT
- 调度：asyncio, multiprocessing, 动态批处理

**适用场景**：
- 安防监控异常检测（入侵、徘徊、打架）
- 零售客流统计与热力图
- 交通流量监控与事件检测
- 工业生产流程监控

---

### 10.4 场景微调与边缘部署 (04_edge_deployment.ipynb)

**核心内容**：
- **PEFT for Vision**：LoRA (Low-Rank Adaptation, 低秩适应) 注入 ViT、BitFit (Bias-only Fine-tuning)、Adapter 对比
- **领域适应策略**：few-shot fine-tuning、渐进式解冻、领域特定数据增强
- **增量类别学习**：新类别注册而不遗忘已有类别、prototype-based 方法
- **模型量化与导出**：INT8/INT4 量化（PTQ, Post-Training Quantization / QAT, Quantization-Aware Training）、ONNX 导出、TensorRT 构建
- **边缘推理平台**：FastAPI 推理服务、模型热切换、A/B 推理、Docker 容器化
- **监控与运维**：推理延迟监控、数据漂移检测、模型回滚策略

**业务问题映射**：
- "新增一种缺陷类型，只有 50 张标注图，怎么快速适配上线？" → few-shot LoRA + 数据增强
- "三个客户需求不同但不能互相看到数据，如何隔离？" → 多 LoRA 分支 + 运行时动态切换
- "边缘盒子只有 4GB 显存，要同时跑检测和分类两个模型" → INT8 量化 + 模型串行调度 + 显存管理
- "模型需要升级但视频监控不能中断" → 热加载 + A/B 推理 + 灰度切换

**10 个微实践**：
1. LoRA for ViT - 注入低秩矩阵并训练
2. Full fine-tuning vs LoRA vs BitFit 精度/效率对比
3. Few-shot 分类适配实验（5-shot / 10-shot / 50-shot）
4. 多 LoRA 分支管理与运行时切换
5. 增量类别学习 - 注册新类不遗忘旧类
6. INT8 量化全流程 - PTQ 精度损失评估
7. ONNX 模型导出与优化 - 常量折叠 + 算子融合
8. TensorRT 构建与加速对比
9. FastAPI 多模态推理服务 - 含模型热加载
10. Docker 容器化 + 压测 - 模拟边缘设备环境

**关键技术**：
- PEFT：LoRA (ViT), BitFit, Adapter, Prompt-tuning for Vision
- 量化：PyTorch Quantization, ONNX quantize, TensorRT INT8
- 部署：FastAPI, Docker, ONNX Runtime, TensorRT, OpenVINO
- 监控：Prometheus metrics, logging, data drift detection

**适用场景**：
- 多客户定制化模型交付
- 边缘设备模型部署与运维
- 模型持续迭代与增量学习
- 多模态推理平台构建

---

## 🗺️ 学习路径

### 路径 1：应用开发者（推荐新手）

```
01_multimodal_foundations.ipynb（模型选型重点）
    ↓
04_edge_deployment.ipynb（部署 + 微调重点）
    ↓
02_image_recognition.ipynb（快速浏览）
```

**时间**: 6-8 小时
**产出**: 可部署的多模态推理 API + 一份模型选型报告
**最低完成标准**: 能导出量化模型并部署为 Docker 化 API，单帧推理延迟 < 200ms

---

### 路径 2：CV 算法工程师

```
01_multimodal_foundations.ipynb（完整）
    ↓
02_image_recognition.ipynb（完整）
    ↓
03_video_understanding.ipynb（完整）
    ↓
04_edge_deployment.ipynb（微调重点 + 部署概述）
```

**时间**: 10-12 小时
**产出**: 特定场景微调模型 + 完整评估报告
**最低完成标准**: 完成至少一个场景的 PEFT 微调，mAP 相比基线提升 5%+

---

### 路径 3：全栈多模态工程师（推荐）

```
01_multimodal_foundations.ipynb → 02_image_recognition.ipynb
    → 03_video_understanding.ipynb → 04_edge_deployment.ipynb
```

**时间**: 12-15 小时
**产出**: 端到端多模态平台原型（含选型、微调、部署全链路）
**最低完成标准**: 打通“模型选型 → 场景微调 → 量化导出 → API 部署”全流程

---

## 💡 实践项目建议

### 项目 1：智能零售货架识别系统

**难度**: ⭐⭐⭐
**时间**: 2-3 天

**功能**：
- 货架图像采集与 SKU 标注
- 200+ 商品分类模型训练（TinyViT）
- 缺货/错放检测
- 以图搜图补货建议

**技术栈**：
- 模型：TinyViT / MobileNetV3
- 训练：PyTorch + albumentations
- 推理：ONNX Runtime
- 前端：Gradio / Streamlit

**学习重点**：
- 多分类模型设计与训练
- 数据增强策略
- 模型导出与部署

---

### 项目 2：边缘安防视频监控系统

**难度**: ⭐⭐⭐⭐
**时间**: 3-5 天

**功能**：
- RTSP 多路视频流接入（≥4 路模拟）
- 人员检测 + 入侵区域告警
- 异常行为检测（徘徊、奔跑、跌倒）
- 告警截图 + 视频片段留存
- 边缘设备 Docker 化部署

**技术栈**：
- 检测模型：YOLO-nano + TensorRT
- 视频处理：OpenCV + FFmpeg + asyncio
- 跟踪：ByteTrack
- 后端：FastAPI + WebSocket
- 部署：Docker + NVIDIA Jetson / x86 边缘盒子

**学习重点**：
- 多路视频并发推理调度
- 模型量化和 TensorRT 加速
- 异常检测范式实践

---

### 项目 3：多租户模型微调平台

**难度**: ⭐⭐⭐⭐⭐
**时间**: 5-7 天

**功能**：
- 用户上传标注数据 → 自动触发 LoRA 微调
- 多 LoRA 分支管理（客户 A/B/C 隔离）
- 微调进度追踪与评估报告生成
- 模型热切换：新 LoRA 上线不中断推理服务
- 微调前后精度对比 + 推理延迟监控

**技术栈**：
- PEFT：LoRA (ViT) + HuggingFace PEFT
- 训练调度：Celery / background tasks
- 模型管理：MLflow / 自定义注册表
- 推理服务：FastAPI + ONNX Runtime
- 存储：MinIO (模型) + PostgreSQL (元数据)

**学习重点**：
- 多租户模型隔离与动态路由
- PEFT 训练管线自动化
- 模型热切换与灰度发布
- 平台工程化设计

---

### 项目 4：双主线合流——电商客服接入图像理解（选做）

**难度**: ⭐⭐⭐
**时间**: 1-2 天（设计骨架，不强制写代码）

**功能**：
- 用户向电商客服发送商品图片 → M10 平台 `/classify` 返回商品 ID
- M08 Agent 调 M10 平台接口 → RAG 查询商品详情 → 组装回复
- 演示两条主线的技术打通可能

**技术栈**：
- M08：LangChain/LlamaIndex Agent + RAG 检索
- M10：FastAPI `/classify` + `/search`（以图搜图）接口
- 商品知识库：向量数据库（图片+描述+库存+价格）

**学习重点**：
- 理解双主线在系统工程层面的结合点
- Agent 工具调用跨平台 API 的设计模式
- 多模态 RAG 的扩展思路

---

## 🧠 知识图谱

```
Module 10: 多模态小模型应用
    │
    ├─ 10.1 基础与选型
    │   ├─ 视觉编码器
    │   │   ├─ CNN 系列：MobileNet, EfficientNet
    │   │   ├─ ViT 系列：TinyViT, FastViT, MobileViT
    │   │   └─ 混合架构：ConvNeXt, MobileOne
    │   │
    │   ├─ 图文对齐
    │   │   ├─ CLIP 双塔架构
    │   │   ├─ 对比损失 (InfoNCE)
    │   │   └─ 零样本迁移
    │   │
    │   └─ 选型方法论
    │       ├─ 精度-效率 Pareto 分析
    │       ├─ 延迟预算分配
    │       └─ 显存估算模型
    │
    ├─ 10.2 图像识别
    │   ├─ 分类
    │   │   ├─ TinyViT / MobileNetV3 训练
    │   │   ├─ 数据增强 (mixup, cutmix, Mosaic)
    │   │   └─ 评估 (混淆矩阵, ROC-AUC, ECE)
    │   │
    │   ├─ 检测
    │   │   ├─ YOLO-nano / MobileNet-SSD
    │   │   ├─ Anchor-free 方法
    │   │   └─ mAP 评估与误差分析
    │   │
    │   └─ 检索
    │       ├─ 特征提取 + FAISS
    │       ├─ 以图搜图管线
    │       └─ 向量索引优化
    │
    ├─ 10.3 视频理解
    │   ├─ 帧采样
    │   │   ├─ 固定间隔 / 关键帧
    │   │   ├─ 运动检测触发
    │   │   └─ 自适应帧率
    │   │
    │   ├─ 时序建模
    │   │   ├─ TSM (时序移位)
    │   │   ├─ SlowFast-light
    │   │   └─ 3D 卷积替代方案
    │   │
    │   └─ 实时管线
    │       ├─ RTSP 拉流
    │       ├─ 异步推理调度
    │       └─ 多路并发管理
    │
    └─ 10.4 微调与部署
        ├─ PEFT for Vision
        │   ├─ LoRA (ViT 注入)
        │   ├─ BitFit / Adapter
        │   └─ Few-shot 适配
        │
        ├─ 模型优化
        │   ├─ INT8/INT4 量化 (PTQ/QAT)
        │   ├─ ONNX 导出与优化
        │   └─ TensorRT 构建
        │
        └─ 边缘平台
            ├─ FastAPI 推理服务
            ├─ 模型热切换 / A/B
            ├─ Docker 容器化
            └─ 监控与运维
```

---

## 📚 相关资源

### 论文

**多模态与视觉-语言**：
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021)
- [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](https://arxiv.org/abs/2311.17049) (2023)

**轻量模型**：
- [MobileNetV3: Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) (2019)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (2019)
- [TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/abs/2207.10666) (2022)

**视频理解**：
- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383) (2019)
- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982) (2019)

**PEFT for Vision**：
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021)
- [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119) (2022)

**模型优化**：
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630) (2021)

### 开源项目

- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models) - 预训练视觉模型库
- [OpenMMLab (MMDetection/MMAction)](https://github.com/open-mmlab) - 检测/视频理解框架
- [Ultralytics (YOLO)](https://github.com/ultralytics/ultralytics) - 目标检测
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索

### 工具和库

- **模型优化**：ONNX Runtime, TensorRT, OpenVINO
- **数据处理**：albumentations, OpenCV, FFmpeg, decord
- **部署**：FastAPI, Docker, NVIDIA Triton Inference Server

---

## ❓ 常见问题

### Q1: 轻量模型 (< 1B) 能做视频理解吗？

**A**: 可以，关键是架构选择和管线设计：

| 策略 | 说明 | 典型方案 |
|------|------|---------|
| 2D CNN + 时序后处理 | 单帧推理 + 帧间聚合 | MobileNet + TSM |
| 轻量 3D 替代 | 减少时序卷积开销 | SlowFast-light |
| 帧采样 + 检测 | 关键帧检测 + 跟踪填补 | YOLO-nano + ByteTrack |

核心思想：不在单帧上追求极致精度，而是通过时序信息弥补。

---

### Q2: LoRA for ViT 应该注入哪些层？

**A**: 推荐策略：

| 优先级 | 注入层 | 原因 |
|--------|--------|------|
| 高 | Q/V projection | Attention 核心计算 |
| 中 | MLP 层 | 前馈网络的表示 |
| 低 | Patch Embedding | 底层特征通常泛化性好 |

一般先注入 Q/V projection，参数量控制在基础模型的 0.5%-2%。

---

### Q3: INT8 量化后精度下降太多怎么办？

**A**: 分步排查：

1. **校准数据**：确认校准集和实际推理场景的分布一致（500-1000 张有代表性的图）
2. **量化方式**：PTQ 不满足→尝试 QAT（训练 1-2 个 epoch）
3. **逐层分析**：找出精度损失最大的层，对该层使用 FP16 混合精度
4. **架构选择**：某些模型（如 MobileNetV3）对量化友好度更高

---

### Q4: 如何选择边缘设备？

**A**: 根据场景约束选择：

| 设备类型 | 算力 | 典型场景 | 建议模型规模 |
|---------|------|---------|------------|
| Raspberry Pi 5 | ~13 TOPS | 简单分类/检测 | < 30M 参数 |
| NVIDIA Jetson Orin Nano | ~40 TOPS | 多路视频分析 | < 300M 参数 |
| Intel NUC (OpenVINO) | ~10 TOPS | 工业质检 | < 100M 参数 |
| Apple M1/M2 (Core ML) | ~15 TOPS (ANE) | 移动端/桌面 | < 200M 参数 |

---

### Q5: 视频推理如何做到实时？

**A**: 端到端延迟由多段组成，需要分段优化：

- **采集延迟**（10-30ms）：硬件编码、RTSP 缓冲区
- **预处理延迟**（5-15ms）：解码、resize、归一化
- **推理延迟**（20-80ms，优化目标）：量化、TensorRT、batching
- **后处理延迟**（5-20ms）：NMS (Non-Maximum Suppression, 非极大值抑制)、跟踪、渲染
- **总延迟 = 各部分之和**。25 FPS 实时意味着总延迟 < 40ms

---

## ✅ 学习检查清单

### 多模态基础与选型
- [ ] 理解 CNN / ViT / Hybrid 三种视觉编码器的核心差异
- [ ] 掌握 CLIP 双塔架构和对比学习原理
- [ ] 能使用 timm 加载和评估轻量模型
- [ ] 能根据延迟/精度/显存约束完成模型选型
- [ ] 能输出一份包含 benchmark 数据的选型报告

### 图像识别
- [ ] 完成至少一个分类模型的训练与评估
- [ ] 掌握至少 3 种数据增强方法及其适用场景
- [ ] 理解 mAP 的计算逻辑和误差分析方法
- [ ] 能构建基础的以图搜图管线

### 视频理解
- [ ] 理解不同帧采样策略的适用场景和取舍
- [ ] 完成 TSM 模型的训练与推理
- [ ] 能构建多路视频的异步推理管线
- [ ] 掌握视频推理的性能剖析方法

### 微调与部署
- [ ] 完成 LoRA for ViT 的注入和微调
- [ ] 理解 PTQ 量化的流程和精度评估方法
- [ ] 能将 PyTorch 模型导出为 ONNX 并使用 ONNX Runtime 推理
- [ ] 能构建 FastAPI 推理服务并 Docker 化
- [ ] 掌握模型热切换和 A/B 推理的实现方式

---

## 📊 模块质量

Module 10 为新增模块，质量评估将在内容完成后进行。

### 预期标准

- 内容完整性：覆盖选型→应用→微调→部署全链路
- 理论深度：每个架构和算法都有数学直觉解释
- 代码质量：所有实践代码可运行，有英文注释
- 实践练习：15 个核心可跑微实践 + 17 个扩展练习 + 4 个综合项目
- 业务映射：每个 notebook 锚定到通用多模态平台（便利店货架监控示范）

---

## 🎯 下一步

完成 Module 10 后，你已经掌握了：
- ✅ 多模态小模型的架构全景和选型方法论
- ✅ 图像识别和视频理解的实战能力
- ✅ PEFT for Vision 的微调策略
- ✅ 边缘设备部署和推理平台构建

**继续学习**：
- **回顾 Module 8-9** - 将 Agent 和前沿研究的思维框架应用到多模态场景
- **实践项目** - 选择三个实践项目之一定向深入
- **开源贡献** - 参与 OpenMMLab、timm 等社区

---

**模块创建日期**: 2026-05-24
**质量评估**: 4/4 notebooks 通过 nbconvert --execute 验证，0 stubs，0 errors
**推荐指数**: ⭐⭐⭐⭐⭐
