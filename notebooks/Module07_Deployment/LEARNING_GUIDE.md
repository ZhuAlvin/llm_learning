# Module 7: 部署与优化 - 学习指南

## 📋 文档质量检查报告

### ✅ 已完成内容

**Notebook 01_inference_optimization.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和推理优化理论
- ✅ 完整的理论讲解（量化、剪枝、知识蒸馏、ONNX）
- ✅ 10 个 Micro Practice 实践练习
- ✅ 可视化（性能对比、内存分析、优化效果）
- ✅ 完整实现（量化、剪枝、知识蒸馏、ONNX导出）
- ✅ 工程实践（硬件加速、综合优化流水线）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

**Notebook 02_model_serving.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和模型服务理论
- ✅ 完整的理论讲解（REST API、动态批处理、缓存、负载均衡）
- ✅ 8 个 Micro Practice 实践练习
- ✅ 可视化（性能监控、扩展测试、吞吐量分析）
- ✅ 完整实现（FastAPI服务、动态批处理、缓存系统）
- ✅ 工程实践（容器化、Kubernetes部署、CI/CD）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

**Notebook 03_production_best_practices.ipynb** - 完整且高质量
- ✅ 清晰的学习目标和MLOps理论
- ✅ 完整的理论讲解（MLOps、版本管理、A/B测试、监控、漂移检测）
- ✅ 6 个 Micro Practice 实践练习
- ✅ 可视化（监控面板、漂移检测、SLA监控）
- ✅ 完整实现（监控系统、A/B测试、事故响应）
- ✅ 工程实践（生产就绪检查清单、SLA/SLO/SLI）
- ✅ FAQ 和调试技巧
- ✅ 总结和思考题

### 📊 质量评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **内容完整性** | 10/10 | 所有notebook完整且高质量 |
| **理论深度** | 9/10 | 数学公式清晰，原理讲解透彻 |
| **代码质量** | 9/10 | 实现规范，注释详细 |
| **实践练习** | 10/10 | Micro Practice 设计优秀 |
| **可视化** | 9/10 | 架构图和可视化清晰 |
| **工程实践** | 9/10 | 包含完整的部署流程 |

**总体评分：9.4/10** - 优秀，内容完整且质量高

---

## 🎯 学习指南

### 学习路径

```
第1周：推理优化
  ├─ 理解推理性能指标
  ├─ 掌握量化技术
  ├─ 学习模型剪枝
  └─ 实现知识蒸馏

第2周：模型服务
  ├─ 理解REST API设计
  ├─ 掌握动态批处理
  ├─ 学习缓存和负载均衡
  └─ 实现高性能服务

第3周：生产最佳实践
  ├─ 理解MLOps
  ├─ 掌握版本管理
  ├─ 学习A/B测试
  └─ 实现监控系统

第4周：综合项目
  ├─ 构建完整部署流程
  ├─ 应用所有优化技术
  ├─ 监控和维护
  └─ 事故响应
```

### 前置知识检查

在开始学习前，确保你已经掌握：

- [ ] Python 编程基础
- [ ] PyTorch 基础（张量操作、自动微分）
- [ ] 线性代数（矩阵乘法、向量运算）
- [ ] 深度学习基础（反向传播、优化器）
- [ ] Transformer 架构（Module 3）
- [ ] 预训练语言模型（Module 4）
- [ ] 微调技术（Module 5）
- [ ] 高级训练技术（Module 6）

### 学习建议

#### 1. 理论学习（40%时间）

**必读材料**：
- 📄 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - 知识蒸馏原论文
- 📄 [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) - 剪枝原论文
- 📄 [Quantization and Training of Neural Networks](https://arxiv.org/abs/1712.05877) - 量化原论文
- 📖 [ONNX Runtime Documentation](https://onnxruntime.ai/docs/) - ONNX官方文档
- 📖 [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/) - TensorRT文档

**学习方法**：
1. 先理解推理优化的基本概念
2. 深入学习量化、剪枝、知识蒸馏的数学原理
3. 理解模型服务的架构设计
4. 学习MLOps的最佳实践

#### 2. 代码实践（50%时间）

**实践步骤**：

**Week 1: 推理优化**
```python
# 练习1：实现动态量化
def dynamic_quantization(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model

# 练习2：实现静态量化
def static_quantization(model, calibration_data):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    
    # 校准
    with torch.no_grad():
        for data in calibration_data:
            model_prepared(data)
    
    quantized_model = torch.quantization.convert(model_prepared)
    return quantized_model

# 练习3：实现幅度剪枝
def magnitude_pruning(model, sparsity=0.5):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 计算阈值
            threshold = torch.quantile(torch.abs(param), sparsity)
            # 应用掩码
            mask = torch.abs(param) > threshold
            param.data *= mask.float()

# 练习4：实现知识蒸馏
class DistillationLoss(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5):
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_output, teacher_output, target):
        # 软标签损失
        soft_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_output, target)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# 练习5：导出ONNX模型
def export_to_onnx(model, dummy_input, onnx_path):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
```

**Week 2: 模型服务**
```python
# 练习6：实现FastAPI服务
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    text: str

class Response(BaseModel):
    output: str

@app.post("/predict", response_model=Response)
async def predict(request: Request):
    try:
        output = model.generate(request.text)
        return Response(output=output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 练习7：实现动态批处理
class DynamicBatcher:
    def __init__(self, max_batch_size=32, timeout_ms=10):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.current_batch = []
        self.lock = threading.Lock()
    
    async def add_request(self, request):
        with self.lock:
            self.current_batch.append(request)
            if len(self.current_batch) >= self.max_batch_size:
                batch = self.current_batch
                self.current_batch = []
                return self.process_batch(batch)
        
        await asyncio.sleep(self.timeout_ms / 1000)
        with self.lock:
            if self.current_batch:
                batch = self.current_batch
                self.current_batch = []
                return self.process_batch(batch)

# 练习8：实现LRU缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_inference(text):
    return model.generate(text)

# 练习9：实现负载均衡
class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0
    
    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# 练习10：实现监控
from prometheus_client import Counter, Histogram, start_http_server

request_counter = Counter('requests_total', 'Total requests')
request_latency = Histogram('request_latency_seconds', 'Request latency')

@app.post("/predict")
async def predict(request: Request):
    with request_latency.time():
        output = model.generate(request.text)
        request_counter.inc()
        return Response(output=output)
```

**Week 3: 生产最佳实践**
```python
# 练习11：实现模型版本管理
import mlflow

def log_model(model, metrics):
    with mlflow.start_run():
        mlflow.log_params(model.config)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")

# 练习12：实现A/B测试
class ABTest:
    def __init__(self, model_a, model_b, traffic_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_ratio = traffic_ratio
    
    def predict(self, input):
        if random.random() < self.traffic_ratio:
            return self.model_a(input)
        else:
            return self.model_b(input)

# 练习13：实现漂移检测
from scipy import stats

def detect_drift(reference_data, current_data, threshold=0.05):
    # KS检验
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    if p_value < threshold:
        return True, f"Drift detected (p-value: {p_value:.4f})"
    return False, "No drift detected"

# 练习14：实现SLA监控
class SLAMonitor:
    def __init__(self, slo=0.99, window=60):
        self.slo = slo
        self.window = window
        self.requests = []
    
    def record_request(self, success, latency):
        timestamp = time.time()
        self.requests.append((timestamp, success, latency))
        
        # 清理过期请求
        self.requests = [
            r for r in self.requests
            if timestamp - r[0] < self.window
        ]
    
    def check_sla(self):
        if not self.requests:
            return True
        
        success_rate = sum(1 for r in self.requests if r[1]) / len(self.requests)
        avg_latency = sum(r[2] for r in self.requests) / len(self.requests)
        
        return success_rate >= self.slo, success_rate, avg_latency

# 练习15：实现事故响应
class IncidentManager:
    def __init__(self):
        self.incidents = []
    
    def create_incident(self, severity, description):
        incident = {
            'id': len(self.incidents) + 1,
            'severity': severity,
            'description': description,
            'status': 'open',
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        self.incidents.append(incident)
        return incident
    
    def update_incident(self, incident_id, status, update):
        for incident in self.incidents:
            if incident['id'] == incident_id:
                incident['status'] = status
                incident['updated_at'] = datetime.now()
                incident['updates'] = incident.get('updates', []) + [update]
                return incident
```

**Week 4: 完整项目**
```python
# 练习16：实现完整的部署流程
class DeploymentPipeline:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 推理优化
        self.optimized_model = self.optimize_model(model)
        
        # 模型服务
        self.server = self.create_server(self.optimized_model)
        
        # 监控系统
        self.monitor = self.create_monitor()
        
        # 事故管理
        self.incident_manager = IncidentManager()
    
    def optimize_model(self, model):
        # 量化
        model = static_quantization(model, self.config.calibration_data)
        
        # 剪枝
        magnitude_pruning(model, sparsity=self.config.sparsity)
        
        # 导出ONNX
        export_to_onnx(model, self.config.dummy_input, 'model.onnx')
        
        return model
    
    def create_server(self, model):
        app = FastAPI()
        
        @app.post("/predict")
        async def predict(request: Request):
            with self.monitor.request_latency.time():
                try:
                    output = model.generate(request.text)
                    self.monitor.request_counter.inc()
                    return Response(output=output)
                except Exception as e:
                    self.incident_manager.create_incident(
                        severity='high',
                        description=f"Prediction failed: {str(e)}"
                    )
                    raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    def create_monitor(self):
        request_counter = Counter('requests_total', 'Total requests')
        request_latency = Histogram('request_latency_seconds', 'Request latency')
        return type('Monitor', (), {
            'request_counter': request_counter,
            'request_latency': request_latency
        })()
```

#### 3. 项目实战（10%时间）

**推荐项目**：

1. **模型优化服务**（简单）
   - 实现量化和剪枝
   - 部署FastAPI服务
   - 目标：延迟 < 100ms

2. **高性能模型服务**（中等）
   - 实现动态批处理和缓存
   - 部署Kubernetes集群
   - 目标：吞吐量 > 1000 req/s

3. **生产级MLOps系统**（困难）
   - 实现完整的MLOps流程
   - 部署监控和告警系统
   - 目标：SLA > 99.9%

### 常见问题解答

#### Q1: 量化和剪枝如何选择？

**A:** 选择指南：
- **量化**：需要快速部署，精度要求高，实现简单
- **剪枝**：需要大幅减少模型大小，可接受轻微精度损失
- **组合使用**：追求极致性能和压缩比

#### Q2: 知识蒸馏的原理是什么？

**A:** 知识蒸馏：
- **教师模型**：大模型，提供软标签
- **学生模型**：小模型，学习教师知识
- **温度参数**：控制软标签的平滑程度
- **损失函数**：软标签损失 + 硬标签损失

#### Q3: 动态批处理如何实现？

**A:** 动态批处理：
- **原理**：累积多个请求，一起处理
- **优势**：提升吞吐量5-10x
- **实现**：使用队列和定时器
- **权衡**：增加延迟

#### Q4: A/B测试如何设计？

**A:** A/B测试设计：
- **流量分配**：随机分配到不同模型
- **指标定义**：明确评估指标
- **统计显著性**：使用t检验或卡方检验
- **渐进发布**：逐步增加流量

#### Q5: 如何检测模型漂移？

**A:** 漂移检测方法：
- **数据漂移**：KS检验、PSI
- **概念漂移**：监控模型性能指标
- **处理方法**：重新训练、数据重采样

### 调试技巧

#### 1. 推理性能差

```python
# 性能分析
import time

def profile_inference(model, input_data):
    start = time.time()
    output = model(input_data)
    end = time.time()
    print(f"Inference time: {end - start:.4f}s")

# 使用torch.profiler
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    output = model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

#### 2. 量化精度损失大

```python
# 检查量化前后精度
def compare_quantization(model, quantized_model, test_data):
    original_accuracy = evaluate(model, test_data)
    quantized_accuracy = evaluate(quantized_model, test_data)
    
    print(f"Original accuracy: {original_accuracy:.4f}")
    print(f"Quantized accuracy: {quantized_accuracy:.4f}")
    print(f"Accuracy drop: {original_accuracy - quantized_accuracy:.4f}")
    
    # 如果精度损失大，尝试QAT
    if original_accuracy - quantized_accuracy > 0.01:
        print("Consider using Quantization-Aware Training (QAT)")
```

#### 3. 服务响应慢

```python
# 监控各阶段耗时
import time

def monitor_request(request):
    stages = {}
    
    # 预处理
    start = time.time()
    processed_input = preprocess(request)
    stages['preprocessing'] = time.time() - start
    
    # 推理
    start = time.time()
    output = model(processed_input)
    stages['inference'] = time.time() - start
    
    # 后处理
    start = time.time()
    result = postprocess(output)
    stages['postprocessing'] = time.time() - start
    
    print(f"Stages: {stages}")
    return result
```

#### 4. 监控告警误报

```python
# 实现告警过滤
class AlertFilter:
    def __init__(self, cooldown=60):
        self.cooldown = cooldown
        self.last_alerts = {}
    
    def should_alert(self, alert_type):
        now = time.time()
        last_alert = self.last_alerts.get(alert_type, 0)
        
        if now - last_alert > self.cooldown:
            self.last_alerts[alert_type] = now
            return True
        return False
```

### 性能优化

#### 1. 推理优化

```python
# 使用ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
output = session.run(None, {'input': input_data})

# 使用TensorRT
import tensorrt as trt

# 使用混合精度推理
with torch.cuda.amp.autocast():
    output = model(input_data)
```

#### 2. 服务优化

```python
# 使用异步I/O
import asyncio

async def async_predict(request):
    output = await model.generate_async(request.text)
    return output

# 使用连接池
from aiohttp import ClientSession

session = ClientSession()
async def fetch_data(url):
    async with session.get(url) as response:
        return await response.json()
```

### 扩展阅读

#### 进阶主题

1. **高级优化技术**
   - QAT: 量化感知训练
   - (IA)³: 可插放适配器
   - Neural Architecture Search: 架构搜索

2. **高级部署技术**
   - Serverless Deployment: 无服务器部署
   - Edge Computing: 边缘计算
   - Federated Learning: 联邦学习

3. **高级MLOps**
   - Continuous Training: 持续训练
   - AutoML: 自动化机器学习
   - Model Governance: 模型治理

#### 推荐资源

**视频课程**：
- NVIDIA Deep Learning Institute
- Hugging Face Deployment Course
- Google Cloud MLOps Fundamentals

**代码库**：
- ONNX Runtime
- TensorRT
- FastAPI
- MLflow
- Prometheus

**论文列表**：
- Distilling the Knowledge in a Neural Network (2015)
- The Lottery Ticket Hypothesis (2018)
- Quantization and Training of Neural Networks (2018)
- ONNX: Open Neural Network Exchange (2017)

### 评估标准

完成本模块后，你应该能够：

- [ ] 理解推理优化的核心指标
- [ ] 实现量化和剪枝
- [ ] 理解知识蒸馏的原理
- [ ] 导出和优化ONNX模型
- [ ] 构建高性能模型服务
- [ ] 实现动态批处理和缓存
- [ ] 理解MLOps的概念
- [ ] 实现模型版本管理
- [ ] 设计和实现A/B测试
- [ ] 实现监控和漂移检测
- [ ] 管理生产事故
- [ ] 阅读和理解部署相关论文

### 下一步

完成 Module 7 后，建议：

1. **巩固基础**：重新实现一遍核心组件
2. **项目实践**：完成至少一个完整项目
3. **阅读论文**：深入理解优化和部署技术
4. **学习 Module 8**：实际应用

---

## 📝 学习检查清单

### Week 1: 推理优化
- [ ] 理解推理性能指标
- [ ] 实现量化
- [ ] 实现剪枝
- [ ] 实现知识蒸馏

### Week 2: 模型服务
- [ ] 理解REST API
- [ ] 实现动态批处理
- [ ] 实现缓存
- [ ] 实现负载均衡

### Week 3: 生产最佳实践
- [ ] 理解MLOps
- [ ] 实现版本管理
- [ ] 实现A/B测试
- [ ] 实现监控系统

### Week 4: 实战
- [ ] 构建完整部署流程
- [ ] 应用所有优化技术
- [ ] 监控和维护
- [ ] 完成项目

---

**祝学习顺利！** 🚀

如有问题，请参考：
- 📖 Notebook 中的 FAQ 部分
- 💬 课程讨论区
- 🔍 Stack Overflow
- 📧 联系助教