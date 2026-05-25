# Capstone Part 2: INT8 Quantization + ONNX Export
import torch, torch.nn as nn, numpy as np, os, copy, json, time

AD = "../../multimodal_platform/artifacts"


class VisionModel(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, n)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


torch.manual_seed(42)
m_fp32 = VisionModel(5).eval()

# Simulate LoRA fine-tuning
opt = torch.optim.AdamW(m_fp32.parameters(), lr=1e-3)
for _ in range(5):
    x, y = torch.randn(4, 3, 224, 224), torch.randint(0, 5, (4,))
    opt.zero_grad()
    l = nn.CrossEntropyLoss()(m_fp32(x), y)
    l.backward()
    opt.step()

# INT8 quantization
m_int8 = copy.deepcopy(m_fp32).cpu().eval()
m_int8 = torch.ao.quantization.quantize_dynamic(
    m_int8, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
torch.save(m_int8.state_dict(), os.path.join(AD, "lora_store_a_int8.pt"))
print("Saved: lora_store_a_int8.pt")

# ONNX export
dummy = torch.randn(1, 3, 224, 224)
onnx_path = os.path.join(AD, "lora_store_a.onnx")
torch.onnx.export(
    m_fp32,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)

# Verify ONNX
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
with torch.no_grad():
    pt_o = m_fp32(dummy)
ort_o = sess.run(["output"], {"input": dummy.numpy()})[0]
diff = np.abs(pt_o.numpy() - ort_o).max()
print(f"Saved: lora_store_a.onnx ({os.path.getsize(onnx_path)/1024:.1f} KB)")
print(f"ONNX verify: max diff = {diff:.6f} {'PASS' if diff < 1e-4 else 'WARN'}")

# Benchmark
pts = []
orts_ = []
for _ in range(50):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = m_fp32(dummy)
    pts.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    _ = sess.run(["output"], {"input": dummy.numpy()})
    orts_.append((time.perf_counter() - t0) * 1000)

report = {
    "fp32_avg_ms": round(np.mean(pts), 3),
    "fp32_p95_ms": round(np.percentile(pts, 95), 3),
    "onnx_avg_ms": round(np.mean(orts_), 3),
    "onnx_p95_ms": round(np.percentile(orts_, 95), 3),
    "onnx_speedup": round(np.mean(pts) / np.mean(orts_), 2),
    "lora_kb": os.path.getsize(os.path.join(AD, "lora_store_a.pt")) // 1024,
    "int8_kb": os.path.getsize(os.path.join(AD, "lora_store_a_int8.pt")) // 1024,
    "onnx_kb": os.path.getsize(onnx_path) // 1024,
}

with open(os.path.join(AD, "benchmark_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("\nBenchmark Report:")
for k, v in report.items():
    print(f"  {k}: {v}")

print(f"\n{AD}/ contents:")
for fn in sorted(os.listdir(AD)):
    print(f"  {fn} ({os.path.getsize(os.path.join(AD, fn))/1024:.1f} KB)")
