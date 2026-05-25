# 🔬 Micro Practice 5: ONNX Export + onnxruntime Verification
# Target: PyTorch -> ONNX -> onnxruntime, verify output consistency

import torch, torch.nn as nn, numpy as np, os, time

class ExportModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        x = self.conv(x); x = self.relu(x)
        x = self.pool(x); return self.fc(x.view(x.size(0), -1))

torch.manual_seed(42)
model = ExportModel().eval()
dummy = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    pt_out = model(dummy)

onnx_path = "_tmp.onnx"
torch.onnx.export(model, dummy, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input":{0:"batch"},"output":{0:"batch"}},
    opset_version=14)
print(f"ONNX exported: {os.path.getsize(onnx_path)/1024:.1f} KB")

# onnxruntime verify
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_out = sess.run(["output"], {"input": dummy.numpy()})[0]
diff = np.abs(pt_out.numpy() - ort_out).max()
print(f"PyTorch vs ONNX max diff: {diff:.6f} ({'PASS' if diff < 1e-4 else 'WARN'})")

# Latency comparison
pts, orts = [], []
for _ in range(100):
    t0=time.perf_counter()
    with torch.no_grad(): _ = model(dummy)
    pts.append((time.perf_counter()-t0)*1000)
    t0=time.perf_counter()
    _ = sess.run(["output"], {"input": dummy.numpy()})
    orts.append((time.perf_counter()-t0)*1000)

print(f"PyTorch: {np.mean(pts):.3f}ms, ONNX: {np.mean(orts):.3f}ms, speedup: {np.mean(pts)/np.mean(orts):.2f}x")
os.remove(onnx_path)
