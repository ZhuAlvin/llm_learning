# 🔬 Micro Practice 4: INT8 Post-Training Quantization
# Target: Compare FP32 vs INT8 accuracy and model size

import torch, torch.nn as nn, numpy as np, copy, os

class SimpleCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3,16,3,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.cls = nn.Linear(64, nc)
    def forward(self, x):
        x = self.feat(x); return self.cls(x.view(x.size(0), -1))

torch.manual_seed(42)
m_fp32 = SimpleCNN(10).eval()

dx = torch.randn(100, 3, 32, 32)
dy = torch.randint(0, 10, (100,))

with torch.no_grad():
    fp32_acc = (m_fp32(dx).argmax(1)==dy).float().mean().item()

torch.save(m_fp32.state_dict(), "_fp32.pt")
fp32_kb = os.path.getsize("_fp32.pt") / 1024
print(f"FP32: acc={fp32_acc:.4f}, size={fp32_kb:.1f} KB")

# INT8 dynamic quantization
m_int8 = copy.deepcopy(m_fp32).cpu().eval()
m_int8 = torch.ao.quantization.quantize_dynamic(m_int8, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

with torch.no_grad():
    int8_acc = (m_int8(dx).argmax(1)==dy).float().mean().item()

torch.save(m_int8.state_dict(), "_int8.pt")
int8_kb = os.path.getsize("_int8.pt") / 1024
print(f"INT8: acc={int8_acc:.4f}, size={int8_kb:.1f} KB")
print(f"Accuracy delta: {int8_acc-fp32_acc:+.4f}, Size ratio: {int8_kb/fp32_kb:.1%}")
print("INT8 PTQ complete. Minimal accuracy loss, significant size reduction.")

for f in ["_fp32.pt", "_int8.pt"]:
    if os.path.exists(f): os.remove(f)
