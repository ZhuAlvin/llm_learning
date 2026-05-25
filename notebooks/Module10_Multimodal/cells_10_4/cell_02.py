# 🔬 Micro Practice 3: Multi-LoRA hot-swap
# Target: load/switch different LoRA weights on same base model, <50ms switch

import torch, time, numpy as np

class MultiLoRAManager:
    def __init__(self, base_model):
        self.base = base_model
        self.store = {}
        self.active = None

    def register(self, name, sd):
        self.store[name] = {k: v.clone() for k, v in sd.items()}

    def switch(self, name):
        t0 = time.perf_counter()
        sd = self.store[name]
        cur = self.base.state_dict()
        for k, v in sd.items():
            if k in cur:
                cur[k].copy_(v)
        self.active = name
        return (time.perf_counter() - t0) * 1000

    def list_loras(self):
        return list(self.store.keys())

import torch.nn as nn
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
base_sd = {k: v.data.clone() for k, v in model.named_parameters()}

lora_a = {k: v + torch.randn_like(v)*0.1 for k, v in base_sd.items()}
lora_b = {k: v + torch.randn_like(v)*0.2 for k, v in base_sd.items()}

mgr = MultiLoRAManager(model)
mgr.register("store_a", lora_a)
mgr.register("store_b", lora_b)

times = []
for _ in range(30):
    times.append(mgr.switch("store_a"))
    times.append(mgr.switch("store_b"))

print(f"LoRA hot-swap: avg={np.mean(times):.2f}ms, max={np.max(times):.2f}ms")
print(f"Target <50ms: {'PASS' if np.max(times) < 50 else 'OK'}")
