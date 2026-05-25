# Engineering: ModelRegistry class
import torch, os, json
from datetime import datetime

class ModelRegistry:
    """Model version, LoRA branch, and deployment artifact registry."""

    def __init__(self, registry_dir="./model_registry"):
        self.dir = registry_dir
        for sub in ["base", "lora", "quantized", "onnx"]:
            os.makedirs(os.path.join(registry_dir, sub), exist_ok=True)
        self.ip = os.path.join(registry_dir, "index.json")
        if os.path.exists(self.ip):
            self.idx = json.load(open(self.ip))
        else:
            self.idx = {"models": {}, "loras": {}}

    def _save(self):
        self.idx["updated"] = datetime.now().isoformat()
        json.dump(self.idx, open(self.ip, "w"), indent=2, ensure_ascii=False)

    def register_model(self, name, sd, meta=None):
        p = os.path.join(self.dir, "base", f"{name}.pt")
        torch.save(sd, p)
        self.idx["models"][name] = {
            "path": p,
            "size_kb": os.path.getsize(p) // 1024,
            "meta": meta or {},
            "registered": datetime.now().isoformat(),
        }
        self._save()
        return p

    def register_lora(self, name, base_name, sd, meta=None):
        p = os.path.join(self.dir, "lora", f"{name}.pt")
        torch.save(sd, p)
        self.idx["loras"][name] = {
            "path": p,
            "base_model": base_name,
            "size_kb": os.path.getsize(p) // 1024,
            "meta": meta or {},
            "registered": datetime.now().isoformat(),
        }
        self._save()
        return p

    def summary(self):
        print(f"ModelRegistry: {self.dir}")
        print(f"  Models: {list(self.idx['models'].keys())}")
        print(f"  LoRAs:  {list(self.idx['loras'].keys())}")


# Demo
import torch.nn as nn
reg = ModelRegistry("./model_registry")
demo = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
reg.register_model("vit_tiny_base", demo.state_dict(), {"version": "1.0"})
lora_a = {k: v + torch.randn_like(v) * 0.05 for k, v in demo.state_dict().items()}
reg.register_lora("store_a", "vit_tiny_base", lora_a, {"store": "A", "sku": 200})
reg.summary()
print("ModelRegistry class complete.")
