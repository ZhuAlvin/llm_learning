# Capstone Part 1: End-to-end pipeline -- Train LoRA (self-implemented)
import torch, torch.nn as nn, torch.optim as optim, timm, numpy as np, os
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)
AD = "../../multimodal_platform/artifacts"
os.makedirs(AD, exist_ok=True)

# Dataset: 5 classes x 10 shots (synthetic)
n_cls, n_shot = 5, 10
tx = torch.randn(n_cls * n_shot, 3, 224, 224)
ty = torch.repeat_interleave(torch.arange(n_cls), n_shot)
vx = torch.randn(n_cls * 20, 3, 224, 224)
vy = torch.repeat_interleave(torch.arange(n_cls), 20)
tl = DataLoader(TensorDataset(tx, ty), batch_size=4, shuffle=True)
vl = DataLoader(TensorDataset(vx, vy), batch_size=32)

# Base model
base = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=n_cls).eval()
with torch.no_grad():
    c = sum((base(x).argmax(1) == y).sum().item() for x, y in vl)
print(f"Baseline acc: {c}/{len(vx)} = {c/len(vx):.4f}")

torch.save(base.state_dict(), os.path.join(AD, "vit_tiny_base.pt"))

# Self-implemented LoRA injection
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=8.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        in_f, out_f = original_linear.in_features, original_linear.out_features
        self.register_buffer('weight', original_linear.weight.data.clone())
        self.bias = original_linear.bias.data.clone() if original_linear.bias is not None else None
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x):
        base_out = nn.functional.linear(x, self.weight, self.bias)
        return base_out + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

# Inject into qkv
lora_params = []
for name, module in base.named_modules():
    if isinstance(module, nn.Linear) and 'qkv' in name:
        lora_layer = LoRALinear(module, rank=4, alpha=8.0)
        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        parent_path, attr = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = base.get_submodule(parent_path) if parent_path else base
        setattr(parent, attr, lora_layer)

for p in base.parameters():
    p.requires_grad = False
for p in lora_params:
    p.requires_grad = True

print(f"LoRA trainable: {sum(p.numel() for p in base.parameters() if p.requires_grad):,}")

# Train
base.train()
opt = optim.AdamW(filter(lambda p: p.requires_grad, base.parameters()), lr=5e-4)
crit = nn.CrossEntropyLoss()
for ep in range(3):
    ls = 0
    for x, y in tl:
        opt.zero_grad()
        l = crit(base(x), y)
        l.backward()
        opt.step()
        ls += l.item()
    print(f"Epoch {ep+1}/3 Loss: {ls/len(tl):.4f}")

base.eval()
with torch.no_grad():
    c = sum((base(x).argmax(1) == y).sum().item() for x, y in vl)
print(f"LoRA acc: {c}/{len(vx)} = {c/len(vx):.4f}")

# Save LoRA weights (just the lora_A, lora_B params)
lora_sd = {}
for name, param in base.named_parameters():
    if 'lora_' in name:
        lora_sd[name] = param.data.clone()
torch.save(lora_sd, os.path.join(AD, "lora_store_a.pt"))
print(f"Saved: lora_store_a.pt ({os.path.getsize(os.path.join(AD, 'lora_store_a.pt'))/1024:.1f} KB, typically <1MB)")
