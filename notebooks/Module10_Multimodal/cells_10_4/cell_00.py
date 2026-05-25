# 🔬 Micro Practice 1: LoRA for ViT — self-implemented + peft comparison
# Demonstrate LoRALinear and inject into ViT attention

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Self-implemented LoRA linear layer for teaching clarity."""
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.register_buffer("weight", torch.zeros(out_features, in_features))
        self.register_buffer("bias", None)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def load_weight(self, weight, bias=None):
        self.weight.copy_(weight.data)
        if bias is not None:
            self.register_buffer("bias", bias.data.clone())

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        if self.merged:
            return base
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + lora_out * self.scaling

    def merge(self):
        if not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        if self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

print("LoRALinear defined. Key design:")
print("  1. weight stored as buffer (frozen), lora_A/lora_B as Parameters (trainable)")
print("  2. scaling = alpha/rank controls LoRA contribution strength")
print("  3. merge/unmerge enables zero-overhead inference")

# Inspect ViT-Tiny architecture
import timm
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"\nViT-Tiny total params: {n_params:.2f}M")
block = model.blocks[0]
print(f"Attention qkv weight shape: {block.attn.qkv.weight.shape}")
print("qkv is a merged projection (3*dim, dim) containing Q/K/V")
