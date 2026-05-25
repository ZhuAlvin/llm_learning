"""LoRA Manager: Load, switch, and manage multiple LoRA branches at runtime."""
import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer for injection into pretrained models."""

    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        in_f, out_f = original_linear.in_features, original_linear.out_features

        self.register_buffer("weight", original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.clone())
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        return base + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class MultiLoRAManager:
    """Manage multiple LoRA branches with runtime hot-swap capability."""

    def __init__(self, base_model: nn.Module):
        self.base = base_model
        self.store: Dict[str, Dict[str, torch.Tensor]] = {}
        self.active: Optional[str] = None

    def register(self, name: str, state_dict: Dict[str, torch.Tensor]):
        """Register a LoRA branch state dict."""
        self.store[name] = {k: v.clone() for k, v in state_dict.items()}

    def switch(self, name: str) -> float:
        """Hot-swap to the specified LoRA branch. Returns switching latency in ms."""
        if name not in self.store:
            raise KeyError(f"LoRA '{name}' not found. Available: {list(self.store.keys())}")
        t0 = time.perf_counter()
        sd = self.store[name]
        cur = self.base.state_dict()
        for k, v in sd.items():
            if k in cur:
                cur[k].copy_(v)
        self.active = name
        return (time.perf_counter() - t0) * 1000

    def list_loras(self) -> List[str]:
        return list(self.store.keys())

    def get_active(self) -> Optional[str]:
        return self.active


def inject_lora_to_vit(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    target_substrings: tuple = ("qkv",),
) -> List[nn.Parameter]:
    """Inject LoRA layers into a ViT model's attention projections.

    Returns the list of trainable LoRA parameters.
    """
    lora_params = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(ts in name for ts in target_substrings):
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

            parent_path, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, attr, lora_layer)

    # Freeze original params, unfreeze LoRA params
    for p in model.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True

    return lora_params
