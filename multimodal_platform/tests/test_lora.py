"""Tests for LoRA manager functionality."""
import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lora_manager import LoRALinear, MultiLoRAManager, inject_lora_to_vit


class TestLoRALinear:
    def test_forward_shape(self):
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 128)

    def test_forward_with_bias(self):
        linear = nn.Linear(64, 128, bias=True)
        lora = LoRALinear(linear, rank=4)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 128)

    def test_gradient_flow(self):
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        # lora_A and lora_B should require grad, weight should not
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad
        assert not lora.weight.requires_grad

        x = torch.randn(2, 64)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert lora.weight.grad is None


class TestMultiLoRAManager:
    def test_register_and_switch(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        mgr = MultiLoRAManager(model)

        sd_a = {k: v.clone() for k, v in model.state_dict().items()}
        sd_b = {k: v + torch.randn_like(v) * 0.1 for k, v in sd_a.items()}

        mgr.register("a", sd_a)
        mgr.register("b", sd_b)

        assert set(mgr.list_loras()) == {"a", "b"}

        lat = mgr.switch("a")
        assert lat < 100  # switching should be fast
        assert mgr.get_active() == "a"

        lat = mgr.switch("b")
        assert lat < 100
        assert mgr.get_active() == "b"

    def test_switch_unknown_raises(self):
        model = nn.Linear(10, 10)
        mgr = MultiLoRAManager(model)
        with pytest.raises(KeyError):
            mgr.switch("nonexistent")
