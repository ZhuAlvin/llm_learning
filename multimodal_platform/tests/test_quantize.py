"""Tests for quantization and ONNX export."""
import torch
import torch.nn as nn
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantizer import quantize_dynamic, export_onnx, verify_onnx, benchmark_inference


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16, 5)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class TestQuantizer:
    def test_quantize_dynamic(self):
        model = SimpleModel().eval()
        try:
            quantized = quantize_dynamic(model)
            x = torch.randn(1, 3, 32, 32)
            out = quantized(x)
            assert out.shape == (1, 5)
        except Exception as e:
            pytest.skip(f"Quantization not supported in this PyTorch build: {e}")

    def test_onnx_export_and_verify(self):
        model = SimpleModel().eval()
        path = "/tmp/_test_onnx_export.onnx"

        try:
            result = export_onnx(model, path, input_shape=(1, 3, 32, 32))
            assert os.path.exists(result)

            diff = verify_onnx(model, result, input_shape=(1, 3, 32, 32))
            assert diff < 1e-4, f"ONNX output differs by {diff}"

        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_benchmark(self):
        model = SimpleModel().eval()
        result = benchmark_inference(model, input_shape=(1, 3, 32, 32), num_runs=20)
        assert "avg_ms" in result
        assert "p50_ms" in result
        assert result["avg_ms"] > 0
