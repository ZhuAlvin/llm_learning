"""Quantizer: INT8 PTQ and ONNX export utilities."""
import torch
import torch.nn as nn
import os
from typing import Optional, Tuple


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 post-training quantization.

    Args:
        model: FP32 PyTorch model in eval mode.

    Returns:
        INT8 quantized model.
    """
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple = (1, 3, 224, 224),
    opset_version: int = 14,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode.
        output_path: Path for the output .onnx file.
        input_shape: Dummy input tensor shape (batch, channels, height, width).
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset_version,
    )

    return output_path


def verify_onnx(model: nn.Module, onnx_path: str, input_shape: Tuple = (1, 3, 224, 224)) -> float:
    """Verify ONNX model output matches PyTorch model.

    Returns the maximum absolute difference between PyTorch and ONNX outputs.
    """
    import onnxruntime as ort
    import numpy as np

    model.eval()
    dummy = torch.randn(*input_shape)

    with torch.no_grad():
        pt_out = model(dummy)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(["output"], {"input": dummy.numpy()})[0]

    return float(np.abs(pt_out.numpy() - ort_out).max())


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple = (1, 3, 224, 224),
    num_runs: int = 100,
    use_onnx: bool = False,
    onnx_path: Optional[str] = None,
) -> dict:
    """Benchmark model inference latency.

    Returns dict with avg_ms, p50_ms, p95_ms, p99_ms.
    """
    import time
    import numpy as np
    import onnxruntime as ort

    times = []

    if use_onnx and onnx_path:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(*input_shape).astype(np.float32)
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = session.run(["output"], {"input": dummy})
            times.append((time.perf_counter() - t0) * 1000)
    else:
        model.eval()
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy)
                times.append((time.perf_counter() - t0) * 1000)

    return {
        "avg_ms": round(np.mean(times), 3),
        "p50_ms": round(np.percentile(times, 50), 3),
        "p95_ms": round(np.percentile(times, 95), 3),
        "p99_ms": round(np.percentile(times, 99), 3),
        "num_runs": num_runs,
    }
