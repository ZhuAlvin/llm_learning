# Engineering: InferenceServer class
import torch, time, numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 8
    timeout_ms: int = 5000


@dataclass
class ServerMetrics:
    total: int = 0
    errors: int = 0
    latencies: List[float] = field(default_factory=list)

    def record(self, lat, is_err=False):
        self.total += 1
        self.latencies.append(lat)
        if is_err:
            self.errors += 1

    def p50(self):
        return float(np.percentile(self.latencies, 50)) if self.latencies else 0

    def p95(self):
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0


class InferenceServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.models = {}
        self.active = None
        self.metrics = ServerMetrics()

    def load_model(self, name, model):
        self.models[name] = model
        if self.active is None:
            self.active = name
        print(f"Model {name!r} loaded.")

    def switch_model(self, name):
        if name not in self.models:
            raise KeyError(name)
        self.active = name
        print(f"Switched to {name!r}.")

    @torch.inference_mode()
    def infer(self, model, tensor):
        t0 = time.perf_counter()
        out = model(tensor)
        return out, (time.perf_counter() - t0) * 1000

    def get_metrics(self):
        return {
            "total": self.metrics.total,
            "errors": self.metrics.errors,
            "p50_ms": self.metrics.p50(),
            "p95_ms": self.metrics.p95(),
            "active": self.active,
        }


# Demo
import torch.nn as nn
cfg = ServerConfig()
server = InferenceServer(cfg)
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).eval()
server.load_model("vit_tiny_store_a", model)
for i in range(50):
    _, lat = server.infer(model, torch.randn(1, 128))
    server.metrics.record(lat)

m = server.get_metrics()
print(f"InferenceServer: P50={m['p50_ms']:.2f}ms, P95={m['p95_ms']:.2f}ms, requests={m['total']}")
print("InferenceServer class complete.")
