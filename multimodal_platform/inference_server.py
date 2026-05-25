"""FastAPI inference server for the multimodal platform."""
import torch
import torch.nn as nn
import os
import io
import json
import time
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse


# Model definition (must match training architecture)
class VisionModel(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


app = FastAPI(
    title="Multimodal Platform Inference API",
    description="Lightweight multimodal inference for convenience store shelf monitoring",
    version="1.0.0",
)

CLASS_NAMES = ["beverage", "snack", "noodle", "dairy", "household"]

# Model loading
MODEL_PATH = os.environ.get("MODEL_PATH", "./artifacts/vit_tiny_base.pt")
model: VisionModel = None
metrics = {"total_requests": 0, "total_errors": 0, "latencies": []}


def get_model() -> VisionModel:
    global model
    if model is None:
        model = VisionModel(n_classes=5)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
        model.eval()
    return model


@app.on_event("startup")
async def startup():
    get_model()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "VisionModel-5class",
        "backend": "PyTorch CPU",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
async def get_metrics():
    import numpy as np
    lats = metrics["latencies"]
    return {
        "total_requests": metrics["total_requests"],
        "total_errors": metrics["total_errors"],
        "p50_ms": round(float(np.percentile(lats, 50)), 2) if lats else 0,
        "p95_ms": round(float(np.percentile(lats, 95)), 2) if lats else 0,
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    try:
        from torchvision import transforms as T

        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        tensor = T.ToTensor()(img).unsqueeze(0)
        tensor = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(tensor)

        m = get_model()
        with torch.inference_mode():
            logits = m(tensor)
            probs = logits.softmax(1)[0]
            pred = probs.argmax().item()

        elapsed = (time.perf_counter() - t0) * 1000
        metrics["total_requests"] += 1
        metrics["latencies"].append(elapsed)

        return JSONResponse({
            "class_id": pred,
            "class_name": CLASS_NAMES[pred],
            "confidence": round(probs[pred].item(), 4),
            "latency_ms": round(elapsed, 2),
            "all_probs": {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))},
        })
    except Exception as e:
        metrics["total_errors"] += 1
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
