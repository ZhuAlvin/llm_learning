"""ModelRegistry: Manage model versions, LoRA branches, and deployment artifacts."""
import torch
import os
import json
from datetime import datetime
from typing import Dict, Optional


class ModelRegistry:
    """Central registry for base models, LoRA branches, quantized models, and ONNX artifacts."""

    def __init__(self, registry_dir: str = "./model_registry"):
        self.dir = registry_dir
        for sub in ["base", "lora", "quantized", "onnx"]:
            os.makedirs(os.path.join(registry_dir, sub), exist_ok=True)
        self.idx_path = os.path.join(registry_dir, "index.json")
        self.idx = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.idx_path):
            with open(self.idx_path) as f:
                return json.load(f)
        return {"models": {}, "loras": {}, "updated": None}

    def _save(self):
        self.idx["updated"] = datetime.now().isoformat()
        with open(self.idx_path, "w") as f:
            json.dump(self.idx, f, indent=2, ensure_ascii=False)

    def register_model(self, name: str, state_dict: Dict, metadata: Optional[Dict] = None) -> str:
        path = os.path.join(self.dir, "base", f"{name}.pt")
        torch.save(state_dict, path)
        self.idx["models"][name] = {
            "path": path,
            "size_kb": os.path.getsize(path) // 1024,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
        }
        self._save()
        return path

    def register_lora(self, name: str, base_model_name: str, state_dict: Dict, metadata: Optional[Dict] = None) -> str:
        path = os.path.join(self.dir, "lora", f"{name}.pt")
        torch.save(state_dict, path)
        self.idx["loras"][name] = {
            "path": path,
            "base_model": base_model_name,
            "size_kb": os.path.getsize(path) // 1024,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
        }
        self._save()
        return path

    def get_model(self, name: str) -> Optional[Dict]:
        return self.idx["models"].get(name)

    def get_lora(self, name: str) -> Optional[Dict]:
        return self.idx["loras"].get(name)

    def list_models(self):
        return list(self.idx["models"].keys())

    def list_loras(self):
        return list(self.idx["loras"].keys())

    def summary(self):
        return {
            "registry_dir": self.dir,
            "num_models": len(self.idx["models"]),
            "num_loras": len(self.idx["loras"]),
            "models": self.list_models(),
            "loras": self.list_loras(),
            "last_updated": self.idx.get("updated"),
        }
