"""Integration tests for the FastAPI inference server."""
import pytest
import subprocess
import sys
import time
import os
import io
import json
import numpy as np
from PIL import Image


@pytest.fixture(scope="module")
def server():
    """Start test server in a subprocess."""
    server_path = os.path.join(os.path.dirname(__file__), "..", "inference_server.py")
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)

    # Verify it started
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
        assert resp.status == 200
    except Exception:
        proc.terminate()
        proc.wait()
        pytest.fail("Server failed to start")

    yield proc

    proc.terminate()
    proc.wait()


class TestInferenceAPI:
    def test_health(self):
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
        data = json.loads(resp.read())
        assert data["status"] == "healthy"

    def test_metrics(self):
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5)
        data = json.loads(resp.read())
        assert "total_requests" in data

    def test_classify(self):
        import requests

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        resp = requests.post(
            "http://127.0.0.1:8000/classify",
            files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")},
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "class_id" in data
        assert "class_name" in data
        assert "confidence" in data
        assert "latency_ms" in data
