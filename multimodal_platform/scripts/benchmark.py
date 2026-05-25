"""Benchmark script: load test the inference API with concurrent requests."""
import argparse
import time
import json
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import requests


def send_request(url: str, image_bytes: bytes) -> dict:
    """Send a single classify request and return latency."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url,
            files={"file": ("bench.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return {"status": resp.status_code, "latency_ms": elapsed, "error": None}
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"status": 0, "latency_ms": elapsed, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference API")
    parser.add_argument("--url", default="http://127.0.0.1:8000/classify")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    # Create test image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    print(f"Benchmark: {args.requests} requests, concurrency={args.concurrency}")
    print(f"Target: {args.url}")

    latencies = []
    errors = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(send_request, args.url, img_bytes) for _ in range(args.requests)]
        for future in as_completed(futures):
            result = future.result()
            latencies.append(result["latency_ms"])
            if result["error"]:
                errors += 1

    total_time = time.perf_counter() - t0

    lats = np.array(latencies)
    report = {
        "total_requests": args.requests,
        "concurrency": args.concurrency,
        "errors": errors,
        "total_time_s": round(total_time, 2),
        "throughput_rps": round(args.requests / total_time, 1),
        "p50_ms": round(np.percentile(lats, 50), 2),
        "p95_ms": round(np.percentile(lats, 95), 2),
        "p99_ms": round(np.percentile(lats, 99), 2),
        "avg_ms": round(np.mean(lats), 2),
        "min_ms": round(np.min(lats), 2),
        "max_ms": round(np.max(lats), 2),
    }

    print(json.dumps(report, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
