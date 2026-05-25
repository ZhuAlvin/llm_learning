# Verification: check all artifacts produced
import os, json

AD = "../../multimodal_platform/artifacts"
required = [
    "vit_tiny_base.pt",
    "lora_store_a.pt",
    "lora_store_a_int8.pt",
    "lora_store_a.onnx",
    "benchmark_report.json",
]
missing = [f for f in required if not os.path.exists(os.path.join(AD, f))]

if missing:
    print(f"MISSING artifacts: {missing}")
else:
    print("All 5 required artifacts present:")
    for f in required:
        p = os.path.join(AD, f)
        print(f"  [OK] {f} ({os.path.getsize(p)/1024:.1f} KB)")

if os.path.exists(os.path.join(AD, "benchmark_report.json")):
    with open(os.path.join(AD, "benchmark_report.json")) as f:
        r = json.load(f)
    print(f"\nBenchmark summary: ONNX speedup = {r.get('onnx_speedup','N/A')}x, LoRA size = {r.get('lora_kb','N/A')} KB")

print("\nEnd-to-end pipeline artifact verification complete.")
