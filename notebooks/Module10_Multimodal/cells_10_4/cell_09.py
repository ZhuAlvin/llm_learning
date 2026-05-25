# Micro Practice 7: Docker containerization scaffolding
# Generate Dockerfile + docker-compose.yml + requirements.txt

import os

pd = "../../multimodal_platform"
os.makedirs(pd, exist_ok=True)

# requirements.txt
with open(os.path.join(pd, "requirements.txt"), "w") as f:
    f.write("fastapi>=0.100.0\n")
    f.write("uvicorn[standard]>=0.23.0\n")
    f.write("torch>=2.0.0\n")
    f.write("torchvision>=0.15.0\n")
    f.write("timm>=0.9.0\n")
    f.write("transformers>=4.30.0\n")
    f.write("peft>=0.6.0\n")
    f.write("pillow>=10.0.0\n")
    f.write("numpy>=1.24.0\n")
    f.write("onnx>=1.14.0\n")
    f.write("onnxruntime>=1.15.0\n")
    f.write("python-multipart>=0.0.6\n")

# Dockerfile
with open(os.path.join(pd, "Dockerfile"), "w") as f:
    f.write("FROM python:3.11-slim\n\n")
    f.write("WORKDIR /app\n\n")
    f.write("RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*\n\n")
    f.write("COPY requirements.txt .\n")
    f.write("RUN pip install --no-cache-dir -r requirements.txt\n\n")
    f.write("COPY . .\n\n")
    f.write("EXPOSE 8000\n\n")
    f.write("HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1\n\n")
    f.write('CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000"]\n')

# docker-compose.yml
with open(os.path.join(pd, "docker-compose.yml"), "w") as f:
    f.write("version: '3.8'\n\n")
    f.write("services:\n")
    f.write("  inference:\n")
    f.write("    build: .\n")
    f.write("    ports:\n")
    f.write('      - "8000:8000"\n')
    f.write("    environment:\n")
    f.write("      - MODEL_PATH=/app/models\n")
    f.write("      - LOG_LEVEL=info\n")
    f.write("    volumes:\n")
    f.write("      - ./models:/app/models\n")
    f.write("      - ./artifacts:/app/artifacts\n")
    f.write("    restart: unless-stopped\n")
    f.write("    healthcheck:\n")
    f.write('      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]\n')
    f.write("      interval: 30s\n")
    f.write("      timeout: 5s\n")
    f.write("      retries: 3\n")
    f.write("      start_period: 10s\n")

print("multimodal_platform/ created:")
for fn in ["requirements.txt", "Dockerfile", "docker-compose.yml"]:
    p = os.path.join(pd, fn)
    print(f"  {fn} ({os.path.getsize(p)} bytes)")
