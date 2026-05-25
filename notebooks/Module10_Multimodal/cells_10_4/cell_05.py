# 🔬 Micro Practice 6: FastAPI Inference Service (local verification)
# Target: Build a curl-callable inference API

import os, sys, json, time, io, subprocess
import numpy as np
from PIL import Image

# Write server script line by line to avoid any quoting issues
sp = "_test_server.py"
with open(sp, "w") as f:
    f.write("import torch,nn,os,json,time,io\n")
    f.write("from PIL import Image\n")
    f.write("class M(nn.Module):\n")
    f.write(" def __init__(s,n=10):\n")
    f.write("  super().__init__()\n")
    f.write("  s.c=nn.Sequential(nn.Conv2d(3,32,3,1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))\n")
    f.write("  s.fc=nn.Linear(64,n)\n")
    f.write(" def forward(s,x):\n")
    f.write("  x=s.c(x);return s.fc(x.view(x.size(0),-1))\n")
    f.write("from fastapi import FastAPI,UploadFile,File\n")
    f.write("from fastapi.responses import JSONResponse\n")
    f.write("import uvicorn\n")
    f.write("app=FastAPI()\n")
    f.write("model=M(10).eval()\n")
    f.write("CN=[f'class_{i}' for i in range(10)]\n")
    f.write("@app.get('/health')\n")
    f.write("async def h(): return {'status':'ok'}\n")
    f.write("@app.post('/classify')\n")
    f.write("async def classify(file:UploadFile=File(...)):\n")
    f.write(" from torchvision import transforms as T\n")
    f.write(" c=await file.read()\n")
    f.write(" img=Image.open(io.BytesIO(c)).convert('RGB').resize((224,224))\n")
    f.write(" tensor=T.ToTensor()(img).unsqueeze(0)\n")
    f.write(" tensor=T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))(tensor)\n")
    f.write(" with torch.inference_mode():\n")
    f.write("  logits=model(tensor);probs=logits.softmax(1)[0];pred=probs.argmax().item()\n")
    f.write(" return JSONResponse({'class_id':pred,'class_name':CN[pred],'confidence':round(probs[pred].item(),4)})\n")
    f.write("if __name__=='__main__':\n")
    f.write(" uvicorn.run(app,host='127.0.0.1',port=8765,log_level='warning')\n")

proc = subprocess.Popen([sys.executable, sp], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(3)

# Test health check
import urllib.request
try:
    r = urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=5)
    print(f"Health: {json.loads(r.read())}")
except Exception as e:
    print(f"Health: {e}")

# Test classify
try:
    import requests
    img = Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8))
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    r = requests.post("http://127.0.0.1:8765/classify",
        files={"file":("t.jpg",buf.getvalue(),"image/jpeg")}, timeout=10)
    print(f"Classify: {r.json()}")
except Exception as e:
    print(f"Classify: {e}")

proc.terminate(); proc.wait(); time.sleep(1)
if os.path.exists(sp): os.remove(sp)
print("FastAPI service verification complete.")
