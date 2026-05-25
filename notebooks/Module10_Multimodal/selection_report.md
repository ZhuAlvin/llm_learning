# 多模态平台技术选型报告

生成时间: 2026-05-25T10:18:13.096343

## 安防-实时入侵检测
约束: latency<=20ms, acc>=65%, mem<=50MB
**推荐**: MobileNetV3-Small (得分: 0.8026)
- MobileNetV3-Small: acc=67.4%, cpu=12ms, mem=10MB

## 零售-货架SKU识别
约束: latency<=50ms, acc>=73%, mem<=30MB
**推荐**: FastViT-T8 (得分: 0.7734)
- FastViT-T8: acc=74.6%, cpu=25ms, mem=15MB
- MobileNetV3-Large: acc=75.2%, cpu=24ms, mem=22MB
- EfficientNet-B0: acc=77.1%, cpu=28ms, mem=21MB

## 医疗-皮肤镜筛查
约束: latency<=100ms, acc>=80%, mem<=100MB
**推荐**: EfficientNet-B3 (得分: 0.5684)
- EfficientNet-B3: acc=81.6%, cpu=68ms, mem=48MB
