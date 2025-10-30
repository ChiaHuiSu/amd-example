import torch
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ 指定權重（含類別名稱）
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights).to(device).eval()
categories = weights.meta["categories"]   # 長度 1000 的類別名稱列表

# ✅ 前處理（用同套權重的 transforms 保險）
preprocess = weights.transforms()

# [TODO] Change your image
img = Image.open("/path/to/your/image")
input_tensor = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)          # logits
    probs = output.softmax(1)

# ✅ Top-1
top1_id = int(probs.argmax(1))
print(f"Top-1 id: {top1_id}, label: {categories[top1_id]}, prob: {probs[0, top1_id].item():.4f}")

# ✅ Top-5（可選）
top5_prob, top5_id = probs.topk(5)
for i in range(5):
    idx = int(top5_id[0, i])
    print(f"#{i+1}: id={idx}, {categories[idx]} ({top5_prob[0, i].item():.4f})")
