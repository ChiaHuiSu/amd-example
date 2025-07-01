import torch
from torchvision import models, transforms
from PIL import Image

# ✅ Check if ROCm device is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load pre-trained MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.eval()
model.to(device)

# ✅ Prepare input image
img_path = "../../Untitled.jpeg"  # Replace with your actual image
img = Image.open(img_path)

# ✅ Image pre-processing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet mean
        std=[0.229, 0.224, 0.225]    # Imagenet std
    )
])

input_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dim and move to GPU

# ✅ Run inference
with torch.no_grad():
    output = model(input_tensor)

# ✅ Get top-1 prediction
predicted_class = torch.argmax(output[0])
print(f"Predicted class index: {predicted_class.item()}")
