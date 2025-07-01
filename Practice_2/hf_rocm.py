import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased").to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello ROCm!", return_tensors="pt").to(device)
outputs = model(**inputs)
print(outputs.logits)
