import torch
import torch.nn as nn
from model import TransformerClassifier

d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
num_classes = 2
vocab_size = 10000

model = TransformerClassifier(d_model,num_heads,d_ff,num_layers,num_classes)
input_data = torch.randint(0,vocab_size,(4,10))

model.eval()
with torch.no_grad():
    predictions = model(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Predictions shape: {predictions.shape}")
print("\n模型给出的原始分数(logits):")
print(predictions)