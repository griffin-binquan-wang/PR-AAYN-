import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import MultiHeadAttention, PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x):
        attn_output, _ = self.mha(x,x,x)
        x = self.norm1(x + attn_output)

        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,num_layers):
        super().__init__()
        self.embedding = nn.Embedding(10000,d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,num_heads,d_ff)
            for _ in range(num_layers)
        ])

    def forward(self,x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class TransformerClassifier(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,num_layers,num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(d_model,num_heads,d_ff,num_layers)
        self.fc = nn.Linear(d_model,num_classes)
    def forward(self,x):
        encoded_output = self.encoder(x)
        cls_feature = encoded_output[:,0,:]
        return self.fc(cls_feature)
