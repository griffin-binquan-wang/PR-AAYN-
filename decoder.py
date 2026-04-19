import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import MultiHeadAttention, PositionalEncoding
from model import EncoderLayer
import copy
 
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(heads, d_model)
        self.encoder_attn = MultiHeadAttention(heads, d_model)
        self.ff = EncoderLayer.feed_forward(d_model)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, trg_mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.encoder_attn(x2, e_outputs, e_outputs, src_mask))
        
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x                       

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.module):
    def __inti__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)

        return self.norm(x)
