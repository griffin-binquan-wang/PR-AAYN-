import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,q,k,v):
        d_k = q.size(-1)
        scores = torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k,dtype=torch.float32))
        attn = F.softmax(scores,dim=-1)
        output = torch.matmul(attn,v)
        return output,attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
     
    def forward(self,q,k,v):
        batch_size = q.size(0)

        q = self.W_q(q).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        k = self.W_k(k).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        v = self.W_v(v).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)

        attn_output, attn_weights = ScaledDotProductAttention()(q,k,v)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.d_k)
        output = self.W_o(attn_output)
        return output, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:,:x.size(1),:]
        return x