import torch.nn as nn
import torch.nn.functional as F
from blocks import MultiHeadAttention, PositionalEncoding
import copy

class EncoderLayer(nn.Module):
    """
    单个编码器层 (Single Encoder Layer)
    包含: 多头自注意力 (Self-Attention) + 前馈网络 (Feed Forward)
    """
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )

        # 层归一化 (Layer Normalization)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        # 1. 自注意力 + 残差连接 + 归一化 (Self-Attn + Residual + Norm)
        attn_output, _ = self.mha(x,x,x,mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. 前馈网络 + 残差连接 + 归一化 (Feed Forward + Residual + Norm)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class TransformerEncoder(nn.Module):
    """
    编码器整体 (Full Transformer Encoder)
    负责将源序列转换为连续的表示向量
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 堆叠 num_layers 个编码器层 (Stack num_layers of EncoderLayers)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self,x,mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x,mask=mask)
        return x
    
class TransformerClassifier(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,num_layers,num_classes,vocab_size):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size,d_model,num_heads,d_ff,num_layers)
        self.fc = nn.Linear(d_model,num_classes)
    def forward(self,x,mask=None):
        encoded_output = self.encoder(x,mask)
        cls_feature = encoded_output[:,0,:]
        return self.fc(cls_feature)

class Transformer(nn.Module):
    """
    Transformer 完整模型 (The Full Transformer Model)
    连接 Encoder 和 Decoder 的桥梁
    """
    def __init__(self, src_vocab, trg_vocab, d_model, num_layers, num_heads, d_ff, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(trg_vocab, d_model, num_layers, num_heads, d_ff, dropout)
        
        # 最后的线性映射，输出维度为目标语言词表大小
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        # 编码源语言序列 (Encoding)
        e_outputs = self.encoder(src, src_mask)

        # 解码生成目标表示 (Decoding)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)

        # 映射到词表概率分布
        output = self.out(d_output)
        return output
    
class DecoderLayer(nn.Module):
    """
    单个解码器层 (Single Decoder Layer)
    比编码器多了一个 '交叉注意力' 层，用于关注编码器的输出
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 1. Masked 自注意力 (防止看到未来信息)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm_1 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)

        # 2. 交叉注意力 (Q来自Decoder, K/V来自Encoder输出)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout)

        # 3. 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        # 第一层：自注意力
        # 注意：MHA 返回通常是 (output, weights)，我们只要 output
        attn_out, _ = self.attn(x, x, x, trg_mask)
        x = self.norm_1(x + self.dropout_1(attn_out))

        # 第二层：交叉注意力 (Q来自Decoder, K,V来自Encoder)
        e_attn_out, _ = self.encoder_attn(x, e_outputs, e_outputs, src_mask)
        x = self.norm_2(x + self.dropout_2(e_attn_out))

        # 第三层：前馈网络
        ff_out = self.ff(x)
        x = self.norm_3(x + self.dropout_3(ff_out))
        return x                   

def get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, num_heads, d_ff, dropout), num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)