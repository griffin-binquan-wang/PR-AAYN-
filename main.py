from model import TransformerClassifier, Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from utils import BertTokenizerAdapter, SentimentDataset, TranslationDataset
import utils
from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

src_data = [
    "I like learning deep learning.",
    "The transformer model is very powerful.",
    "Artificial intelligence will change the world.",
    "I am building a neural network.",
    "Attention is all you need."
]

trg_data = [
    "我喜欢学习深度学习。",
    "Transformer模型非常强大。",
    "人工智能将改变世界。",
    "我正在构建一个神经网络。",
    "你只需要注意力机制。"
]

tokenizer = BertTokenizerAdapter("bert-base-chinese")
max_len = 15
dataset = TranslationDataset(src_data, trg_data, tokenizer, max_len=max_len)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in loader:
    src_ids = batch['src_ids']
    trg_ids = batch['trg_ids']

    print("Source IDs Batch Shape:", src_ids.shape)
    print("First Source ID Sequence:", src_ids[0])
    print("First Target ID Sequence:", trg_ids[0])
    break



# # 1. 配置参数
# src_vocab_size = 100
# trg_vocab_size = 100
# d_model = 512
# num_layers = 6
# num_heads = 8
# d_ff = 2048
# dropout = 0.1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 2. 实例化你写的模型
# model = Transformer(src_vocab_size, trg_vocab_size, d_model, num_layers, num_heads, d_ff, dropout).to(device)

# # 3. 模拟输入 (Batch=2, 源句长=10, 目标句长=8)
# src = torch.randint(1, src_vocab_size, (2, 10)).to(device)
# trg = torch.randint(1, trg_vocab_size, (2, 8)).to(device)

# # 4. 设置 PAD 的 ID（假设是 0）
# src_pad_idx = 0
# trg_pad_idx = 0

# # 5. 调用你亲手写的 create_masks
# src_mask, trg_mask = utils.create_masks(src, trg, src_pad_idx, trg_pad_idx, device)

# # 6. 开启冒烟测试
# model.eval()
# with torch.no_grad():
#     output = model(src, trg, src_mask, trg_mask)

# print("-" * 30)
# print(f"src_mask 形状: {src_mask.shape}") # 应该是 [2, 1, 1, 10]
# print(f"trg_mask 形状: {trg_mask.shape}") # 应该是 [2, 1, 8, 8]
# print(f"模型输出形状: {output.shape}")      # 应该是 [2, 8, 100]
# print("-" * 30)
# print("恭喜！如果形状全对，说明你的逻辑链条已经彻底打通了！")
    


    