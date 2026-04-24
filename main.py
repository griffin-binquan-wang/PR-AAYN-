import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Transformer 
from utils import BertTokenizerAdapter, TranslationDataset, create_masks

def translate(model, sentence, tokenizer, max_len=20):
    model.eval() # 切换到评估模式
    src_ids, _ = tokenizer.encode(sentence, max_len=max_len)
    src = src_ids.unsqueeze(0).to(device) # 增加 Batch 维度
    
    # 初始状态：只输入 SOS (假设 SOS 的 ID 是 101，请根据你的 tokenizer 确认)
    trg_input = torch.tensor([[101]]).to(device) 
    
    for i in range(max_len - 1):
        # 掩码
        src_mask, trg_mask = create_masks(src, trg_input, 0, 0, device)
        
        # 预测
        with torch.no_grad():
            output = model(src, trg_input, src_mask, trg_mask)
            
        # 取出最后一个词的预测结果，找到概率最大的 ID
        next_token = output.argmax(dim=-1)[:, -1].item()
        
        # 拼接到 trg_input
        trg_input = torch.cat([trg_input, torch.tensor([[next_token]]).to(device)], dim=1)
        
        if next_token == 102: # 遇到 EOS 就停
            break
            
    return tokenizer.tokenizer.decode(trg_input[0].tolist())

# --- 1. 配置与超参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
max_len = 20
lr = 0.0001

# --- 2. 准备数据 ---
src_data = ["I like learning deep learning.", "Attention is all you need."]
trg_data = ["我喜欢学习深度学习。", "你只需要注意力机制。"]

tokenizer = BertTokenizerAdapter("bert-base-multilingual-cased")
dataset = TranslationDataset(src_data, trg_data, tokenizer, max_len=max_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 实例化模型 ---
vocab_size = tokenizer.get_vocab_size()
model = Transformer(
    src_vocab=vocab_size, 
    trg_vocab=vocab_size, 
    d_model=512, 
    num_layers=6, 
    num_heads=8, 
    d_ff=2048, 
    dropout=0.1, 
).to(device)

# --- 4. 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss(ignore_index=0) # 忽略 [PAD]
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

# --- 5. 训练循环 ---
print(f"正在使用 {device} 开始炼丹...")

model.train()
for epoch in range(50):
    total_loss = 0
    for batch in loader:
        src = batch['src_ids'].to(device)
        trg = batch['trg_ids'].to(device)
        
        # 核心：构造错位
        trg_input = trg[:, :-1]
        trg_y = trg[:, 1:]
        
        # 创建掩码
        src_mask, trg_mask = create_masks(src, trg_input, 0, 0, device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, trg_input, src_mask, trg_mask)
        
        # 计算 Loss
        # output.size(-1) 实际上就是 vocab_size
        loss = criterion(output.view(-1, output.size(-1)), trg_y.contiguous().view(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(loader):.4f}")

print("训练完成！")

# 测试训练过的句子（看看它记住了没）
test_sentence = "I like learning deep learning."
result = translate(model, test_sentence, tokenizer)

print(f"\n[测试原句]: {test_sentence}")
print(f"[模型翻译]: {result}")

# 尝试一个没写过的句子（看看它的泛化能力）
test_sentence_new = "The model is powerful."
result_new = translate(model, test_sentence_new, tokenizer)

print(f"\n[测试新句]: {test_sentence_new}")
print(f"[模型翻译]: {result_new}")


    