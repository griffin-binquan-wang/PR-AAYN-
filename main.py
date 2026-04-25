import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Transformer 
from utils import BertTokenizerAdapter, TranslationDataset, create_masks, load_data_from_file, ScheduledOptim
from tqdm import tqdm

# # --- 1. 基础配置 ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# max_len = 32

# # --- 2. 实例化加载 ---
# tokenizer = BertTokenizerAdapter("bert-base-multilingual-cased")
# vocab_size = tokenizer.get_vocab_size()

# # 参数必须和训练时完全一致！
# model = Transformer(
#     src_vocab=vocab_size, 
#     trg_vocab=vocab_size, 
#     d_model=512, 
#     num_layers=8, 
#     num_heads=8, 
#     d_ff=2048, 
#     dropout=0.1
# ).to(device)

# # 加载你从 5090 拷下来的权重
# model_path = "transformer_epoch_15.pth"
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()


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


# # --- 4. 真正开始测试 ---
# if __name__ == "__main__":
#     while True:
#         sentence = input("\n请输入要翻译的英文 (输入 q 退出): ")
#         if sentence.lower() == 'q':
#             break
#         translation = translate(model, sentence, tokenizer, max_len=max_len)
#         print(f"翻译结果: {translation}")

# --- 1. 配置与超参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
max_len = 32
lr = 0

# --- 2. 准备数据 ---
# 1. 指定你下载并改名后的文件路径
file_path = 'data/train_cmn.txt'

# 2. 调用函数加载
src_data, trg_data = load_data_from_file(file_path)

# 打印一下看看加载了多少条
print(f"语料加载完成，共计 {len(src_data)} 条记录。")

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
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1) # 忽略 [PAD]
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
scheduled_optim = ScheduledOptim(optimizer, d_model=512, n_warmup_steps=2000)

# # --- 5. 训练循环 ---
print(f"正在使用 {device} 开始炼丹...")

model.train()
for epoch in range(100):
    total_loss = 0

    # 用 tqdm 把 loader 包起来
    # desc 是描述，leave=True 表示循环结束后进度条保留
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/100", leave=True)

    for batch in progress_bar:
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
        scheduled_optim.step()
        scheduled_optim.zero_grad()

        total_loss += loss.item()

        # 实时更新进度条显示的 Loss
        progress_bar.set_postfix(loss=loss.item())
    
    if (epoch + 1) % 3 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {total_loss/len(loader):.4f}")
    
    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), f"transformer_epoch_{epoch+1}.pth")

print("训练完成!")

    