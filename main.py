from model import TransformerClassifier, Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from utils import BertTokenizerAdapter, SentimentDataset
import utils
from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# def evaluate(model, data_label, device):
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch in test_loader:
#             ids = batch['input_ids'].to(device)
#             mask = batch['attention_mask'].unsqueeze(1).unsqueeze(2).to(device)
#             label = batch['label'].to(device)

#             output = model(ids, mask=mask)
#             _,predicted = torch.max(outputs.data, 1)

#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     return 100 * correct / total

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     dataset = load_dataset("imdb")
#     train_data = dataset['train'].shuffle(seed=42)
#     test_data = dataset['test'].shuffle(seed=42)
#     tokenizer = BertTokenizerAdapter()
#     train_dataset = SentimentDataset(train_data['text'], train_data['label'], tokenizer, max_len=128)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_dataset = SentimentDataset(test_data['text'], test_data['label'], tokenizer, max_len=128)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#     model = TransformerClassifier(
#         d_model=512,
#         num_heads=8,
#         d_ff=2048,
#         num_layers=8,
#         num_classes=2,
#         vocab_size=tokenizer.get_vocab_size()
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#     criterion = torch.nn.CrossEntropyLoss()

#     print("Start Training...")
#     for epoch in range(2):
#         model.train()
#         progress_bar = tqdm(train_loader, desc=f"Epoch{epoch+1}")
#         total_loss = 0
#         for batch in progress_bar:
#             ids = batch['input_ids'].to(device)
#             mask = batch['attention_mask'].unsqueeze(1).unsqueeze(2).to(device)
#             labels = batch['label'].to(device)

#             optimizer.zero_grad()
#             outputs = model(ids, mask=mask)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             progress_bar.set_postfix(loss=loss.item())
#         accuracy = evaluate(model, test_loader, device)
#         print(f"Epoch {epoch+1} Test acc:{accuracy:.2f}%")

#         print(f"Epoch {epoch+1}/2 | Avg Loss: {total_loss/len(train_loader):.4f}")
#     torch.save(model.state_dict(), "transformer_imdb_v1.pth")
#     print("Model saved to transformer_imdb_v1.pth")

# 1. 配置参数
src_vocab_size = 100
trg_vocab_size = 100
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 实例化你写的模型
model = Transformer(src_vocab_size, trg_vocab_size, d_model, num_layers, num_heads, d_ff, dropout).to(device)

# 3. 模拟输入 (Batch=2, 源句长=10, 目标句长=8)
src = torch.randint(1, src_vocab_size, (2, 10)).to(device)
trg = torch.randint(1, trg_vocab_size, (2, 8)).to(device)

# 4. 设置 PAD 的 ID（假设是 0）
src_pad_idx = 0
trg_pad_idx = 0

# 5. 调用你亲手写的 create_masks
src_mask, trg_mask = utils.create_masks(src, trg, src_pad_idx, trg_pad_idx, device)

# 6. 开启冒烟测试
model.eval()
with torch.no_grad():
    output = model(src, trg, src_mask, trg_mask)

print("-" * 30)
print(f"src_mask 形状: {src_mask.shape}") # 应该是 [2, 1, 1, 10]
print(f"trg_mask 形状: {trg_mask.shape}") # 应该是 [2, 1, 8, 8]
print(f"模型输出形状: {output.shape}")      # 应该是 [2, 8, 100]
print("-" * 30)
print("恭喜！如果形状全对，说明你的逻辑链条已经彻底打通了！")
    


    