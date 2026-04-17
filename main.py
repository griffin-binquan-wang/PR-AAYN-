from utils import SimpleTokenizer
from model import TransformerClassifier
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":

    raw_data = ["我 爱 学习", "学习 使 我 快乐", "我不 喜欢 熬夜"]
    tokenizer = SimpleTokenizer(raw_data)

    model = TransformerClassifier(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        num_classes=2,
        vocab_size=tokenizer.get_vocab_size()
    )
    
    train_data = [
        ("我 爱 学习", 1),
        ("学习 使 我 快乐", 1),
        ("我不 喜欢 熬夜", 0),
        ("我 喜欢 编程", 1),
        ("我 讨厌 失败", 0),
        ("我不 喜欢 早起", 0)
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()

    print("开始训练……")
    for epoch in range(50):
        total_loss = 0
        for text,label in train_data:
            ids = torch.tensor([tokenizer.encode(text,max_len=8)])
            target = torch.tensor([label])
            mask = (ids != 0).unsqueeze(1).unsqueeze(2)

            optimizer.zero_grad()
            output = model(ids,mask=mask)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss{total_loss/len(train_data):.4f}")
    print("训练完成！")
    