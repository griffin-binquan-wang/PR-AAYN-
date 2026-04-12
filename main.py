import torch
import torch.nn as nn
from model import TransformerClassifier

if __name__ == "__main__":
    
    # 1. 设定超参数
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    num_classes = 2
    vocab_size = 10000

    # 2. 实例化模型
    model = TransformerClassifier(d_model, num_heads, d_ff, num_layers, num_classes)

    # 假设 Batch=2，句子长度=5
    # 第一句 3 个词 + 2 个补位 0；第二句 5 个词全满
    input_data = torch.tensor([
        [12, 45, 67, 0, 0],
        [89, 12, 34, 56, 78]
    ])

    # 3. 生成 Mask (掩码)
    mask = (input_data != 0).unsqueeze(1).unsqueeze(2) # 形状变为 [2, 1, 1, 5]

    # 4. 测试模型
    model.eval()
    with torch.no_grad():
        predictions = model(input_data, mask=mask)

    print(f"输入数据:\n{input_data}")
    print(f"掩码矩阵(Mask):\n{mask.int()}") # 转成 int 好看一点
    print(f"预测输出形状: {predictions.shape}")
    print("\n模型输出结果:")
    print(predictions)