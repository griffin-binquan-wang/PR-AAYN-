from utils import BertTokenizerAdapter
from model import TransformerClassifier
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":
    tokenizer = BertTokenizerAdapter()

    test_sentence = "I love machine learning"
    input_ids, raw_mask = tokenizer.encode(test_sentence, max_len=12)
    input_tensor = input_ids.unsqueeze(0)
    mask = raw_mask.unsqueeze(1).unsqueeze(2)

    model = TransformerClassifier(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        num_classes=2,
        vocab_size=tokenizer.get_vocab_size()
    )

    model.eval()
    with torch.no_grad():
        output = model(input_tensor, mask=mask)

    print(f"原句子：{test_sentence}")
    print(f"转换后的ID:{input_ids}")
    print(f"模型输出：{output}")
    print(f"BERT 词表大小: {tokenizer.get_vocab_size()}")


    