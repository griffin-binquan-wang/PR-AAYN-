from utils import SimpleTokenizer
from model import TransformerClassifier
import torch

if __name__ == "__main__":
    raw_data = ["我 爱 学习", "学习 使 我 快乐","我不 喜欢 熬夜"]

    tokenizer = SimpleTokenizer(raw_data)

    test_sentence = "我 喜欢 学习"
    input_ids = tokenizer.encode(test_sentence,max_len=8)

    input_tensor = torch.tensor([input_ids])

    mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2)

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
        output = model(input_tensor,mask=mask)

    print(f"原始句子:{test_sentence}")
    print(f"转换后的ID:{input_ids}")
    print(f"模型输出:{output}")