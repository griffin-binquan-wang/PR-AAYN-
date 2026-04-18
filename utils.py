import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class BertTokenizerAdapter:
    def __init__(self,model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self,text,max_len=12):
        encoding = self.tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return encoding['input_ids'].flatten(),encoding['attention_mask']
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    # 必须缩进！告诉程序数据集总共有多少条数据
    def __len__(self):
        return len(self.texts)

    # 必须缩进！告诉程序怎么根据编号 idx 拿走一条数据
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        ids, mask = self.tokenizer.encode(text, max_len=self.max_len)
        return {
            'input_ids': ids,
            'attention_mask': mask,
            'label': torch.tensor(label, dtype=torch.long)
        }