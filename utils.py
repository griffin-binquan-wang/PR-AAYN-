import torch
from transformers import AutoTokenizer

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