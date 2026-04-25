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
    
def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0

def create_masks(src, trg, src_pad_idx, trg_pad_idx, device):
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(3)
    size = trg.size(1)

    l_mask = subsequent_mask(size).to(device)
    trg_mask = trg_pad_mask & l_mask

    return src_mask, trg_mask

class TranslationDataset(Dataset):
    def __init__(self, src_texts, trg_texts, tokenizer, max_len=128):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        # 编码源句子 (Encoder 输入)        
        src_ids, _ = self.tokenizer.encode(self.src_texts[idx], max_len=self.max_len)
        # 编码目标句子 (Decoder 输入/输出) 
        trg_ids, _ = self.tokenizer.encode(self.trg_texts[idx], max_len=self.max_len)

        return {
            'src_ids': src_ids,
            'trg_ids': trg_ids
        }

def load_data_from_file(file_path):
    src_texts, trg_texts = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            # 只有当这一行至少有两部分时才处理
            if len(parts) >= 2:
                src_texts.append(parts[0]) # 英文
                trg_texts.append(parts[1]) # 中文
    return src_texts, trg_texts

class ScheduledOptim:
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.d_model = d_model
        self.n_steps = 0

    def step(self):
        "更新参数和学习率"
        self.n_steps += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr(self):
        # 论文公式：d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
        return (self.d_model ** -0.5) * min(
            self.n_steps ** -0.5,
            self.n_steps * (self.n_warmup_steps ** -1.5)
        )