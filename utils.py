import torch

class SimpleTokenizer:
    def __init__(self,sentences):
        self.vocab =  {"<PAD>":0,"<UNK>":1}

        for sentence in sentences:
            for word in sentence.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        self.id_to_word = {v:k for k,v in self.vocab.items()}

    def encode(self,text,max_len=10):
        tokens = text.split()
        ids = []

        for word in tokens:
            ids.append(self.vocab.get(word,1))

        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids
    
    def get_vocab_size(self):
        return len(self.vocab)