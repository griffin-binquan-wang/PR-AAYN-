from utils import BertTokenizerAdapter
from model import TransformerClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from utils import BertTokenizerAdapter, SentimentDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    tokenizer = BertTokenizerAdapter()

    train_texts = ["this movie was great", "i hated this film", "acting was bad"] * 33
    train_labels = [1, 0, 0] * 33
    dataset = SentimentDataset(train_texts, train_labels, tokenizer)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TransformerClassifier(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        num_classes=2,
        vocab_size=tokenizer.get_vocab_size()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    print("Start Training...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids']
            mask = batch['attention_mask'].unsqueeze(1).unsqueeze(2)
            labels = batch['label']

            optimizer.zero_grad()
            outputs = model(ids, mask=mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")


    