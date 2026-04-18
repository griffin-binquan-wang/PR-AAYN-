from utils import BertTokenizerAdapter
from model import TransformerClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from utils import BertTokenizerAdapter, SentimentDataset
from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dataset("imdb")
    train_data = dataset['train'].shuffle(seed=42).select(range(2000))
    tokenizer = BertTokenizerAdapter()
    train_dataset = SentimentDataset(train_data['text'], train_data['label'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = TransformerClassifier(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        num_classes=2,
        vocab_size=tokenizer.get_vocab_size()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    print("Start Training...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].unsqueeze(1).unsqueeze(2).to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(ids, mask=mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/3 | Avg Loss: {total_loss/len(train_loader):.4f}")


    