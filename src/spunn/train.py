import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, random_split

class StockDataset(Dataset):
    def __init__(self, data_dir, window_size=30):
        self.data = []
        self.window_size = window_size
        for file in os.listdir(data_dir):
            if file.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(data_dir, file))
                self.data.append(torch.tensor(df.values, dtype=torch.float32))
        self.data = torch.cat(self.data, dim=0)
    
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        return (self.data[idx:idx + self.window_size], self.data[idx + self.window_size])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.decoder = nn.Linear(d_model, input_dim)
    
    def forward(self, src):
        src = self.encoder(src)
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch, feature)
        output = self.transformer(src, src)
        output = output.permute(1, 0, 2)
        return self.decoder(output)

# Load dataset
data_dir = "processed_data"
dataset = StockDataset(data_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model training
model = TransformerModel(input_dim=dataset.data.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[:, -1, :], targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

print("Training complete!")
