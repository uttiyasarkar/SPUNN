import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(StockPredictor, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, input_dim)
        output = self.transformer_encoder(src)
        output = self.fc(output[-1, :, :])  # Use the last output for prediction
        return output

# Hyperparameters
input_dim = 1  # Using only the 'Close' price
model_dim = 64
num_heads = 4
num_layers = 3
output_dim = 1

model = StockPredictor(input_dim, model_dim, num_heads, num_layers, output_dim)