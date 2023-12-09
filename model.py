"""model.py specifies the model architecture."""
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DialectNN(torch.nn.Module):
    def __init__(self, vocab_size, dim, max_length, output_size=12):
        super(DialectNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.positional_encoding = torch.nn.Embedding(max_length, dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=max(dim//128, 1), dropout=0.0)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.linear = torch.nn.Linear(dim * max_length, output_size)
        self.activation = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(output_size)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.embedding(x)
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        #x = x + self.positional_encoding(positions)
        x = self.encoder(x)
        x = x.view(-1)
        x = self.linear(x)
        # x = self.activation(x)
        # x = self.layer_norm(x)
        # x = self.dropout(x)
        x = x.unsqueeze(0)
        return x