"""model.py specifies the model architecture."""
import torch

class DialectNN(torch.nn.Module):
    def __init__(self, vocab_size, dim, max_length, output_size=12):
        super(DialectNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=max(dim//64, 1))
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.linear = torch.nn.Linear(dim * max_length, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.view(-1)
        x = self.linear(x)
        x = x.unsqueeze(0)
        return x