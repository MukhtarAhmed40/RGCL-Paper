import torch.nn as nn
from models.ega_layer import EGALayer

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.ega = EGALayer(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        h = self.ega(x, edge_index)
        return self.linear(h)
