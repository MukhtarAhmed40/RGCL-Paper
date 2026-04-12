import torch.nn as nn
from models.encoder import Encoder
from models.contrastive import contrastive_loss

class RGCL(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim)

    def forward(self, g1, g2):
        z1 = self.encoder(g1.x, g1.edge_index)
        z2 = self.encoder(g2.x, g2.edge_index)

        loss = contrastive_loss(z1, z2)
        return z1, loss
