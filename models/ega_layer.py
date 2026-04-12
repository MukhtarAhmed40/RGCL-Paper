import torch
import torch.nn as nn
import torch.nn.functional as F

class EGALayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8):
        super().__init__()
        self.heads = heads
        self.W = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.randn(2*out_dim, 1)) for _ in range(heads)])
        self.gate = nn.Parameter(torch.ones(heads))

    def forward(self, x, edge_index):
        outputs = []

        for m in range(self.heads):
            Wh = self.W[m](x)
            row, col = edge_index

            e = torch.cat([Wh[row], Wh[col]], dim=1)
            e = F.leaky_relu(torch.matmul(e, self.a[m]).squeeze())

            alpha = torch.zeros_like(e)
            alpha = torch.softmax(e, dim=0)

            out = torch.zeros_like(Wh)
            out.index_add_(0, row, alpha.unsqueeze(1) * Wh[col])
            outputs.append(out)

        g = torch.softmax(self.gate, dim=0)

        h = sum(g[m] * outputs[m] for m in range(self.heads))
        return h
