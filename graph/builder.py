import torch
from torch_geometric.data import Data

def build_graph(X, edge_list):
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)
