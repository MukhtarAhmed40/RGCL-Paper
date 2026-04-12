import torch

def edge_attack(edge_index, perturb_ratio=0.1):
    num_edges = edge_index.size(1)
    k = int(num_edges * perturb_ratio)

    perm = torch.randperm(num_edges)
    edge_index[:, perm[:k]] = edge_index[:, perm.flip(0)[:k]]

    return edge_index
