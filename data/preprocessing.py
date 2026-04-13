import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import BaseTransform


class NormalizeFeatures(BaseTransform):
    """Normalize node features"""
    
    def __init__(self):
        self.scaler = None
    
    def __call__(self, data):
        if self.scaler is None:
            self.scaler = StandardScaler()
            # Fit on first call
            features = data.x.numpy()
            self.scaler.fit(features)
        
        data.x = torch.tensor(self.scaler.transform(data.x.numpy()), dtype=torch.float)
        return data


class AddSelfLoops(BaseTransform):
    """Add self-loops to adjacency matrix"""
    
    def __call__(self, data):
        # Add self-loops
        n_nodes = data.num_nodes
        self_loops = torch.tensor([[i, i] for i in range(n_nodes)], dtype=torch.long).t()
        data.edge_index = torch.cat([data.edge_index, self_loops], dim=1)
        return data


class AugmentGraph(BaseTransform):
    """Graph augmentation for contrastive learning"""
    
    def __init__(self, drop_edge_prob=0.1, mask_feature_prob=0.1):
        self.drop_edge_prob = drop_edge_prob
        self.mask_feature_prob = mask_feature_prob
    
    def __call__(self, data):
        # Edge dropping
        if self.drop_edge_prob > 0:
            n_edges = data.edge_index.size(1)
            keep_mask = torch.rand(n_edges) > self.drop_edge_prob
            data.edge_index = data.edge_index[:, keep_mask]
        
        # Feature masking
        if self.mask_feature_prob > 0:
            mask = torch.rand(data.x.size()) > self.mask_feature_prob
            data.x = data.x * mask.float()
        
        return data
