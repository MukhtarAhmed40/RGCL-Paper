import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from models.ega import EnsembleGraphAttention


class GraphEncoder(nn.Module):
    """Graph encoder with EGA for node and graph representations"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, 
                 num_heads=8, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Multiple EGA layers
        self.ega_layers = nn.ModuleList([
            EnsembleGraphAttention(hidden_dim, hidden_dim, hidden_dim, 
                                  num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection for node embeddings
        self.node_proj = nn.Linear(hidden_dim, out_dim)
        
        # Graph pooling
        self.pooling = global_mean_pool
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        """
        Args:
            data: PyG Data object with x, edge_index, batch
        
        Returns:
            node_embeddings: Node-level representations
            graph_embedding: Graph-level representation
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply EGA layers
        for ega in self.ega_layers:
            x = ega(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Node embeddings
        node_embeddings = self.node_proj(x)
        
        # Graph embedding via pooling
        graph_embedding = self.pooling(node_embeddings, batch)
        
        return node_embeddings, graph_embedding
