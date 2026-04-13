import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class EnsembleGraphAttention(nn.Module):
    """
    Ensemble Graph Attention (EGA) mechanism
    Stabilizes feature aggregation under noisy and perturbed graph structures
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8, dropout=0.5):
        super().__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Multiple attention heads
        self.attentions = nn.ModuleList([
            GATConv(in_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        # Attention aggregation weights (learnable)
        self.gate_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
        
        Returns:
            Aggregated node representations [N, out_dim]
        """
        # Compute attention from each head
        head_outputs = []
        
        for i, gat in enumerate(self.attentions):
            h = gat(x, edge_index)
            head_outputs.append(h)
        
        # Stack head outputs
        head_outputs = torch.stack(head_outputs, dim=0)  # [H, N, D]
        
        # Learnable aggregation using gate weights
        gate_weights = F.softmax(self.gate_weights, dim=0)
        gate_weights = gate_weights.view(-1, 1, 1)
        
        # Weighted sum of head outputs
        aggregated = (head_outputs * gate_weights).sum(dim=0)  # [N, D]
        
        # Residual connection and layer norm
        aggregated = self.layer_norm(aggregated + x[:, :self.hidden_dim])
        
        # Output projection
        output = self.out_proj(aggregated)
        
        return output
    
    def get_attention_weights(self, x, edge_index):
        """Extract attention weights for interpretability"""
        attention_weights = []
        
        for gat in self.attentions:
            # GATConv stores attention weights
            _ = gat(x, edge_index, return_attention_weights=True)
            # Note: This is a simplified version
            attention_weights.append(gat.att)
        
        return attention_weights
