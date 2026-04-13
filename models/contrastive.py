import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive learning loss with local and global objectives"""
    
    def __init__(self, temperature=0.2, local_weight=1.0, global_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.local_weight = local_weight
        self.global_weight = global_weight
    
    def forward(self, z1, z2, graph_z1, graph_z2):
        """
        Args:
            z1, z2: Node embeddings from two augmented views
            graph_z1, graph_z2: Graph embeddings from two views
        
        Returns:
            total_loss: Combined contrastive loss
            local_loss: Node-level contrastive loss
            global_loss: Graph-level contrastive loss
        """
        # Local contrastive loss (node-level)
        local_loss = self._node_contrastive_loss(z1, z2)
        
        # Global contrastive loss (graph-level)
        global_loss = self._graph_contrastive_loss(graph_z1, graph_z2)
        
        # Combined loss
        total_loss = self.local_weight * local_loss + self.global_weight * global_loss
        
        return total_loss, local_loss, global_loss
    
    def _node_contrastive_loss(self, z1, z2):
        """Compute node-level contrastive loss"""
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Positive pairs (diagonal)
        pos_sim = torch.diag(sim_matrix)
        
        # Negative pairs
        neg_sim = sim_matrix
        
        # InfoNCE loss
        loss = -torch.log(
            pos_sim.exp() / (pos_sim.exp() + neg_sim.exp().sum(dim=1) - pos_sim.exp())
        ).mean()
        
        return loss
    
    def _graph_contrastive_loss(self, g1, g2):
        """Compute graph-level contrastive loss"""
        # Normalize graph embeddings
        g1 = F.normalize(g1, dim=1)
        g2 = F.normalize(g2, dim=1)
        
        # For graph-level, we treat each graph as a sample
        batch_size = g1.size(0)
        
        # Similarity between positive pairs
        pos_sim = (g1 * g2).sum(dim=1) / self.temperature
        
        # Negative pairs (across batch)
        all_embeddings = torch.cat([g1, g2], dim=0)
        sim_matrix = torch.mm(g1, all_embeddings.t()) / self.temperature
        
        # Positive indices
        pos_indices = torch.arange(batch_size, device=g1.device)
        
        # InfoNCE loss
        loss = -torch.log(
            sim_matrix[pos_indices, pos_indices].exp() / 
            sim_matrix.exp().sum(dim=1)
        ).mean()
        
        return loss


class DiversityLoss(nn.Module):
    """Diversity loss to prevent representation collapse"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings):
        """Encourage diverse representations"""
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Penalize high similarity (encourage diversity)
        # Exclude self-similarity
        mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        
        diversity_loss = (sim_matrix ** 2).mean()
        
        return diversity_loss
