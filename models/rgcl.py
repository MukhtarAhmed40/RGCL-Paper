import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from models.encoder import GraphEncoder
from models.contrastive import ContrastiveLoss, DiversityLoss
from data.preprocessing import AugmentGraph


class RGCL(nn.Module):
    """
    Robust Graph Contrastive Learning for Adversarial Network Intrusion Detection
    """
    
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_layers=3,
                 num_heads=8, temperature=0.2, dropout=0.5, 
                 local_weight=1.0, global_weight=1.0, div_weight=0.1):
        super().__init__()
        
        self.encoder = GraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.contrastive_loss = ContrastiveLoss(
            temperature=temperature,
            local_weight=local_weight,
            global_weight=global_weight
        )
        
        self.diversity_loss = DiversityLoss(temperature=temperature)
        self.div_weight = div_weight
        
        # Classifier for downstream tasks
        self.classifier = nn.Linear(out_dim, 2)
        
        # Graph augmentation
        self.augment = AugmentGraph(drop_edge_prob=0.1, mask_feature_prob=0.1)
        
    def forward(self, data):
        """Forward pass through encoder"""
        return self.encoder(data)
    
    def compute_contrastive_loss(self, data):
        """Compute contrastive loss on augmented views"""
        # Create two augmented views
        data1 = self.augment(data.clone())
        data2 = self.augment(data.clone())
        
        # Encode both views
        z1_nodes, z1_graph = self.encoder(data1)
        z2_nodes, z2_graph = self.encoder(data2)
        
        # Compute contrastive loss
        total_loss, local_loss, global_loss = self.contrastive_loss(
            z1_nodes, z2_nodes, z1_graph, z2_graph
        )
        
        # Add diversity loss
        all_embeddings = torch.cat([z1_nodes, z2_nodes], dim=0)
        div_loss = self.diversity_loss(all_embeddings)
        
        return total_loss + self.div_weight * div_loss, local_loss, global_loss, div_loss
    
    def adversarial_regularization(self, data, epsilon=0.15):
        """
        Adversarial regularization for robustness
        Args:
            data: Input graph
            epsilon: Perturbation budget
        """
        # Clean embeddings
        clean_nodes, clean_graph = self.encoder(data)
        
        # Generate adversarial perturbations
        # Structural perturbation (edge dropping/adding)
        perturbed_data = self._generate_adversarial_perturbation(data, epsilon)
        
        # Perturbed embeddings
        perturbed_nodes, perturbed_graph = self.encoder(perturbed_data)
        
        # Robustness loss (L2 distance between clean and perturbed embeddings)
        robust_loss = torch.norm(clean_graph - perturbed_graph, p=2, dim=1).mean()
        
        return robust_loss
    
    def _generate_adversarial_perturbation(self, data, epsilon):
        """Generate adversarial perturbations on graph structure and features"""
        perturbed = data.clone()
        
        # Structural perturbation: drop edges randomly
        if epsilon > 0:
            n_edges = data.edge_index.size(1)
            n_drop = int(n_edges * epsilon)
            if n_drop > 0 and n_edges > n_drop:
                drop_indices = torch.randperm(n_edges)[:n_drop]
                keep_mask = torch.ones(n_edges, dtype=torch.bool)
                keep_mask[drop_indices] = False
                perturbed.edge_index = data.edge_index[:, keep_mask]
        
        # Feature perturbation: add Gaussian noise
        if epsilon > 0:
            noise = torch.randn_like(perturbed.x) * epsilon
            perturbed.x = perturbed.x + noise
        
        return perturbed
    
    def classify(self, data):
        """Classification for intrusion detection"""
        _, graph_emb = self.encoder(data)
        logits = self.classifier(graph_emb)
        return logits
    
    def get_embeddings(self, loader):
        """Extract embeddings from all samples"""
        self.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch, batch_labels in loader:
                _, graph_emb = self.encoder(batch)
                embeddings.append(graph_emb.cpu())
                labels.append(batch_labels.cpu())
        
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)
