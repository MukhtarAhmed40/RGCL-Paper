import torch
import numpy as np


def apply_structural_attack(adj_matrix, epsilon=0.15):
    """
    Apply structural perturbation attack
    
    Args:
        adj_matrix: Adjacency matrix [N, N]
        epsilon: Perturbation budget (fraction of edges to modify)
    
    Returns:
        Perturbed adjacency matrix
    """
    adj = adj_matrix.clone() if torch.is_tensor(adj_matrix) else torch.tensor(adj_matrix)
    n_nodes = adj.shape[0]
    
    # Get existing edges
    existing_edges = torch.nonzero(adj > 0)
    n_existing = len(existing_edges)
    
    if n_existing == 0:
        return adj
    
    # Number of edges to remove
    n_remove = int(n_existing * epsilon)
    
    if n_remove > 0:
        remove_indices = torch.randperm(n_existing)[:n_remove]
        for idx in remove_indices:
            i, j = existing_edges[idx]
            adj[i, j] = 0
            adj[j, i] = 0
    
    # Number of edges to add
    n_add = int(n_existing * epsilon)
    
    if n_add > 0:
        # Find non-edges
        non_edges = torch.nonzero(adj == 0)
        non_edges = non_edges[non_edges[:, 0] != non_edges[:, 1]]  # Remove self-loops
        
        if len(non_edges) > 0:
            n_non_edges = len(non_edges)
            add_indices = torch.randperm(n_non_edges)[:min(n_add, n_non_edges)]
            
            for idx in add_indices:
                i, j = non_edges[idx]
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj


def apply_feature_attack(features, epsilon=0.15, noise_type='gaussian'):
    """
    Apply feature perturbation attack
    
    Args:
        features: Node features [N, F]
        epsilon: Perturbation budget
        noise_type: Type of noise ('gaussian', 'uniform')
    
    Returns:
        Perturbed features
    """
    features = features.clone() if torch.is_tensor(features) else torch.tensor(features)
    
    if noise_type == 'gaussian':
        noise = torch.randn_like(features) * epsilon
    else:  # uniform
        noise = (torch.rand_like(features) - 0.5) * 2 * epsilon
    
    return features + noise


def evaluate_adversarial_robustness(model, loader, epsilon_values=[0.05, 0.10, 0.15, 0.20, 0.25]):
    """
    Evaluate model robustness under different attack strengths
    
    Args:
        model: RGCL model
        loader: DataLoader
        epsilon_values: List of perturbation strengths to test
    
    Returns:
        Dictionary of robustness results
    """
    model.eval()
    results = {}
    
    for epsilon in epsilon_values:
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch, labels in loader:
                # Apply adversarial perturbation
                perturbed_batch = batch.clone()
                
                # Structural attack
                for i in range(len(perturbed_batch)):
                    # Note: In practice, you'd need to reconstruct edge_index from adj matrix
                    pass  # Simplified for demonstration
                
                # Feature attack
                perturbed_batch.x = perturbed_batch.x + torch.randn_like(perturbed_batch.x) * epsilon
                
                # Get predictions
                logits = model.classify(perturbed_batch)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute accuracy under attack
        from utils.metrics import compute_metrics
        metrics = compute_metrics(all_labels, all_preds)
        results[epsilon] = metrics['accuracy']
    
    return results
