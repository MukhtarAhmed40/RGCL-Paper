import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np

from data.dataset import TrafficGraphDataset, collate_fn
from models.rgcl import RGCL
from utils.metrics import compute_metrics, compute_robust_metrics
from utils.adversarial import evaluate_adversarial_robustness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--adversarial', action='store_true', help='Evaluate adversarial robustness')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = TrafficGraphDataset(data_path=args.dataset)
    
    # Create test loader
    test_dataset = torch.utils.data.Subset(dataset, dataset.test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                              shuffle=False, collate_fn=collate_fn)
    
    # Get input dimension
    sample_graph, _ = dataset[0]
    in_dim = sample_graph.x.size(1)
    
    # Initialize model
    model = RGCL(in_dim=in_dim).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    from train import evaluate
    metrics = evaluate(model, test_loader, device)
    
    print("\n=== Test Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Adversarial evaluation
    if args.adversarial:
        print("\n=== Adversarial Robustness Evaluation ===")
        adv_results = evaluate_adversarial_robustness(model, test_loader)
        
        print("Accuracy under different perturbation strengths:")
        for epsilon, acc in adv_results.items():
            print(f"  ε={epsilon}: {acc:.4f}")


if __name__ == '__main__':
    main()
