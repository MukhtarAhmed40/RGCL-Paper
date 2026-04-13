import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import random

from data.dataset import TrafficGraphDataset, collate_fn
from models.rgcl import RGCL
from utils.metrics import compute_metrics


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, epoch, device, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_contrastive_loss = 0
    total_robust_loss = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for batch, _ in pbar:
        batch = batch.to(device)
        
        # Compute contrastive loss
        contrastive_loss, local_loss, global_loss, div_loss = model.compute_contrastive_loss(batch)
        
        # Compute adversarial regularization loss
        robust_loss = model.adversarial_regularization(batch, epsilon=args.epsilon)
        
        # Total loss
        loss = contrastive_loss + args.robust_weight * robust_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_robust_loss += robust_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cl': f'{contrastive_loss.item():.4f}',
            'rob': f'{robust_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(loader)
    avg_cl = total_contrastive_loss / len(loader)
    avg_rob = total_robust_loss / len(loader)
    
    return avg_loss, avg_cl, avg_rob


def evaluate(model, loader, device):
    """Evaluate model on classification task"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            logits = model.classify(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--dataset', type=str, default='synthetic', 
                        help='Dataset path or name')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--out-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.15)
    parser.add_argument('--robust-weight', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Load config if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if hasattr(args, k):
                            setattr(args, k, v)
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = TrafficGraphDataset(
        data_path=args.dataset,
        window_size=60,
        aging_factor=0.8
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, dataset.train_indices)
    val_dataset = torch.utils.data.Subset(dataset, dataset.val_indices)
    test_dataset = torch.utils.data.Subset(dataset, dataset.test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, collate_fn=collate_fn)
    
    # Get input dimension from first graph
    sample_graph, _ = dataset[0]
    in_dim = sample_graph.x.size(1)
    print(f"Input dimension: {in_dim}")
    
    # Initialize model
    model = RGCL(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_heads=args.num_heads,
        temperature=args.temperature,
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_cl, train_rob = train_epoch(
            model, train_loader, optimizer, epoch, device, args
        )
        
        # Evaluate
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
            }, 'best_model.pt')
            print(f"  -> Saved best model (F1: {best_val_f1:.4f})")
    
    print(f"\nTraining completed. Best epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}")
    
    # Final evaluation on test set
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== Test Results ===")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    main()
