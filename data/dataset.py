import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_geometric.data import Data, Batch
import pandas as pd
from collections import defaultdict
import pickle


class TrafficGraphDataset(Dataset):
    """Dynamic Traffic Graph Dataset for Network Intrusion Detection"""
    
    def __init__(self, data_path, window_size=60, aging_factor=0.8, 
                 transform=None, temporal_split=0.8):
        super().__init__()
        self.data_path = data_path
        self.window_size = window_size
        self.aging_factor = aging_factor
        self.transform = transform
        self.temporal_split = temporal_split
        
        # Load raw data
        self.raw_data = self._load_data()
        
        # Build temporal graphs
        self.graphs, self.labels = self._build_temporal_graphs()
        
        # Split indices
        self._create_splits()
    
    def _load_data(self):
        """Load network traffic data"""
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.pkl'):
            with open(self.data_path, 'rb') as f:
                df = pickle.load(f)
        else:
            # Generate synthetic data for demonstration
            df = self._generate_synthetic_data()
        return df
    
    def _generate_synthetic_data(self):
        """Generate synthetic network traffic data for testing"""
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'timestamp': np.arange(n_samples),
            'src_ip': np.random.randint(1, 101, n_samples),
            'dst_ip': np.random.randint(1, 101, n_samples),
            'src_port': np.random.randint(1024, 65535, n_samples),
            'dst_port': np.random.randint(1, 1024, n_samples),
            'protocol': np.random.choice([6, 17, 1], n_samples),  # TCP, UDP, ICMP
            'duration': np.random.exponential(10, n_samples),
            'bytes': np.random.exponential(1000, n_samples),
            'packets': np.random.poisson(10, n_samples),
            'label': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        }
        
        return pd.DataFrame(data)
    
    def _build_temporal_graphs(self):
        """Build sequence of temporal graphs from traffic flows"""
        graphs = []
        labels = []
        
        # Sort by timestamp
        self.raw_data = self.raw_data.sort_values('timestamp')
        
        # Group into temporal windows
        time_steps = np.arange(0, len(self.raw_data), self.window_size)
        
        prev_adj = None
        prev_features = None
        
        for t in time_steps:
            window_data = self.raw_data.iloc[t:t+self.window_size]
            
            if len(window_data) < 10:
                continue
            
            # Build graph for this window
            graph, label = self._window_to_graph(window_data, prev_adj, prev_features)
            
            graphs.append(graph)
            labels.append(label)
            
            # Update previous state with aging
            if prev_adj is None:
                prev_adj = graph.adj_matrix if hasattr(graph, 'adj_matrix') else None
                prev_features = graph.x
            else:
                # Apply aging factor: A_t = λ * A_{t-1} + (1-λ) * A_hat
                if hasattr(graph, 'adj_matrix'):
                    prev_adj = self.aging_factor * prev_adj + (1 - self.aging_factor) * graph.adj_matrix
                prev_features = self.aging_factor * prev_features + (1 - self.aging_factor) * graph.x
        
        return graphs, labels
    
    def _window_to_graph(self, window_data, prev_adj=None, prev_features=None):
        """Convert a time window to a graph structure"""
        # Create node mapping
        nodes = set(window_data['src_ip'].unique()) | set(window_data['dst_ip'].unique())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        n_nodes = len(nodes)
        
        # Build adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        for _, row in window_data.iterrows():
            src_idx = node_to_idx[row['src_ip']]
            dst_idx = node_to_idx[row['dst_ip']]
            adj_matrix[src_idx, dst_idx] += 1
            adj_matrix[dst_idx, src_idx] += 1  # undirected
        
        # Normalize adjacency
        adj_matrix = adj_matrix / (adj_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        # Build node features
        node_features = []
        for node in nodes:
            # Aggregate features for this node
            node_flows = window_data[(window_data['src_ip'] == node) | (window_data['dst_ip'] == node)]
            
            features = [
                node_flows['bytes'].mean() if len(node_flows) > 0 else 0,
                node_flows['packets'].mean() if len(node_flows) > 0 else 0,
                node_flows['duration'].mean() if len(node_flows) > 0 else 0,
                len(node_flows),  # degree
                node_flows['protocol'].mode().iloc[0] if len(node_flows) > 0 else 0,
            ]
            node_features.append(features)
        
        node_features = np.array(node_features)
        
        # Apply temporal aging if previous state exists
        if prev_adj is not None and prev_features is not None:
            # Resize to match current graph size
            prev_adj_resized = self._resize_matrix(prev_adj, n_nodes)
            prev_features_resized = self._resize_features(prev_features, n_nodes)
            
            adj_matrix = self.aging_factor * prev_adj_resized + (1 - self.aging_factor) * adj_matrix
            node_features = self.aging_factor * prev_features_resized + (1 - self.aging_factor) * node_features
        
        # Create PyTorch Geometric Data object
        edge_index = self._adj_to_edge_index(adj_matrix)
        
        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            num_nodes=n_nodes
        )
        
        # Store additional info
        graph.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
        
        # Determine label (1 if any malicious flow in window)
        label = 1 if (window_data['label'] == 1).any() else 0
        
        return graph, label
    
    def _adj_to_edge_index(self, adj_matrix):
        """Convert adjacency matrix to edge_index format"""
        edges = np.where(adj_matrix > 0)
        edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)
        return edge_index
    
    def _resize_matrix(self, matrix, new_size):
        """Resize matrix to new dimensions"""
        if isinstance(matrix, np.ndarray):
            old_size = matrix.shape[0]
            if old_size == new_size:
                return matrix
            
            new_matrix = np.zeros((new_size, new_size))
            min_size = min(old_size, new_size)
            new_matrix[:min_size, :min_size] = matrix[:min_size, :min_size]
            return new_matrix
        else:
            # Handle torch tensor
            old_size = matrix.shape[0]
            if old_size == new_size:
                return matrix
            
            new_matrix = torch.zeros((new_size, new_size))
            min_size = min(old_size, new_size)
            new_matrix[:min_size, :min_size] = matrix[:min_size, :min_size]
            return new_matrix
    
    def _resize_features(self, features, new_size):
        """Resize feature matrix to new node count"""
        if isinstance(features, np.ndarray):
            old_size = features.shape[0]
            if old_size == new_size:
                return features
            
            feat_dim = features.shape[1]
            new_features = np.zeros((new_size, feat_dim))
            min_size = min(old_size, new_size)
            new_features[:min_size, :] = features[:min_size, :]
            return new_features
        else:
            old_size = features.shape[0]
            if old_size == new_size:
                return features
            
            feat_dim = features.shape[1]
            new_features = torch.zeros((new_size, feat_dim))
            min_size = min(old_size, new_size)
            new_features[:min_size, :] = features[:min_size, :]
            return new_features
    
    def _create_splits(self):
        """Create train/val/test splits"""
        n_graphs = len(self.graphs)
        train_end = int(n_graphs * 0.8)
        val_end = int(n_graphs * 0.9)
        
        self.train_indices = list(range(0, train_end))
        self.val_indices = list(range(train_end, val_end))
        self.test_indices = list(range(val_end, n_graphs))
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        
        if self.transform:
            graph = self.transform(graph)
        
        return graph, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function for batching graphs"""
    graphs, labels = zip(*batch)
    batched_graph = Batch.from_data_list(graphs)
    labels = torch.stack(labels)
    return batched_graph, labels
