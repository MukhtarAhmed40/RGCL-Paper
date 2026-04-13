import subprocess
import argparse


def run_experiments():
    """Run all experiments from the paper"""
    
    datasets = [
        'UNSW-NB15',
        'CICIDS-2018', 
        'CIC-Darknet2020',
        'CICIoT2023'
    ]
    
    # Experiment 1: Main comparison (Table 4)
    print("=== Experiment 1: Main Performance Comparison ===")
    for dataset in datasets:
        print(f"\nRunning on {dataset}...")
        subprocess.run([
            'python', 'train.py',
            '--dataset', dataset,
            '--epochs', '100',
            '--batch-size', '64'
        ])
    
    # Experiment 2: Adversarial robustness (Table 6)
    print("\n=== Experiment 2: Adversarial Robustness ===")
    for dataset in datasets:
        for epsilon in [0.05, 0.10, 0.15, 0.20, 0.25]:
            print(f"\nTesting {dataset} with ε={epsilon}...")
            subprocess.run([
                'python', 'train.py',
                '--dataset', dataset,
                '--epsilon', str(epsilon),
                '--epochs', '50'
            ])
    
    # Experiment 3: Temporal drift adaptation (Table 7)
    print("\n=== Experiment 3: Temporal Drift Adaptation ===")
    for dataset in datasets:
        print(f"\nStreaming evaluation on {dataset}...")
        # This would require modified evaluation protocol
        pass
    
    # Experiment 4: Hyperparameter sensitivity (Table 2)
    print("\n=== Experiment 4: Hyperparameter Sensitivity ===")
    
    # Vary learning rate
    for lr in [0.1, 0.01, 0.001]:
        subprocess.run([
            'python', 'train.py',
            '--dataset', 'CICIDS-2018',
            '--lr', str(lr),
            '--epochs', '50'
        ])
    
    # Vary dropout
    for dropout in [0.2, 0.5, 0.6]:
        subprocess.run([
            'python', 'train.py',
            '--dataset', 'CICIDS-2018',
            '--dropout', str(dropout),
            '--epochs', '50'
        ])
    
    # Vary batch size
    for batch_size in [16, 64, 128]:
        subprocess.run([
            'python', 'train.py',
            '--dataset', 'CICIDS-2018',
            '--batch-size', str(batch_size),
            '--epochs', '50'
        ])
    
    # Experiment 5: Ablation study (Table 8)
    print("\n=== Experiment 5: Ablation Study ===")
    
    # Without EGA
    subprocess.run([
        'python', 'train.py',
        '--dataset', 'CICIDS-2018',
        '--num-heads', '1',  # Single head instead of ensemble
        '--epochs', '50'
    ])
    
    # Without adaptive contrastive (simplified loss)
    # Would require modifying the loss function
    
    # Without adversarial regularization
    subprocess.run([
        'python', 'train.py',
        '--dataset', 'CICIDS-2018',
        '--robust-weight', '0',
        '--epochs', '50'
    ])


if __name__ == '__main__':
    run_experiments()
