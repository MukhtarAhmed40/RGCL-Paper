import torch

def feature_attack(x, epsilon=0.1):
    noise = epsilon * torch.randn_like(x)
    return x + noise
