import torch

def update_graph(A_prev, A_new, X_prev, X_new, lam=0.8):
    A_t = lam * A_prev + (1 - lam) * A_new
    X_t = lam * X_prev + (1 - lam) * X_new
    return A_t, X_t
