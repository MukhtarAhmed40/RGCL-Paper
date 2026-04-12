import torch
from utils.data_utils import perturb_features, perturb_edges

def adversarial_attack(graph, epsilon=0.15):
    graph.x = perturb_features(graph.x, epsilon)
    graph.edge_index = perturb_edges(graph.edge_index, epsilon)
    return graph
