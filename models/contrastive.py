import torch
import torch.nn.functional as F

def sim(z1, z2):
    return F.cosine_similarity(z1, z2)

def contrastive_loss(z1, z2, tau=0.2):
    N = z1.size(0)
    loss = 0

    for i in range(N):
        pos = torch.exp(sim(z1[i], z2[i]) / tau)
        neg = torch.sum(torch.exp(sim(z1[i], z2) / tau))
        loss += -torch.log(pos / neg)

    return loss / N
