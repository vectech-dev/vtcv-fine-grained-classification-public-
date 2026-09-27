import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalTaxonomicLoss(nn.Module):
    """Eq. 1: one triplet term per negative type, intra-genus margin delta < inter-genus margin beta."""

    def __init__(self, delta=0.3, beta=0.6):
        super().__init__()
        assert delta < beta, "delta must be smaller than beta"
        self.delta = delta
        self.beta = beta

    def forward(self, anchor, positive, neg_intra, neg_inter):
        dp = F.pairwise_distance(anchor, positive)
        dn_intra = F.pairwise_distance(anchor, neg_intra)
        dn_inter = F.pairwise_distance(anchor, neg_inter)
        loss = F.relu(self.delta + dp - dn_intra) + F.relu(self.beta + dp - dn_inter)
        return 0.5 * loss.mean()


class DiversityLoss(nn.Module):
    """Eq. 16: mean over the batch of max(0, m - d_b)^2, d_b = RMS difference of two fused maps.

    For k > 2 the loss is averaged over all pairs of maps; for k = 1 it is zero.
    """

    def __init__(self, margin=0.5, eps=1e-8):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, maps):
        k = maps.shape[1]
        if k < 2:
            return maps.new_zeros(())
        terms = []
        for i in range(k):
            for j in range(i + 1, k):
                d = ((maps[:, i] - maps[:, j]) ** 2).mean(dim=(1, 2)).add(self.eps).sqrt()
                terms.append(F.relu(self.margin - d).pow(2).mean())
        return torch.stack(terms).mean()
