import torch
import torch.nn.functional as F


def grad_cam(logits, feature_maps, classes, eps=1e-7):
    """Eq. 4 / 5: channel weights are the spatial mean of d(logit of `classes`) / d(feature_maps).

    Returns (B, 1, h, w) maps, ReLU-rectified and min-max normalised to [0, 1] per image.
    The weights are treated as constants; the maps stay differentiable through feature_maps.
    """
    score = logits.gather(1, classes.view(-1, 1)).sum()
    grads = torch.autograd.grad(score, feature_maps, retain_graph=True)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature_maps).sum(1, keepdim=True))

    flat = cam.flatten(1)
    lo = flat.min(1)[0].view(-1, 1, 1, 1)
    hi = flat.max(1)[0].view(-1, 1, 1, 1)
    return (cam - lo) / (hi - lo + eps)
