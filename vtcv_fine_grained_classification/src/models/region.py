import torch


def attention_crop(maps, alpha):
    """Eq. 8-10: sum the k maps, keep pixels >= alpha * max, return normalised boxes [x0, y0, x1, y1]."""
    _, _, h, w = maps.shape
    boxes = []
    for m in maps.sum(1):
        ys, xs = torch.nonzero(m >= alpha * m.max(), as_tuple=True)
        if len(ys) == 0:
            boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0]))
        else:
            boxes.append(torch.tensor([
                xs.min().item() / w, ys.min().item() / h,
                (xs.max().item() + 1) / w, (ys.max().item() + 1) / h,
            ]))
    return torch.stack(boxes).to(maps.device)


def attention_drop(images, maps, d_phi):
    """Eq. 12-14: tau = d_phi * max over (x, y, i); zero every pixel that is >= tau in any of the k maps."""
    tau = d_phi * maps.flatten(1).max(1)[0].view(-1, 1, 1, 1)
    flagged = (maps >= tau).any(dim=1, keepdim=True)
    return images * (~flagged).to(images.dtype)
