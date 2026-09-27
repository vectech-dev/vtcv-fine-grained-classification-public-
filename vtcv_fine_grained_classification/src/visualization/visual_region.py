import torch
import torch.nn.functional as F


def crop_images(images, boxes):
    """Eq. 11: cut each box out of its image and resize it to the full image size (bilinear)."""
    _, _, h, w = images.shape
    out = torch.empty_like(images)
    for i, (x0, y0, x1, y1) in enumerate(boxes.tolist()):
        xa, ya = int(x0 * w), int(y0 * h)
        xb, yb = max(xa + 1, round(x1 * w)), max(ya + 1, round(y1 * h))
        patch = images[i:i + 1, :, ya:yb, xa:xb]
        out[i] = F.interpolate(patch, size=(h, w), mode="bilinear", align_corners=False)[0]
    return out
