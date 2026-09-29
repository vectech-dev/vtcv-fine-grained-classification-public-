import torch


def crop_images(images, boxes, background):
    """Eq. 11: keep the pixels inside each box and set all pixels outside it to the background colour.
    The image size does not change."""
    _, _, h, w = images.shape
    bg = torch.tensor(background, device=images.device, dtype=images.dtype).view(1, 3, 1, 1) / 255.0
    out = bg.expand_as(images).clone()
    for i, (x0, y0, x1, y1) in enumerate(boxes.tolist()):
        xa, ya = int(x0 * w), int(y0 * h)
        xb, yb = max(xa + 1, round(x1 * w)), max(ya + 1, round(y1 * h))
        out[i, :, ya:yb, xa:xb] = images[i, :, ya:yb, xa:xb]
    return out
