import torch
import torch.nn as nn
import torch.nn.functional as F

from vtcv_fine_grained_classification.src.losses.losses import DiversityLoss
from vtcv_fine_grained_classification.src.models.region import attention_crop, attention_drop
from vtcv_fine_grained_classification.src.visualization.attention_map import grad_cam
from vtcv_fine_grained_classification.src.visualization.visual_region import crop_images


def fused_attention_maps(images, logits, feature_maps, topk, nets, k):
    """Section III-B: one Grad-CAM map per stream for each of the top-k classes of the semantic
    stream, fused with the shared 3x3 conv (Eq. 6). Without the taxonomic stream the semantic
    maps are used directly."""
    tax = "emb_model" in nets
    if tax:
        tax_logits, tax_out = nets["emb_model"](images)

    maps = []
    for i in range(k):
        m = grad_cam(logits, feature_maps, topk[:, i])
        if tax:
            m_tax = grad_cam(tax_logits, tax_out["feature_maps"], topk[:, i])
            m = nets["combine_attn_maps"](torch.cat([m, m_tax], dim=1))
        maps.append(m)
    return torch.cat(maps, dim=1)


def forward_pass(images, nets, config, alpha, d_phi, train, need_maps):
    logits, end = nets["base_model"](images)
    prob, topk = logits.softmax(1).topk(config.k, dim=1)
    out = {"logits": logits, "prop": prob.detach(), "topk": topk}
    if not need_maps:
        return out

    maps = fused_attention_maps(images, logits, end["feature_maps"], topk, nets, config.k)
    out["fused_maps"] = maps
    maps_img = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False)
    out["attention_map"] = maps_img.detach()

    if config.use_aug:
        a = maps_img.detach()
        out["images_crop"] = crop_images(images, attention_crop(a, alpha), config.crop_background)
        out["images_drop"] = attention_drop(images, a, d_phi)
        if train:
            out["logits_crop"] = nets["base_model"](out["images_crop"])[0]
            out["logits_drop"] = nets["base_model"](out["images_drop"])[0]
    return out


def compute_losses(out, labels, config, div_loss):
    """Eq. 15-17."""
    z = out["logits"]
    if "logits_crop" in out:
        z = config.gamma_base * z + config.gamma_drop * out["logits_drop"] + config.gamma_crop * out["logits_crop"]
    l_cls = F.cross_entropy(z, labels)

    l_div = z.new_zeros(())
    if config.w_div > 0 and "fused_maps" in out:
        l_div = div_loss(out["fused_maps"])

    total = config.w_cls * l_cls + config.w_div * l_div
    return {"total": total, "cls": l_cls.detach(), "div": l_div.detach()}


def thresholds(config, epoch):
    """Linear annealing of alpha and d_phi from their start values to the final values."""
    t = 1.0 if config.epochs <= 1 else min(1.0, max(0.0, (epoch - 1) / (config.epochs - 1)))
    alpha = config.alpha_crop_start + t * (config.alpha_crop - config.alpha_crop_start)
    d_phi = config.d_phi_start + t * (config.d_phi - config.d_phi_start)
    return alpha, d_phi


class MainModel(nn.Module):
    def __init__(self, nets, config):
        super().__init__()
        self.nets = nets
        self.config = config
        self.div_loss = DiversityLoss(config.div_margin)

    def forward(self, images, labels, train, epoch=1, visualize=False):
        cfg = self.config
        need_maps = visualize or (train and (cfg.use_aug or cfg.w_div > 0))
        alpha, d_phi = thresholds(cfg, epoch)
        with torch.set_grad_enabled(train or need_maps):
            out = forward_pass(images, self.nets, cfg, alpha, d_phi, train, need_maps)
            losses = compute_losses(out, labels, cfg, self.div_loss)
        return losses, out
