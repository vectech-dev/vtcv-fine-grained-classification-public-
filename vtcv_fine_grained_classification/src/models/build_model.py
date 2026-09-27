import torch
import torch.nn as nn

from vtcv_fine_grained_classification.src.models.model import Xception_V2_1


def build_model(config, device, load_tax=True):
    nets = {"base_model": Xception_V2_1(config.num_classes, imagenet=config.imagenet_init)}

    if config.use_taxonomic:
        tax = Xception_V2_1(config.num_classes, imagenet=False)
        if load_tax:
            if not config.tax_model_path:
                raise ValueError("use_taxonomic needs tax_model_path; run train.pretrain_tmn first")
            tax.load_state_dict(torch.load(config.tax_model_path, map_location="cpu"))
        tax.freeze_first_blocks(config.tax_frozen_blocks)
        nets["emb_model"] = tax
        # Eq. 6: one 3x3 conv, 2 channels in (semantic, taxonomic), 1 out, shared over the k pairs
        nets["combine_attn_maps"] = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.LeakyReLU())

    return nn.ModuleDict(nets).to(device)
