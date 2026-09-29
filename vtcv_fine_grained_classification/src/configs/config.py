from typing import List, Optional

from pydantic import BaseModel


class ExperimentationConfig(BaseModel):
    exp_name: str
    save_dir: str = "experiments"
    mode: str = "train"
    seed: int = 1
    device: str = "cuda"

    # data
    datasheet_path: str
    data_root: str = ""
    imsize: int = 299
    num_classes: int = 32
    use_mask: bool = True
    mask_color: List[int] = [0, 0, 0]
    train_flips: bool = True

    # optimisation (Section IV-A)
    epochs: int = 60
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    lr_patience: int = 5
    early_stopping_patience: int = 15
    imagenet_init: bool = True

    # components of Table II
    use_aug: bool = True
    use_taxonomic: bool = True
    w_cls: float = 1.0
    w_div: float = 0.15
    div_margin: float = 0.5

    # attention and augmentation (Section III-B, III-C)
    k: int = 2
    gamma_base: float = 0.2
    gamma_drop: float = 0.2
    gamma_crop: float = 0.6
    alpha_crop: float = 0.2
    alpha_crop_start: float = 0.0
    d_phi: float = 0.8
    d_phi_start: float = 1.0
    crop_background: List[int] = [255, 0, 255]

    # taxonomic stream (Section III-A)
    tax_model_path: Optional[str] = None
    tax_frozen_blocks: int = 8
    tmn_epochs: int = 30
    tmn_batch_size: int = 16
    tmn_lr: float = 3e-4
    tmn_delta: float = 0.3
    tmn_beta: float = 0.6

    num_vis_batches_valid: int = 0
    num_vis_batches_test: int = 0
