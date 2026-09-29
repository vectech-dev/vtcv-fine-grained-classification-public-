"""Section III-A: pretrain the Taxonomic Metric Network (TMN).

The backbone is trained only with the hierarchical taxonomic loss (Eq. 1) on L2-normalised
Block-11 embeddings. The linear head is trained at the same time as a probe on detached
embeddings, so it never changes the backbone; the dual-stream phase uses this head for the
taxonomic Grad-CAM maps.

    python -m vtcv_fine_grained_classification.train.pretrain_tmn --config config_examples/paper/tmn.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig
from vtcv_fine_grained_classification.src.configs.train_config import get_dataset, set_seed
from vtcv_fine_grained_classification.src.losses.losses import HierarchicalTaxonomicLoss
from vtcv_fine_grained_classification.src.models.model import Xception_V2_1


class HierarchicalTriplets(Dataset):
    """Each item: anchor, positive (same species, other sex when known),
    intra-genus negative (other species, same genus), inter-genus negative (other genus)."""

    def __init__(self, base):
        self.base = base
        df = base.df
        self.labels = df["y"].to_numpy()
        self.sex = df["Sex"].to_numpy()
        self.by_species = {y: np.flatnonzero(self.labels == y) for y in np.unique(self.labels)}
        genus_of = df.groupby("y")["Genus"].first().to_dict()
        self.genus_of = genus_of
        self.same_genus = {y: [s for s in genus_of if s != y and genus_of[s] == genus_of[y]] for y in genus_of}
        self.other_genus = {y: [s for s in genus_of if genus_of[s] != genus_of[y]] for y in genus_of}

    def __len__(self):
        return len(self.base)

    def _positive(self, idx):
        pool = self.by_species[self.labels[idx]]
        pool = pool[pool != idx] if len(pool) > 1 else pool
        if self.sex[idx] != "unknown":
            other = pool[(self.sex[pool] != self.sex[idx]) & (self.sex[pool] != "unknown")]
            if len(other):
                pool = other
        return int(random.choice(pool))

    def __getitem__(self, idx):
        y = self.labels[idx]
        intra_species = random.choice(self.same_genus[y] or self.other_genus[y])
        inter_species = random.choice(self.other_genus[y])
        ids = [idx, self._positive(idx),
               int(random.choice(self.by_species[intra_species])),
               int(random.choice(self.by_species[inter_species]))]
        return torch.stack([self.base.load_image(i) for i in ids]), int(y)


def run_epoch(model, loader, loss_fn, device, optim=None):
    train = optim is not None
    model.train(train)
    total_tri, total_ce, correct, n = 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for quads, labels in tqdm(loader, leave=False):
            b = quads.shape[0]
            x = quads.transpose(0, 1).reshape(-1, *quads.shape[2:]).to(device)
            labels = labels.to(device)
            _, end = model(x)

            emb = F.normalize(end["pooled"], dim=1)
            anchor, pos, neg_intra, neg_inter = emb.split(b)
            l_tri = loss_fn(anchor, pos, neg_intra, neg_inter)

            head_logits = model.model.last_linear(end["pooled"][:b].detach())
            l_ce = F.cross_entropy(head_logits, labels)

            if train:
                optim.zero_grad()
                (l_tri + l_ce).backward()
                optim.step()

            total_tri += l_tri.item() * b
            total_ce += l_ce.item() * b
            correct += (head_logits.argmax(1) == labels).sum().item()
            n += b
    return total_tri / n, total_ce / n, 100.0 * correct / n


def pretrain(config):
    set_seed(config.seed)
    device = config.device
    exp_dir = Path(config.save_dir) / config.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "TrainConfig.json").write_text(json.dumps(config.dict(), indent=2))

    model = Xception_V2_1(config.num_classes, imagenet=config.imagenet_init).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.tmn_lr, weight_decay=config.weight_decay)
    loss_fn = HierarchicalTaxonomicLoss(config.tmn_delta, config.tmn_beta)

    train_loader = DataLoader(HierarchicalTriplets(get_dataset(config, "Train")), batch_size=config.tmn_batch_size,
                              shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    valid_base = get_dataset(config, "Valid")
    valid_base.augment = False
    valid_loader = DataLoader(HierarchicalTriplets(valid_base), batch_size=config.tmn_batch_size,
                              shuffle=False, num_workers=config.num_workers, pin_memory=True)

    best = float("inf")
    for epoch in range(1, config.tmn_epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, device, optim)
        va = run_epoch(model, valid_loader, loss_fn, device)
        print(f"epoch {epoch}/{config.tmn_epochs}  train tri {tr[0]:.4f} head acc {tr[2]:.1f}  "
              f"valid tri {va[0]:.4f} head acc {va[2]:.1f}")
        if va[0] < best:
            best = va[0]
            torch.save(model.state_dict(), exp_dir / "tmn_best.pth")
    print(f"saved {exp_dir / 'tmn_best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    pretrain(ExperimentationConfig.parse_obj(json.loads(Path(args.config).read_text())))
