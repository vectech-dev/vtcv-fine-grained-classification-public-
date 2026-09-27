import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig
from vtcv_fine_grained_classification.src.configs.train_config import get_dataset, set_seed
from vtcv_fine_grained_classification.src.models.build_model import build_model
from vtcv_fine_grained_classification.src.models.process_model import MainModel
from vtcv_fine_grained_classification.train.train_utils import save_visualizations


def run_epoch(model, loader, config, device, epoch, optim=None, vis_dir=None):
    train = optim is not None
    model.train(train)
    sums = {"total": 0.0, "cls": 0.0, "div": 0.0}
    correct, n = 0, 0

    for b, (images, labels) in enumerate(tqdm(loader, leave=False)):
        images, labels = images.to(device), labels.to(device)
        visualize = vis_dir is not None and b < config.num_vis_batches_valid
        losses, out = model(images, labels, train=train, epoch=epoch, visualize=visualize)

        if train:
            optim.zero_grad()
            losses["total"].backward()
            optim.step()
        if visualize:
            save_visualizations(images, labels, out, b, vis_dir)

        bs = len(labels)
        for key in sums:
            sums[key] += losses[key].item() * bs
        correct += (out["logits"].argmax(1) == labels).sum().item()
        n += bs

    stats = {key: value / n for key, value in sums.items()}
    stats["acc"] = 100.0 * correct / n
    return stats


def train(config: ExperimentationConfig):
    set_seed(config.seed)
    device = config.device
    exp_dir = Path(config.save_dir) / config.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "TrainConfig.json").write_text(json.dumps(config.dict(), indent=2))

    model = MainModel(build_model(config, device), config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.1, patience=config.lr_patience)

    train_loader = DataLoader(get_dataset(config, "Train"), batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(get_dataset(config, "Valid"), batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True)

    writer = SummaryWriter(log_dir=f"runs/{config.exp_name}")
    best_acc, best_loss, bad_epochs = -1.0, float("inf"), 0

    for epoch in range(1, config.epochs + 1):
        tr = run_epoch(model, train_loader, config, device, epoch, optim=optim)
        va = run_epoch(model, valid_loader, config, device, epoch, vis_dir=str(exp_dir / "visualizations_valid"))
        scheduler.step(va["cls"])

        for key, value in tr.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in va.items():
            writer.add_scalar(f"valid/{key}", value, epoch)
        print(f"epoch {epoch}/{config.epochs}  train loss {tr['total']:.4f} acc {tr['acc']:.2f}  "
              f"valid loss {va['cls']:.4f} acc {va['acc']:.2f}  lr {optim.param_groups[0]['lr']:.2e}")

        if va["acc"] > best_acc or (va["acc"] == best_acc and va["cls"] < best_loss):
            best_acc, best_loss, bad_epochs = va["acc"], va["cls"], 0
            torch.save(model.state_dict(), exp_dir / "model_best.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stopping_patience:
                print(f"early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), exp_dir / "model_last.pth")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(ExperimentationConfig.parse_obj(json.loads(Path(args.config).read_text())))
