import argparse
import json
from pathlib import Path

from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig


def run():
    parser = argparse.ArgumentParser(prog="vtcv_fine_grained_classification")
    parser.add_argument("--config", help="training or pretraining config (json)")
    parser.add_argument("--exp-dir", help="experiment folder to test")
    parser.add_argument("--mode", choices=["pretrain", "train", "test"], default="train")
    args = parser.parse_args()

    if args.mode == "test":
        from vtcv_fine_grained_classification.test.test import test
        test(args.exp_dir)
        return

    config = ExperimentationConfig.parse_obj(json.loads(Path(args.config).read_text()))
    if args.mode == "pretrain":
        from vtcv_fine_grained_classification.train.pretrain_tmn import pretrain
        pretrain(config)
    else:
        from vtcv_fine_grained_classification.train.train import train
        train(config)


if __name__ == "__main__":
    run()
