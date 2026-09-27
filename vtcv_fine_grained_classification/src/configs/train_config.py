import random

import numpy as np
import pandas as pd
import torch

from vtcv_fine_grained_classification.src.loader.dataset import MosquitoDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset(config, mode):
    return MosquitoDataset(config, mode)


def get_class_names(config):
    df = pd.read_csv(config.datasheet_path)
    df = df[(df["Split"] == "Train") & (df["y"] >= 0)]
    names = df.groupby("y")["Species_Name"].first()
    return [names[i] for i in range(config.num_classes)]
