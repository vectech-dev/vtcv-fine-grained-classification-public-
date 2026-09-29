import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def pad_to_square(img, fill=None):
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    if fill is None:
        edge = np.concatenate([arr[:5].reshape(-1, *arr.shape[2:]), arr[-5:].reshape(-1, *arr.shape[2:])])
        fill = tuple(int(v) for v in np.atleast_1d(edge.mean(0)))
        fill = fill[0] if arr.ndim == 2 else fill
    canvas = Image.new(img.mode, (size, size), fill)
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


class MosquitoDataset(Dataset):
    """Rows of the datasheet with Split == mode.

    Needed columns: Id (image path), y (class), Split, Species_Name.
    Optional: Genus_Name, Sex_Name, Mask_Path, Specimen_Id.
    """

    def __init__(self, config, mode, augment=None):
        df = pd.read_csv(config.datasheet_path, dtype={"Id": str, "Specimen_Id": str})
        df = df[(df["Split"] == mode) & (df["y"] >= 0)].reset_index(drop=True)
        # one genus per class: majority of Genus_Name, or the first word of Species_Name
        genus = df["Genus_Name"] if "Genus_Name" in df.columns else df["Species_Name"].str.split(r"[_ ]").str[0]
        genus = genus.astype(str).str.lower()
        df["Genus"] = df["y"].map(genus.groupby(df["y"]).agg(lambda g: g.mode().iloc[0]))
        sex = df["Sex_Name"] if "Sex_Name" in df.columns else df.get("Sex", pd.Series("", index=df.index))
        sex = sex.astype(str).str.lower()
        df["Sex"] = sex.where(sex.isin(["female", "male"]), "unknown")
        self.df = df
        self.config = config
        self.mode = mode
        self.augment = (mode == "Train" and config.train_flips) if augment is None else augment
        self.use_mask = config.use_mask and "Mask_Path" in df.columns

    def __len__(self):
        return len(self.df)

    def _path(self, p):
        if os.path.isabs(p) or not self.config.data_root:
            return p
        return os.path.join(self.config.data_root, p)

    def load_image(self, idx):
        row = self.df.iloc[idx]
        size = self.config.imsize
        img = pad_to_square(Image.open(self._path(row["Id"])).convert("RGB"))
        img = np.array(img.resize((size, size), Image.BILINEAR))

        if self.use_mask and isinstance(row["Mask_Path"], str):
            mask = Image.open(self._path(row["Mask_Path"])).convert("L")
            mask = np.array(pad_to_square(mask, fill=255).resize((size, size), Image.NEAREST))
            # mask files mark the background in white, same as the original loader
            img[mask >= 128] = self.config.mask_color

        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.augment:
            if random.random() < 0.5:
                x = x.flip(2)
            if random.random() < 0.5:
                x = x.flip(1)
            x = torch.rot90(x, random.randint(0, 3), dims=(1, 2))
        return x

    def __getitem__(self, idx):
        return self.load_image(idx), int(self.df.at[idx, "y"])
