# TaxoAttention: fine-grained mosquito classification

Code for *TaxoAttention: Fusing Semantic and Hierarchical Taxonomic Models with Attention-Guided Augmentation for Robust Mosquito Identification*.

## Install

```bash
pip install -e .
```

A CUDA GPU is expected (`"device": "cuda"` in the configs).

## Data

MosquitoTax-200 (images, CMU-Net masks, labels, specimen source, and the split) is on Zenodo under a CC BY-NC 4.0 licence: DOI [10.5281/zenodo.23043581](https://doi.org/10.5281/zenodo.23043581)

The datasheet CSV needs the columns `Id` (image path, relative to `data_root`), `y` (class 0-31), `Split` (`Train` / `Valid` / `Test`) and `Species_Name`. Optional: `Genus_Name` (otherwise the first word of `Species_Name`), `Sex_Name` (`female` / `male`, used for sex-aware positives), `Mask_Path`.

Set `datasheet_path` and `data_root` in the configs under `config_examples/paper/`.

### Split used in the paper

The Zenodo datasheet `MosquitoTax200_split_anonymized.csv` has 6,400 images. The paper uses 6,378 of them: **4,479 train / 957 valid / 942 test**. The other 22 images are left out for two reasons:

1. **4 images without a split label.** Four front-view *Ae. triseriatus* images from one tray have an empty `Split` value because of a formatting error in the source file. They are not used.
2. **18 test images of specimens that also appear in train or valid.** Most specimens have a front and a back image, and the split is made at the specimen level. For 21 specimens, however, the two images were put in different subsets (3 test-train, 15 test-valid, 3 train-valid). These specimens have `specimen_split_conflict = 1` in the datasheet. So that no test specimen is seen during training or model selection, we drop the 18 test images of these specimens. Their train and valid images are kept.

After this, the 942 test images come from 480 specimens, and no test specimen has an image in train or valid. Most classes keep 30 test images; eight classes have fewer (*Cx. tritaeniorhynchus* has 22).

To create the split file used in the paper from the Zenodo datasheet:

```python
import pandas as pd

df = pd.read_csv("MosquitoTax200_split_anonymized.csv")
df = df[df["Split"].isin(["Train", "Valid", "Test"])]                        # drops the 4 unlabeled images
df = df[~((df["Split"] == "Test") & (df["specimen_split_conflict"] == 1))]   # drops the 18 test images

counts = df["Split"].value_counts()
assert (counts["Train"], counts["Valid"], counts["Test"]) == (4479, 957, 942), counts
df.to_csv("data/mosquitotax200_split.csv", index=False)
```

The configs expect this file at `data/mosquitotax200_split.csv`.

## Reproduce the paper

1. Pretrain the taxonomic stream (Section III-A):

```bash
python -m vtcv_fine_grained_classification --mode pretrain --config config_examples/paper/tmn.json
```

2. Train the ablation runs of Table II (one config per experiment and seed):

```bash
for f in config_examples/paper/exp*_seed*.json; do
  python -m vtcv_fine_grained_classification --mode train --config "$f"
done
```

3. Test each run on the held-out test split:

```bash
for d in experiments/exp*_seed*; do
  python -m vtcv_fine_grained_classification --mode test --exp-dir "$d"
done
```

4. Tables II and III (mean and SD over seeds, paired bootstrap CI, McNemar with Holm correction):

```bash
python -m vtcv_fine_grained_classification.test.paper_stats --runs config_examples/paper/paper_runs.json
```

## Experiments

| Exp | Aug | Div (`w_div > 0`) | Taxonomic | k |
|---|---|---|---|---|
| 0 | | | | 2 |
| 1 | ✓ | | | 2 |
| 2 | ✓ | | ✓ | 2 |
| 3 | | ✓ | | 2 |
| 4 | ✓ | ✓ | | 2 |
| 5 | | ✓ | ✓ | 2 |
| 6 | ✓ | ✓ | ✓ | 2 |
| 7 | | | ✓ | 2 |
| 8 | ✓ | ✓* | ✓ | 1 |
| 9 | ✓ | ✓ | ✓ | 3 |

\* Exp 8 sets `w_div > 0`, but with k = 1 there is only one attention map, so the diversity term is zero (Table II marks it as ×).

TensorBoard logs are written to `runs/`.
