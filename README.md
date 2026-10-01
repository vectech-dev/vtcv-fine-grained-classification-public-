# TaxoAttention: fine-grained mosquito classification

Code for *TaxoAttention: Fusing Semantic and Hierarchical Taxonomic Models with Attention-Guided Augmentation for Robust Mosquito Identification*.

## Install

```bash
pip install -e .
```

A CUDA GPU is expected (`"device": "cuda"` in the configs).

## Data

MosquitoTax-200 (white-balanced and masked images, labels, specimen source, and the official split) is on Zenodo: DOI (https://zenodo.org/records/23043581)

The datasheet CSV needs the columns `Id` (image path, relative to `data_root`), `y` (class 0-31), `Split` (`Train` / `Valid` / `Test`) and `Species_Name`. Optional: `Genus` (otherwise the first word of `Species_Name`), `Sex` (used for sex-aware positives), `Mask_Path`.

Set `datasheet_path` and `data_root` in the configs under `config_examples/paper/`.

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

| Exp | Aug | Contrast (`w_div > 0`) | Taxonomic | k |
|---|---|---|---|---|
| 0 | | | | 2 |
| 1 | ✓ | | | 2 |
| 2 | ✓ | | ✓ | 2 |
| 3 | | ✓ | | 2 |
| 4 | ✓ | ✓ | | 2 |
| 5 | | ✓ | ✓ | 2 |
| 6 | ✓ | ✓ | ✓ | 2 |
| 7 | | | ✓ | 2 |
| 8 | ✓ | ✓ | ✓ | 1 |
| 9 | ✓ | ✓ | ✓ | 3 |

TensorBoard logs are written to `runs/`.
