# Data directory

Raw and processed datasets are intentionally excluded from Git.

## Primary dataset

Sparkov simulated credit-card transactions:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

The dataset is listed as CC0 by Amazon's Fraud Dataset Benchmark. Expected files:

```text
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

Run `bash scripts/download_sparkov.sh` after configuring Kaggle credentials, or download
the archive manually from the source page and extract both CSV files into `data/raw/`.

## Version record

Downloaded from Kaggle on 2026-09-06 using dataset slug
`kartik2112/fraud-detection`.

| File | Rows | Period | Fraud count | Fraud rate | SHA-256 |
|---|---:|---|---:|---:|---|
| `fraudTrain.csv` | 1,296,675 | 2019-01-01 to 2020-06-21 | 7,506 | 0.5789% | `fd7139200dbfcbed` |
| `fraudTest.csv` | 555,719 | 2020-06-21 to 2020-12-31 | 2,145 | 0.3860% | `12d553ab19440c75` |

The source files are chronologically contiguous: the test period starts immediately after
the training period. Do not redistribute the data from this repository.

The table shows shortened checksums for readability. Full SHA-256 values:

```text
fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d  fraudTrain.csv
12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0  fraudTest.csv
```
