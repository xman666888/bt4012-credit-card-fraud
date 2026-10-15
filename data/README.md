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

## Replication dataset: IEEE-CIS (Vesta)

IEEE-CIS Fraud Detection competition data:
https://www.kaggle.com/competitions/ieee-fraud-detection

Access requires accepting the competition rules on Kaggle. Only the **labelled training
files** are used:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

The official `test_*.csv` files have no `isFraud` labels and are not used.

| File | Rows | Columns | Notes |
|---|---:|---:|---|
| `train_transaction.csv` | 590,540 | 394 | 20,663 fraud (3.499%); `TransactionDT` spans about 182 days |
| `train_identity.csv` | 144,233 | 41 | Matches 24.42% of transactions on `TransactionID` |

Figures are from the data audit in `notebooks/01_data_audit.ipynb` (PR #1).

Two caveats apply when modelling this dataset:

- `TransactionDT` is elapsed seconds from an undisclosed reference point, not a calendar
  date.
- There is no card identifier. See `docs/dataset_decision.md` for card reconstruction and
  label-propagation caveats.

Team copy (shared in the group chat on 2026-10-13) checksums:

```text
3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642  train_transaction.csv
b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c  train_identity.csv
```
