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

Before final experiments, record the download date, file sizes, row counts and SHA-256
checksums in this file. Do not redistribute the data from this repository.
