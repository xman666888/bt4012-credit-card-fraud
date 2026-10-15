# Dataset decision record

Status: **Two datasets confirmed** (team meeting, 10 October 2026; consistent with the
proposal submitted on 20 September 2026 and the TA feedback of 29 September 2026).

| Dataset | Role | Rows used | Source type |
|---|---|---:|---|
| Sparkov | **Primary dataset** | 1,852,394 (train 1,296,675 + test 555,719) | Synthetic, CC0 |
| IEEE-CIS (Vesta) | **Real-data replication** | 590,540 labelled training transactions | Real, anonymised |

The complete M0-M4 modelling and evaluation procedure is run **independently** on each
dataset. The aim is to test whether the incremental gains observed on simulated data also
hold on real transactions. Results are reported side by side. A model trained on one
dataset is **never** scored on the other, because the two datasets have different fields,
generators and label processes.

## Why these two datasets

**Sparkov** has interpretable columns, a calendar timestamp, a true card number (`cc_num`)
and an official chronological train/test split. Card-level history features can therefore
be built directly, without reconstruction.

**IEEE-CIS** contains real card-not-present transactions provided by Vesta. Because Sparkov
is simulated, IEEE-CIS is used to check whether findings on simulated data carry over to
real data. It has three known complications:

- **No card or customer identifier.** `card1`-`card6` do not form a verified unique ID, so a
  card/entity ID has to be reconstructed. Lycia owns this; the method and its limitations
  must be documented.
- **Label propagation.** Once an account is reported as fraud, its later transactions are
  also labelled fraud. Card-history features may therefore learn "this card was flagged
  before" rather than genuine fraud patterns. Every IEEE-CIS result is reported both with
  and without card-history features.
- **Anonymised features and relative time.** Most `C`, `D`, `M`, `V` and `id_*` features
  have undisclosed meanings. `TransactionDT` is elapsed seconds from an unknown reference
  point (about 182 days), not a calendar date.

## Split policy

### Sparkov

- `fraudTrain.csv` is split chronologically into training and validation.
- `fraudTest.csv` stays untouched until features, model family, hyperparameters and
  threshold are all frozen.

### IEEE-CIS

- Only `train_transaction.csv` and `train_identity.csv` are used. The two files are
  left-joined on `TransactionID`; about 24.4% of transactions have an identity record.
- The official competition test files have **no `isFraud` labels**, so they are not used.
- The labelled training file is split chronologically by `TransactionDT` into train,
  validation and a final held-out test block. The exact boundaries are recorded in
  `configs/project.yaml` once agreed.

### Both datasets

- No resampling is applied to validation or test data.
- Every transformer and resampler is fitted on the training block only.

## Rejected or backup candidates

| Candidate | Decision | Main reason |
|---|---|---|
| Fraud Detection Handbook simulator | Backup | Reproducible but simpler synthetic features |
| IBM TabFormer | Backup | Rich and interpretable, but 24M rows require careful sampling |
| Feedzai BAF | Rejected | Models account-opening fraud, not card transactions |
| ULB credit-card fraud | Rejected | Main variables are PCA-anonymised; no usable time or entity |

## Access and licence notes

- Sparkov is listed as CC0 by Amazon's Fraud Dataset Benchmark.
- IEEE-CIS is distributed under the Kaggle competition rules. Each member must accept the
  competition rules to download it.
- Neither dataset may be committed or redistributed from this repository.

## Sources

- Sparkov: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- IEEE-CIS Fraud Detection: https://www.kaggle.com/competitions/ieee-fraud-detection
- Amazon Fraud Dataset Benchmark: https://github.com/amazon-science/fraud-dataset-benchmark
- Fraud Detection Handbook: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
- IBM TabFormer: https://github.com/IBM/TabFormer
