# BT4012 Credit Card Fraud Detection

Course project for **BT4012 Fraud Analytics, AY2026/2027 Semester 1, Group 17**.

## Project objective

Build a reproducible binary-classification pipeline that flags suspicious credit-card
transactions while controlling the cost of missed fraud and unnecessary investigations.
The project focuses on three fraud-analytics challenges:

1. severe class imbalance;
2. time-dependent fraud patterns and data leakage; and
3. the trade-off between fraud recall and false alerts.

## Current scope and decisions

### Datasets

Both datasets are used. The full pipeline is run independently on each, and results are
compared side by side. See [`docs/dataset_decision.md`](docs/dataset_decision.md).

| Dataset | Role | Evaluation |
|---|---|---|
| **Sparkov** (synthetic, CC0) | Primary | Chronological split within `fraudTrain.csv`; untouched `fraudTest.csv` as final test |
| **IEEE-CIS / Vesta** (real, anonymised) | Real-data replication | Chronological split within the labelled training file; the unlabelled competition test is not used |

### Models

An incremental M0-M4 ladder, with a Dummy classifier as reference:

| Stage | Change from previous stage |
|---|---|
| M0 | LR, RF and CatBoost base classifiers |
| M1 | Cost-sensitive class weights |
| M2 | Sliding-window retraining |
| M3 | Semi-supervised use of recent unlabelled transactions (algorithm to be named) |
| M4 | Unsupervised anomaly score (integration rule to be decided) |

The strongest of LR, RF and CatBoost after M2 is carried into M3 and M4. See
[`docs/experiment_plan.md`](docs/experiment_plan.md).

### Evaluation

- **Primary metric:** PR-AUC.
- **Also reported:** recall, precision, F1, ROC-AUC, confusion matrices and Precision@k.
  The unit of `k` (transactions or cards) must match what the model scores.
- **Accuracy:** not a primary metric, because the non-fraud class dominates the data.

### Checks required by the TA feedback

- **Concept-drift check:** compare KS/PSI with model performance across chronological
  windows.
- **Label-delay simulation:** hide the labels of every transaction in the latest seven-day
  window, not only the fraud labels.
- **Card-history ablation:** report results with and without card-history features. On
  IEEE-CIS this tests whether the model only learns that a card was flagged before.

### Current code status

`src/` currently implements:

- the Sparkov loader and the chronological split;
- the evaluation helpers and the Dummy/LR baseline;
- leakage-safe transaction and card-history features for both datasets
  ([`docs/features.md`](docs/features.md));
- the transaction-only versus history ablation script.

Still to be added: the IEEE-CIS card reconstruction (a provisional key is used until
then), and M1-M4.

## Repository structure

```text
.
├── configs/                 # Human-readable experiment configuration
├── data/                    # Data instructions; raw data is never committed
├── docs/                    # Dataset decision, experiment and report plans
├── notebooks/               # Numbered exploratory notebooks
├── reports/figures/         # Version-controlled final figures
├── scripts/                 # Data acquisition helpers
├── src/bt4012_fraud/        # Reusable data, metric and model code
└── tests/                   # Automated checks for leakage-sensitive utilities
```

## Reproduce the baseline

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
bash scripts/download_sparkov.sh
# IEEE-CIS: accept the competition rules on Kaggle, then place train_transaction.csv and
# train_identity.csv in data/raw/ (see data/README.md)
python -m bt4012_fraud.baseline \
  --train-csv data/raw/fraudTrain.csv \
  --output-dir artifacts/baseline
```

The baseline command currently runs on Sparkov only. It sorts transactions by time, trains only on earlier observations,
validates on later observations, and writes metrics plus fitted pipelines to
`artifacts/baseline/`. The official test file is intentionally excluded from this step.

To run the feature-engineering ablation (transaction-only versus card-history features):

```bash
make features
```

See [`docs/features.md`](docs/features.md) for the feature list, leakage rules and the
IEEE-CIS command.

To install the optional boosting libraries:

```bash
python -m pip install -e ".[boosting,dev]"
```

## Dataset and privacy policy

Sparkov is a synthetic dataset released under CC0. It contains realistic-looking names,
addresses and card numbers. These direct identifiers are excluded from model inputs.
Card identifiers may later be used only to construct past-behaviour features, after which
the raw identifier must be removed.

Do not commit raw data, Kaggle credentials, fitted models, or personal information. See
[`data/README.md`](data/README.md) and
[`docs/dataset_decision.md`](docs/dataset_decision.md).

## Team workflow

- Create one GitHub issue for each task and assign an owner.
- Work on a short feature branch and open a pull request into `main`.
- Keep notebooks exploratory; move reusable logic into `src/`.
- Record important experiments, random seeds and data versions.
- Do not tune models against the final test set.
- Use `random_state=4012` and fit preprocessing on training data only.

## Course deliverables

The repository is organised around the marking criteria: problem formulation, data and
preprocessing, justified methodology, experimental design, results and applicability,
limitations, and reproducibility. The final 8-12 page report must include the GitHub URL
and the repository must be accessible to the teaching team by the submission deadline.
A rubric-aligned writing plan is in [`docs/report_outline.md`](docs/report_outline.md).

## Team

- Group number: **17**
- Yang Tonghang (Gary) — GitHub: `xman666888` — feature engineering
- Phang Xin Yi — GitHub: `xin-yi-phang` — data audit and exploratory data analysis
- Xiang Qiuyu — GitHub: `xiang-qiuyu` — M3 algorithm proposal and M4 decision
- Lycia Yeo Ying Xuan — GitHub: `lyciayeo` — IEEE-CIS card-ID reconstruction

The current internal checkpoint is **17 October 2026**, with the next team meeting at
**9:00 pm**. See [`docs/team_plan.md`](docs/team_plan.md) for the latest work split and
consultation-action checklist.

## License and attribution

Original project code is released under the MIT License. Dataset terms remain governed by
their respective providers and are not covered by this repository's license.
