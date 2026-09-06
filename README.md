# BT4012 Credit Card Fraud Detection

Course project for **BT4012 Fraud Analytics, AY2026/2027 Semester 1**.

## Project objective

Build a reproducible binary-classification pipeline that flags suspicious credit-card
transactions while controlling the cost of missed fraud and unnecessary investigations.
The project focuses on three fraud-analytics challenges:

1. severe class imbalance;
2. time-dependent fraud patterns and data leakage; and
3. the trade-off between fraud recall and false alerts.

## Current scope and decisions

- **Primary dataset:** Sparkov simulated credit-card transactions.
- **Training/validation:** chronological split within `fraudTrain.csv`.
- **Final in-domain test:** the untouched `fraudTest.csv` file.
- **Optional external robustness test:** a compatible subset generated with the
  Fraud Detection Handbook simulator or sampled from IBM TabFormer. External results
  will be reported separately because they represent distribution shift.
- **Baselines:** Dummy Classifier and Logistic Regression.
- **Candidate extensions:** Random Forest, XGBoost, LightGBM and CatBoost.
- **Primary evaluation metric:** PR-AUC. Recall, precision, F1, ROC-AUC, confusion
  matrices and threshold/cost analysis will also be reported.

Logistic Regression is treated as the linear baseline. Accuracy is not a primary metric
because the non-fraud class dominates the data.

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
python -m bt4012_fraud.baseline \
  --train-csv data/raw/fraudTrain.csv \
  --output-dir artifacts/baseline
```

The baseline command sorts transactions by time, trains only on earlier observations,
validates on later observations, and writes metrics plus fitted pipelines to
`artifacts/baseline/`. The official test file is intentionally excluded from this step.

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
- Add any external code, public notebook or generative-AI assistance to
  [`docs/attribution_log.md`](docs/attribution_log.md).

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete collaboration rules.

## Course deliverables

The repository is organised around the marking criteria: problem formulation, data and
preprocessing, justified methodology, experimental design, results and applicability,
limitations, and reproducibility. The final 8-12 page report must include the GitHub URL
and the repository must be accessible to the teaching team by the submission deadline.
A rubric-aligned writing plan is in [`docs/report_outline.md`](docs/report_outline.md).

## Team

- Group number: **17**
- Members: **to be added after invitations are accepted**

## License and attribution

Original project code is released under the MIT License. Dataset terms remain governed by
their respective providers and are not covered by this repository's license.
