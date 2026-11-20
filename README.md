# BT4012 Credit Card Fraud Detection

Course project for **BT4012 Fraud Analytics, AY2026/2027 Semester 1, Group 17**.

We build a chronological fraud-detection pipeline and test, stage by stage, whether common
techniques improve it on two datasets: **Sparkov** (simulated) and **IEEE-CIS** (real,
anonymised). The project addresses three challenges of fraud detection:

1. **Class imbalance**: fraud is 0.58% of Sparkov and 3.5% of IEEE-CIS transactions.
2. **Concept drift**: fraud patterns change over time, so a model trained once can decay.
3. **Verification latency**: fraud labels arrive late, simulated as a seven-day label delay.

## Model ladder

Every stage is run on both datasets with the same chronological split and metrics.

| Stage | What changes | Compared with |
|---|---|---|
| M0 | Logistic Regression, Random Forest, CatBoost and LightGBM, trained once | Random ranking |
| M1 | Balanced class weights (cost-sensitive learning) | M0 |
| M2 | Retraining every 7 days on all earlier days, with a 7-day label delay | Static model with the same delay |
| M3 | Pseudo-labels for the unlabelled most recent week, with reduced weights | M2 |
| M4 | Isolation Forest anomaly score added as a feature | Best previous stage |

Retraining settings, feature selection and M3 parameters are chosen by chronological
simulation inside the training period, never on the validation or test sets. The test sets
are used once, for the final models.

## Main results

Validation PR-AUC (CatBoost). The primary metric is PR-AUC; a random ranking scores about
0.006 on Sparkov and 0.039 on IEEE-CIS.

| Stage | Sparkov | IEEE-CIS |
|---|---:|---:|
| M0 baseline | 0.987 | 0.626 |
| M1 class weights, change vs M0 | +0.002 | -0.035 |
| M2 periodic retraining, change vs static | +0.001 | +0.053 |
| M3 pseudo-labelling, change vs M2 | +0.0008 | +0.0005 (mean of 3 seeds) |
| M4 anomaly score, change vs best previous stage | -0.003 | -0.004 |
| Final model | M1 CatBoost | Tuned M2 CatBoost |
| Final model, test PR-AUC | 0.974 | 0.622 (static model: 0.583) |

- Periodic retraining helps only on IEEE-CIS, where the data drifts. The decline of a static
  model follows the time since training rather than the PSI of each week (notebook 05d).
- Pseudo-labels and anomaly scores do not beat the best previous stage on either dataset.
  Class weights help slightly on Sparkov (+0.002) but lower PR-AUC on IEEE-CIS.
- The final models are M1 CatBoost for Sparkov and the tuned M2 CatBoost (115 features) for
  IEEE-CIS. A daily top-K alert rule chosen on validation catches 99.3% of Sparkov test fraud
  (K = 50) and 90.9% of IEEE-CIS test fraud (K = 750).

## Repository structure

```text
.
├── data/                 # Raw data (not committed), processed features and IEEE-CIS card IDs
├── docs/                 # Dataset decision and feature documentation
├── notebooks/            # All experiments, numbered in run order
├── reports/              # Feature ablation summaries and weekly drift tables
├── results/              # Saved model scores and result tables, so slow notebooks rerun without retraining
├── scripts/              # Data download helper
├── src/bt4012_fraud/     # Data loading, leakage-safe features, evaluation and baseline
└── tests/                # Tests for the leakage rules and evaluation helpers
```

## How to reproduce

### 1. Environment

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[boosting,dev]"
python -m pip install optuna
```

Some saved runs in `results/ieee/` were written with pandas 3. Notebook 08 falls back to
the CSV copies in `results/ieee/m2_tuning/` when a pickle cannot be read with pandas 2.

### 2. Data

The raw files are not committed. Place them in `data/raw/` (see [`data/README.md`](data/README.md)):

- **Sparkov**: `fraudTrain.csv` and `fraudTest.csv`. Run `bash scripts/download_sparkov.sh`
  with Kaggle credentials, or download them from
  https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **IEEE-CIS**: `train_transaction.csv` and `train_identity.csv` from
  https://www.kaggle.com/competitions/ieee-fraud-detection (accept the competition rules
  first). The official test files have no labels and are not used.

### 3. Notebooks

Run `03` and `04` first: they build the IEEE-CIS card IDs and the feature tables in
`data/processed/`, which are too large for GitHub. The other notebooks load the saved runs
in `results/` and `data/interim/`, so they rerun without retraining. Run them from the `notebooks/` folder.

| Notebook | Content |
|---|---|
| `01_data_audit` | Schema, missing values, time range and class balance |
| `02_exploratory_analysis` | Fraud patterns and early vs late comparisons |
| `03_ieee_card_reconstruction` | Card ID reconstruction for IEEE-CIS |
| `04_feature_engineering` | Feature tables of both datasets |
| `04b_temporal_drift_and_label_delay` | Fraud rate over time, weekly PSI/KS drift, covariate vs concept drift, and the share of fraud hidden by the label delay |
| `04c_drift_and_network_diagnostics` | Adversarial validation, decay of a model trained once, and fraud clusters on shared entities |
| `05_ieee_m0_m2` | IEEE-CIS M0 to M2 with LR, RF and CatBoost, card-history checks and SHAP |
| `05_sparkov_m0_m2` | Sparkov M0 to M2 with all four model families |
| `05b_ieee_lightgbm_m0_m1` | IEEE-CIS LightGBM M0 and M1, and a training-budget check |
| `05c_ieee_lightgbm_m2` | IEEE-CIS LightGBM M2 |
| `05d_ieee_drift_vs_performance` | Drift vs model performance by validation week |
| `06_ieee_m2_catboost_tuning` | Feature selection and Optuna tuning of the IEEE-CIS M2 CatBoost |
| `06_sparkov_m3_lightgbm_pseudolabeling` | Early Sparkov M3 experiment |
| `07_ieee_m3_pseudo_labelling` | IEEE-CIS M3 |
| `07_sparkov_m3_catboost_8x3` | Sparkov M3 with CatBoost |
| `08_ieee_m4_isolation_forest` | IEEE-CIS M4: six anomaly score variants |
| `08_sparkov_m4_catboost` | Sparkov M4: Isolation Forest score |
| `09_sparkov_ieee_comparison` | Why the results differ between the two datasets |
| `10_ieee_final_test` | IEEE-CIS final model on the test set and the daily alert rule |
| `10_sparkov_final_test_catboost` | Sparkov final model on the test set and the daily alert rule |

`notebooks/archive/` keeps the earlier LightGBM versions of the Sparkov M3 to final test
notebooks. `notebooks/README.md` lists the saved-run folders used by each notebook.

### 4. Tests

```bash
pytest -q
```

The tests check the leakage rules of the card-history features: only strictly earlier
transactions are used, a transaction never enters its own history, and future rows cannot
change past features. See [`docs/features.md`](docs/features.md).

## Team

| Member | GitHub | Main contributions |
|---|---|---|
| Phang Xin Yi | `xin-yi-phang` | Data audit, EDA, feature engineering notebook, early Sparkov M3 |
| Lycia Yeo Ying Xuan | `lyciayeo` | IEEE-CIS card reconstruction, IEEE-CIS M0 to M3, tuning, dataset comparison, final test |
| Xiang Qiuyu | `xiang-qiuyu` | Sparkov M0 to M4, Sparkov final test and threshold selection |
| Yang Tonghang (Gary) | `xman666888` | Feature module and tests, drift and label-delay analysis, IEEE-CIS LightGBM, IEEE-CIS M4 |

## Data and licence

Sparkov is synthetic and released under CC0; IEEE-CIS follows the Kaggle competition
rules and is not redistributed here. Direct identifiers (names, addresses, card numbers)
are never used as model inputs. Project code is released under the MIT License.
