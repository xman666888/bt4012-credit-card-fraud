# Feature ablation: Sparkov

Run on 15 October 2026 with `make features` (seed 4012).

- **Data:** `fraudTrain.csv`, SHA-256 `fd713920…` (matches `data/README.md`).
- **Not used:** `fraudTest.csv`.
- **Split:** chronological, last 20% for validation.

| Block | Rows | Period | Fraud rate |
|---|---:|---|---:|
| Train | 1,037,340 | 2019-01-01 to 2020-03-06 07:15 | 0.575% |
| Validation | 259,335 | 2020-03-06 07:16 to 2020-06-21 | 0.593% |

## Result

Adding card-history features lifts Logistic Regression PR-AUC from **0.289 to 0.663**
on the validation block. Precision, recall and F1 below use the default 0.5 threshold;
threshold selection is a later stage.

| Feature set | Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transaction-only | Dummy (prior) | 0.0059 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1,538 |
| Transaction-only | Logistic Regression | 0.2892 | 0.9244 | 0.0305 | 0.8485 | 0.0589 | 1,305 | 41,503 | 233 |
| + card history | Dummy (prior) | 0.0059 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1,538 |
| + card history | Logistic Regression | **0.6632** | **0.9868** | 0.1176 | 0.9382 | 0.2090 | 1,443 | 10,829 | 95 |

At the 0.5 threshold, history features cut false alerts from 41,503 to 10,829 (-74%).
Over the same window, missed frauds fall from 233 to 95.

## Why history helps, and the caveat

The strongest single features are short-window spending: past-24h spend sum and mean,
then past-1h sum. Each was measured alone on the validation block:

| Feature | Univariate PR-AUC |
|---|---:|
| `amt_sum_24h` | 0.247 |
| `amt_mean_24h` | 0.239 |
| `amt_sum_1h` | 0.222 |
| `log_amt` (transaction-only) | 0.190 |

Sparkov fraud arrives in bursts. Across the training file, 762 cards have fraud, with a
median of 10 fraudulent transactions per card, spread over a median of 1.9 days.

The history features use no labels and only strictly earlier transactions, so this is
not label leakage. However, the burst pattern may be partly an artefact of the
simulator. This is why the TA asked for the result to be checked on real data: repeat
this ablation on IEEE-CIS once card reconstruction is merged. On IEEE-CIS, also compare
with and without history to test the label-propagation concern.

Full metrics, feature lists and split boundaries: `sparkov_metrics.json`.
