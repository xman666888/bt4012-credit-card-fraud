# Feature ablation: IEEE-CIS (provisional entity key)

Run on 15 October 2026 (seed 4012).

**Data:** `train_transaction.csv` (590,540 rows) left-joined with `train_identity.csv` on
`TransactionID`.

| File | SHA-256 |
|---|---|
| `train_transaction.csv` | `3a5c83ab…` |
| `train_identity.csv` | `b63c725d…` |

The full checksums are in `ieee_cis_metrics.json` and `data/README.md`.

**Entity key is provisional.** History features group by `card1+addr1` until the
team's card reconstruction is merged. These results must not be reported as final.

## Split

Day = `TransactionDT` / 86,400, relative to an unknown reference point.

| Block | Rows | Days | Fraud rate | Use |
|---|---:|---|---:|---|
| Train | 377,945 | 1 to 109.6 | 3.41% | Model fitting |
| Validation | 94,487 | 109.6 to 141.1 | 3.92% | Scored here |
| Held-out test | 118,108 | 141.1 to 183 | — | **Untouched** for the final evaluation |

## Result

Unlike Sparkov, card-history features add almost nothing to Logistic Regression on
IEEE-CIS: PR-AUC moves from **0.170 to 0.171**.

| Feature set | Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transaction-only | Dummy (prior) | 0.0392 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 3,704 |
| Transaction-only | Logistic Regression | 0.1704 | 0.8012 | 0.1064 | 0.7006 | 0.1847 | 2,595 | 21,802 | 1,109 |
| + card history | Dummy (prior) | 0.0392 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 3,704 |
| + card history | Logistic Regression | 0.1714 | 0.8053 | 0.0959 | 0.7468 | 0.1699 | 2,766 | 26,090 | 938 |

Precision, recall and F1 use the default 0.5 threshold. Transaction-only features here
exclude the anonymised `C`/`D`/`M` columns (`--include-anonymised` adds them).

## Sensitivity to the entity key

The same ablation was repeated with two finer provisional keys. This is a quick check to
inform the card-reconstruction work, not a replacement for it.

`first_use_day` = floor(`TransactionDT` / 86,400 − `D1`). It is a common community
heuristic that treats `D1` as days since the card was first used.

| Provisional key | Entities | Rows covered | Median txns per entity | LR PR-AUC, transaction-only | LR PR-AUC, + history |
|---|---:|---:|---:|---:|---:|
| `card1+addr1` | 37,531 | 88.9% | 2 | 0.170 | 0.171 |
| `card1+addr1+P_emaildomain` | 73,407 | 73.3% | 2 | 0.170 | 0.171 |
| `card1+addr1+first_use_day` | 199,070 | 88.7% | 1 | 0.170 | **0.184** |

## Reading

- **The Sparkov gain does not replicate.** On Sparkov, history raised PR-AUC from 0.289 to
  0.663; on IEEE-CIS the gain is at most +0.014.
  - This is consistent with the concern that Sparkov's burst-shaped fraud is partly a
    simulator artefact.
  - It may also reflect weaker entity keys on IEEE-CIS.
- **Key quality matters.** The finer `D1`-based key gives the only noticeable gain. A
  better card reconstruction is likely to raise the history contribution further.
- **Little history is available.** Most provisional entities have one or two transactions,
  so many history values are empty or near-empty.
- **Next steps:**
  - re-run with Lycia's reconstructed ID;
  - try tree models (M0: RF and CatBoost), which may use history features non-linearly;
  - compare with the anonymised features enabled.

Full metrics, feature lists, split boundaries and key-sensitivity numbers:
`ieee_cis_metrics.json`.
