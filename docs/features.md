# Feature engineering

Owner: Gary. Code: `src/bt4012_fraud/features.py`. Tests: `tests/test_features.py`.

Features come in two groups so the card-history ablation can be run directly:

- **transaction-only** features describe the current transaction alone;
- **history** features summarise the same entity's **strictly earlier** transactions.

## Leakage rules enforced in code and tests

1. **Strictly past windows.** History for a transaction at time `t` uses only the same
   entity's transactions before `t`.
   - Rolling windows are half-open, `[t - window, t)`.
   - Transactions with the identical timestamp are excluded.
   - A brute-force test recomputes every window value and must match exactly.
2. **The current transaction never enters its own history.** A test changes one
   transaction's amount and checks that its history features do not move.
3. **Future rows cannot change past features.** A test computes features on a truncated
   file and on the full file; the shared rows must be identical.
4. **No labels.** History features never use `is_fraud`/`isFraud`, so they stay valid in
   the seven-day label-delay experiment. They also cannot directly encode "this card was
   flagged before".
5. **Identifiers are grouping keys only.** `cc_num`, names, addresses and transaction IDs
   are never returned as features (tested).
6. **Missing entity means missing history.** IEEE-CIS rows with a missing key get
   missing history values. They are not pooled into one artificial card.

Because of rules 1-3, history features can be computed on a whole file before the
chronological split: validation rows only see earlier transactions, which is exactly
what would be known at scoring time.

## Transaction-only features

| Feature | Sparkov | IEEE-CIS | Notes |
|---|:-:|:-:|---|
| `log_amt` | ✓ | ✓ | `log1p` of amount |
| `amt_cents` | | ✓ | Fractional part of the amount |
| `hour`, `day_of_week` | ✓ | ✓ | IEEE-CIS: relative cycles from `TransactionDT`, not calendar values |
| `is_weekend`, `is_night` | ✓ | | Night is 22:00-03:59 |
| `age` | ✓ | | Age at transaction time, from `dob` |
| `distance_km` | ✓ | | Cardholder home to merchant (Haversine) |
| `log_city_pop` | ✓ | | |
| `category`, `gender`, `state` | ✓ | | One-hot encoded |
| `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType` | | ✓ | One-hot encoded |
| `has_identity` | | ✓ | Whether an identity record exists |
| `C1`-`C14`, `D1`-`D15`, `M1`-`M9` | | optional | Anonymised; enabled with `--include-anonymised` |

## History features (per entity, strictly past)

| Feature | Meaning |
|---|---|
| `secs_since_prev` | Seconds since the entity's previous transaction |
| `prior_txn_count`, `is_first_txn` | Number of earlier transactions; whether this is the first |
| `amt_to_prior_mean` | Current amount ÷ mean of all earlier amounts |
| `amt_prior_zscore` | Current amount's z-score against earlier amounts |
| `txn_count_{w}`, `amt_sum_{w}`, `amt_mean_{w}` | Count, total and mean amount over the past 1h / 24h / 7d / 30d |
| `is_new_category` | First time this entity uses this category (Sparkov) or `ProductCD` (IEEE-CIS) |
| `km_from_prev_txn`, `implied_speed_kmh` | Merchant-location jump since the previous transaction, and the speed it implies (Sparkov) |

## Entity keys

- **Sparkov:** the true card number `cc_num`.
- **IEEE-CIS:** pending Lycia's card reconstruction. Until then the experiment script
  uses a **provisional** key (`card1` + `addr1` by default, `--entity-columns`). Any
  result using it is labelled `provisional:` in the output and must not be reported as
  final.
  - When the reconstruction is merged, pass its entity Series to
    `ieee_history_features(frame, entity)`.
  - No other change is needed.

## Ablation experiment

```bash
# Sparkov: uses fraudTrain.csv only; fraudTest.csv is not read
python -m bt4012_fraud.feature_ablation sparkov \
  --train-csv data/raw/fraudTrain.csv \
  --output-dir artifacts/feature_ablation/sparkov

# IEEE-CIS: drops the latest 20% as an untouched held-out block, then validates on the
# latest 20% of the remainder
python -m bt4012_fraud.feature_ablation ieee \
  --transaction-csv data/raw/train_transaction.csv \
  --identity-csv data/raw/train_identity.csv \
  --output-dir artifacts/feature_ablation/ieee
```

Each run writes two files:

- `metrics.json`: feature lists, split boundaries, metrics and timings;
- `results.md`: a table ready for the report.

The models are a Dummy reference and a class-weighted Logistic Regression, the M0 linear
baseline. Heavy-tailed history features are signed-log transformed before scaling.
Imputers and encoders are fitted on the training block only.
