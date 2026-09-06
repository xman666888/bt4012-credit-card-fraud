# Contributing

## Branch and pull-request workflow

1. Create or claim a GitHub issue with a clear deliverable.
2. Create a branch such as `feature/temporal-split` or `analysis/eda`.
3. Make small, descriptive commits.
4. Run `make test` and `make lint` before opening a pull request.
5. Request review from at least one teammate before merging into `main`.

## Reproducibility rules

- Use `random_state=4012` unless an experiment explicitly varies the seed.
- Fit imputers, encoders, scalers and resampling methods on training data only.
- Preserve chronological ordering for validation and testing.
- Record the feature set, split dates, hyperparameters and metrics for every final run.
- Keep the final test set untouched until model and threshold selection are complete.

## Data and privacy rules

- Never commit raw datasets, credentials or direct personal identifiers.
- Direct identifiers such as card number, name, street and transaction ID must not be
  model inputs.
- When an identifier is needed for sequential feature engineering, use it only for
  grouping, calculate features from past transactions, and remove it before fitting.

## Academic integrity and attribution

Record external code, public notebooks, copied or adapted snippets, and material
generative-AI assistance in `docs/attribution_log.md`. Contributors must understand and
be able to explain every submitted component.
