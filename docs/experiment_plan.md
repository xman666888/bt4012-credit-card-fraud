# Experiment plan

This plan follows the proposal submitted on 20 September 2026 and incorporates the TA
feedback of 29 September 2026. Every experiment below is run separately on Sparkov
(primary) and IEEE-CIS (real-data replication). See `dataset_decision.md`.

## Research questions

1. How much does each model increment (M0 to M4) improve fraud detection under
   chronological evaluation?
2. Do feature-distribution changes over time (KS/PSI) correspond to an actual drop in model
   performance?
3. How does a seven-day verification latency affect training and detection, and does using
   recent unlabelled transactions (M3) recover part of the loss?
4. Do card-history features improve detection, or do they mainly exploit label propagation?
5. Do the improvements found on simulated data (Sparkov) generalise to real data (IEEE-CIS)?

## Model ladder

| Stage | Model | Purpose |
|---|---|---|
| Reference | Dummy (prior) | Lower bound for every metric |
| M0 | LR, Random Forest, CatBoost | Base classifiers; LR is the linear baseline |
| M1 | M0 + class weights | Cost-sensitive learning for class imbalance |
| M2 | M1 + periodic sliding-window retraining | Adaptation to concept drift |
| M3 | M2 + a named semi-supervised algorithm | Uses recent unlabelled transactions (verification latency) |
| M4 | M3 + unsupervised anomaly score | Catches novel patterns (e.g. Isolation Forest or autoencoder) |

LR, RF and CatBoost are compared at M0-M2. The strongest family is carried into M3 and M4.

**Open decisions (owner: Qiu Yu)**

- **M3:** name and justify a specific semi-supervised algorithm, for example
  self-training/pseudo-labelling or label propagation. Dal Pozzolo et al. (2018) must not
  be described as semi-supervised learning on unlabelled data.
- **M4:** state whether the anomaly score is an input feature to the classifier or is
  combined with the classifier's output score.

## Experimental stages

1. **EDA.** Validate schema, missingness, class balance and time coverage for both
   datasets (PR #1).
2. **IEEE-CIS card reconstruction.** Reconstruct a card/entity ID and document its
   assumptions and failure cases (Lycia). Sparkov uses `cc_num` directly.
3. **Feature pipeline (Gary).** Build a leakage-safe pipeline in `src/` with two feature
   groups:
   - *transaction-only* features;
   - *card-history* features computed from strictly earlier transactions.
4. **Reference and M0.** Train Dummy and M0 under identical splits and metrics.
5. **M1-M2.** Add cost-sensitive learning, then sliding-window retraining.
6. **M3-M4.** Add the semi-supervised and anomaly-detection extensions to the strongest family.
7. **Drift check.** Measure KS/PSI and model performance across the same chronological
   windows. This is required especially on Sparkov, which is simulated.
8. **Label delay.** Simulate a seven-day delay by hiding the labels of **all**
   transactions in the latest seven days. Then vary the delay length.
9. **Threshold.** Select a threshold on validation data using a documented
   investigation-cost assumption.
10. **Final test.** Freeze the pipeline and evaluate once on each dataset's held-out test
    block. Compare the two datasets.

## Leakage controls

- Split by time before fitting any transformer or resampler.
- Compute each historical feature from strictly earlier transactions. Sort by time and
  shift, so a transaction never contributes to its own history.
- Report IEEE-CIS results with and without card-history features (label-propagation check).
- Exclude names, addresses, card numbers and transaction IDs from model inputs. Raw or
  reconstructed card IDs may be used for grouping only, then dropped.
- Never resample validation or test data.
- Never select hyperparameters or thresholds using the final test result.
- In the label-delay experiment, recent unlabelled transactions are never implicitly
  treated as legitimate.

## Metrics

- **Primary:** PR-AUC.
- **Also reported:** Recall, Precision, F1, ROC-AUC, confusion matrix, Precision@k.
- **Precision@k:** the unit must match what the model scores. Either define `k` as a
  number of transactions per day, or aggregate transaction scores to card level before
  using cards. This is a team decision and is recorded in `configs/project.yaml`.
- **Accuracy:** descriptive only.

## Planned ablations

- M0 to M4, each increment against the previous stage;
- transaction-only versus transaction plus card-history features;
- with versus without reconstructed card ID (IEEE-CIS);
- default threshold versus validation-selected threshold;
- label delay of 0, 7 and longer periods;
- Sparkov versus IEEE-CIS: does the ranking of increments replicate?
