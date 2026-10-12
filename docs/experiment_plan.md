# Experiment plan

## Research questions

1. How much do non-linear ensemble models improve PR-AUC over a Logistic Regression
   baseline under chronological validation?
2. How does the alert threshold change fraud recall and the volume of false alerts?
3. Do past-behaviour features improve detection without using future information?
4. Do feature-distribution changes correspond to an actual decline in model performance
   over time, and how stable are results under dataset distribution shift?
5. How does a seven-day label delay affect model training and operational evaluation?

## Experimental stages

1. Validate schema, missingness, class balance and chronological coverage through EDA.
2. Reconstruct a defensible card identifier and document its limitations.
3. Train Dummy and Logistic Regression baselines using only transaction-level features.
4. Add Random Forest and selected boosting models under the same split and metrics.
5. Add leakage-safe historical features and compare performance with and without them.
6. Measure KS/PSI and model performance across the same chronological windows.
7. Simulate a seven-day reporting delay by hiding labels for all recent transactions.
8. Select a threshold using validation data and a documented investigation-cost assumption.
9. Freeze the pipeline, evaluate once on the in-domain test, and optionally run an
   external robustness test.

## Leakage controls

- Split by time before fitting any transformer or resampler.
- Calculate each historical feature using transactions that occurred strictly earlier.
- Treat post-report card history as a possible shortcut: compare a transaction-only model
  with the card-history model and report the difference.
- Exclude names, addresses, card numbers and transaction identifiers from model inputs.
- Never downsample or oversample validation and test sets.
- Do not select hyperparameters or thresholds using the final test result.
- In the label-delay experiment, hide labels for every transaction in the final seven
  days; do not implicitly mark recent unlabelled transactions as legitimate.

## Fair model comparison

All models use the same observations, split boundaries and metric functions. Report model
configuration, training time, PR-AUC, ROC-AUC, precision, recall, F1 and confusion matrix.
Accuracy may be included only as a secondary descriptive statistic.

## Planned ablations

- transaction-only versus transaction plus historical profile features;
- reconstructed-card history enabled versus disabled;
- default threshold versus validation-selected threshold;
- no imbalance treatment versus class weighting or training-only downsampling;
- in-domain test versus optional external distribution-shift test.

## Consultation actions and open decisions

- **Card reconstruction:** explain why the original card values do not by themselves form
  a verified unique identifier. Lycia owns the reconstruction proposal.
- **Concept drift:** KS and PSI are descriptive only; results must also show whether model
  performance declines across time windows.
- **M3:** choose and justify a specific semi-supervised algorithm. The previously cited
  Dal Pozzolo work must not be described as semi-supervised learning on unlabelled data.
- **M4:** decide whether the anomaly score is an input feature or is combined with the
  supervised model output.
- **Precision@k:** decide whether `k` means transactions or aggregate model scores to the
  card level before using cards as the unit. This remains a team decision.
