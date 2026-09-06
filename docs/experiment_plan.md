# Experiment plan

## Research questions

1. How much do non-linear ensemble models improve PR-AUC over a Logistic Regression
   baseline under chronological validation?
2. How does the alert threshold change fraud recall and the volume of false alerts?
3. Do past-behaviour features improve detection without using future information?
4. How stable are results under time or dataset distribution shift?

## Experimental stages

1. Validate schema, missingness, class balance and chronological coverage.
2. Train Dummy and Logistic Regression baselines using only transaction-level features.
3. Add Random Forest and selected boosting models under the same split and metrics.
4. Add leakage-safe historical features in an ablation experiment.
5. Select a threshold using validation data and a documented investigation-cost assumption.
6. Freeze the pipeline, evaluate once on the in-domain test, and optionally run an
   external robustness test.

## Leakage controls

- Split by time before fitting any transformer or resampler.
- Calculate each historical feature using transactions that occurred strictly earlier.
- Exclude names, addresses, card numbers and transaction identifiers from model inputs.
- Never downsample or oversample validation and test sets.
- Do not select hyperparameters or thresholds using the final test result.

## Fair model comparison

All models use the same observations, split boundaries and metric functions. Report model
configuration, training time, PR-AUC, ROC-AUC, precision, recall, F1 and confusion matrix.
Accuracy may be included only as a secondary descriptive statistic.

## Planned ablations

- transaction-only versus transaction plus historical profile features;
- default threshold versus validation-selected threshold;
- no imbalance treatment versus class weighting or training-only downsampling;
- in-domain test versus optional external distribution-shift test.
