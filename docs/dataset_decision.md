# Dataset decision record

Status: **Sparkov selected as the primary dataset**

## Selection criteria

The primary dataset must represent credit-card transaction fraud, have an accessible
license, interpretable fields, a timestamp, a binary label, and enough observations for
tree and boosting models.

| Candidate | Decision | Main reason |
|---|---|---|
| Sparkov | Primary | Interpretable, temporal, synthetic, CC0 and directly relevant |
| Fraud Detection Handbook simulator | External/backup | Reproducible and temporal but simpler synthetic features |
| IBM TabFormer | External/backup | Rich and interpretable, but 24M rows require careful sampling |
| Feedzai BAF | Reject for current scope | Models account-opening fraud, not card transactions |
| ULB credit-card fraud | Reject | Main variables are PCA-anonymised |
| IEEE-CIS | Reject unless scope changes | Many feature meanings are hidden and access terms are restrictive |

## Split policy

`fraudTrain.csv` is used for chronological training and validation. `fraudTest.csv` remains
untouched until the feature set, model family, hyperparameters and operating threshold
have been selected. No resampling is applied to validation or test data.

If a second source is evaluated, it is an external distribution-shift test rather than a
replacement for the in-domain test. Only common, semantically equivalent features may be
used, and the result must be reported separately.

## Sources

- Sparkov: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Amazon Fraud Dataset Benchmark: https://github.com/amazon-science/fraud-dataset-benchmark
- Fraud Detection Handbook: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
- IBM TabFormer: https://github.com/IBM/TabFormer
