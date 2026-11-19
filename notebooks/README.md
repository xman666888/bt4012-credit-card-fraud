# Notebooks

Run the notebooks in this order:

1. `01_data_audit.ipynb` - schema, missing values, time range and class balance of both datasets.
2. `02_exploratory_analysis.ipynb` - fraud patterns and early vs late comparisons of both datasets.
3. `03_ieee_card_reconstruction.ipynb` - builds a card ID for IEEE-CIS.
4. `04_feature_engineering.ipynb` - builds the feature tables of both datasets.
5. `05_ieee_m0_m2.ipynb` - M0, M1 and M2 with LR, RF and CatBoost on IEEE-CIS.
6. `05_sparkov_m0_m2.ipynb` - M0, M1 and M2 with LR, RF, Catboost and LightGBM on Sparkov.
7. `05b_ieee_lightgbm_m0_m1.ipynb` - LightGBM M0 and M1 on IEEE-CIS.
8. `05c_ieee_lightgbm_m2.ipynb` - LightGBM M2 on IEEE-CIS.
9. `05d_ieee_drift_vs_performance.ipynb` - drift vs model performance on IEEE-CIS.
10. `06_ieee_m2_catboost_tuning.ipynb` - feature selection and tuning of M2 CatBoost on IEEE-CIS.
11. `07_ieee_m3_pseudo_labelling.ipynb` - M3 pseudo-labelling on IEEE-CIS.
12. `08_ieee_m4_isolation_forest.ipynb` - M4 anomaly score features on IEEE-CIS.
13. `09_sparkov_ieee_comparison.ipynb` - why the results differ between Sparkov and IEEE-CIS.
14. `10_ieee_final_test.ipynb` - final model on the IEEE-CIS test set, with the daily alert budget.

## Data needed

03 and 04 need the raw Kaggle files in `data/raw/` (see `data/README.md`).
04 writes the feature tables to `data/processed/`.
They are too big for GitHub, so 04 has to be run before the other notebooks.

## Saved results

The slow runs are saved in `data/interim/`.
A rerun loads them instead of training again.

- `data/interim/ieee/` - IEEE-CIS card IDs, scores and tuning results (05 to 07 and 10)
- `data/interim/sparkov/` - Sparkov label delay check (09)
- `data/interim/comparison_runs/` - other runs of 09

The trained final models of 10 are about 200 MB each, so they are not committed.
Their test scores are saved, so 10 still runs without them.
