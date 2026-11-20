# Notebooks

Run the notebooks in this order:

1. `01_data_audit.ipynb` - schema, missing values, time range and class balance of both datasets.
2. `02_exploratory_analysis.ipynb` - fraud patterns and early vs late comparisons of both datasets.
3. `03_ieee_card_reconstruction.ipynb` - builds a card ID for IEEE-CIS.
4. `04_feature_engineering.ipynb` - builds the feature tables of both datasets.
5. `04b_temporal_drift_and_label_delay.ipynb` - fraud rate and feature drift (PSI, KS) per week, and how much fraud the 7-day label delay hides.
6. `04c_drift_and_network_diagnostics.ipynb` - adversarial validation, decay of a model trained once, and fraud clusters on shared entities.
7. `05_ieee_m0_m2.ipynb` - M0, M1 and M2 with LR, RF and CatBoost on IEEE-CIS.
8. `05_sparkov_m0_m2.ipynb` - M0, M1 and M2 with LR, RF, CatBoost and LightGBM on Sparkov.
9. `05b_ieee_lightgbm_m0_m1.ipynb` - LightGBM M0 and M1 on IEEE-CIS.
10. `05c_ieee_lightgbm_m2.ipynb` - LightGBM M2 on IEEE-CIS.
11. `05d_ieee_drift_vs_performance.ipynb` - drift vs model performance on IEEE-CIS.
12. `06_ieee_m2_catboost_tuning.ipynb` - feature selection and tuning of M2 CatBoost on IEEE-CIS.
13. `06_sparkov_m3_lightgbm_pseudolabeling.ipynb` - early M3 pseudo-labelling experiment on Sparkov.
14. `07_ieee_m3_pseudo_labelling.ipynb` - M3 pseudo-labelling on IEEE-CIS.
15. `07_sparkov_m3_catboost_8x3.ipynb` - M3 pseudo-labelling with CatBoost on Sparkov.
16. `08_ieee_m4_isolation_forest.ipynb` - M4 anomaly score features on IEEE-CIS.
17. `08_sparkov_m4_catboost.ipynb` - M4 Isolation Forest score with CatBoost on Sparkov.
18. `09_sparkov_ieee_comparison.ipynb` - why the results differ between Sparkov and IEEE-CIS.
19. `10_ieee_final_test.ipynb` - final model on the IEEE-CIS test set, with the daily alert budget.
20. `10_sparkov_final_test_catboost.ipynb` - final model on the Sparkov test set, with the daily alert budget.

`archive/` keeps the earlier LightGBM versions of the Sparkov M3, M4 and final test notebooks.

## Data needed

03 and 04 need the raw Kaggle files in `data/raw/` (see `data/README.md`).
04 writes the feature tables to `data/processed/`.
They are too big for GitHub, so 04 has to be run before the other notebooks.

## Saved results

The slow runs are saved, so a rerun loads them instead of training again.

- `results/ieee/` - IEEE-CIS scores and tuning results (05 to 08 and 10_ieee)
- `results/comparison_runs/` and `results/sparkov_delay_check/` - extra runs of 09
- `results/` (other folders) - Sparkov results (05_sparkov, 07_sparkov, 08_sparkov and 10_sparkov)

The IEEE-CIS card IDs from 03 are saved in `data/processed/ieee_card_id.csv` and used by 04, 04b and 04c.

The trained final models of 10_ieee are about 200 MB each, so they are not committed.
Their test scores are saved, so 10_ieee still runs without them.
