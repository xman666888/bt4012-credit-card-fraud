# Group 17 team plan

Last aligned with the team discussion and consultation notes on **12 October 2026**.

## Members and current ownership

| Member | GitHub username | Current responsibility | Status / hand-off |
|---|---|---|---|
| Yang Tonghang (Gary) | `xman666888` | Feature engineering | Starts from Xin Yi's EDA outputs |
| Phang Xin Yi | `xin-yi-phang` | Exploratory data analysis | Target: clear the main EDA work by 13 October |
| Xiang Qiuyu | `xiang-qiuyu` | M3 algorithm proposal and M4 decision | Specify the method and integration rule |
| Lycia Yeo Ying Xuan | `lyciayeo` | Card-ID reconstruction | Explain assumptions and leakage risk |

## Immediate timeline

- Internal checkpoint: **17 October 2026**.
- Next team call: **17 October 2026, 9:00 pm Singapore time**.
- Feature engineering begins after the EDA output and data-quality findings are available.

## Consultation-action checklist

- [ ] Define and validate the reconstructed card identifier.
- [ ] Compare model results with and without card-history features.
- [ ] Pair KS/PSI results with model performance by chronological window.
- [ ] Select a concrete semi-supervised method for M3.
- [ ] Decide how the M4 anomaly score interacts with the supervised model.
- [ ] Hide labels for all transactions in the last seven days in the delay experiment.
- [ ] Define Precision@k at the transaction level or aggregate scores to card level.
- [ ] Confirm with the teaching team whether a 2-code / 2-report contribution split is
      acceptable and how GitHub contribution will be assessed.
- [ ] Confirm the consultation-booking policy.

## Collaboration workflow

Each member should work on a named branch, link the work to an issue, and open a pull
request into `main`. Important experiment decisions and external code must be recorded in
the repository so that contribution and reproducibility are visible.
