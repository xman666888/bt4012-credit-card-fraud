# Group 17 team plan

Last updated: **15 October 2026**. Reflects the 10 October meeting, the TA feedback of
29 September, and the group chat up to 14 October.

## Members and current ownership

| Member | GitHub | Responsibility | Status (15 Oct) |
|---|---|---|---|
| Yang Tonghang (Gary) | `xman666888` | Feature engineering | Review/merge PR #1, then start the leakage-safe feature pipeline for both datasets |
| Phang Xin Yi | `xin-yi-phang` | Data audit and EDA | Done; PR #1 awaiting review |
| Xiang Qiuyu | `xiang-qiuyu` | M3 algorithm and M4 decision | In progress; holds the remaining consultation booking |
| Lycia Yeo Ying Xuan | `lyciayeo` | IEEE-CIS card-ID reconstruction | In progress; IEEE-CIS training files shared in the chat on 13 Oct |

## Key dates

| Date | Item |
|---|---|
| 17 Oct 2026 (Sat), 9:00 pm | Internal checkpoint and team call |
| 20 Nov 2026 (Fri), 23:59 | Final report (`report17.pdf`), slides (`presentation17.pptx`) and video (`video17.mp4`) |

The course guideline PDF gives 20 November. The Canvas assignment page shows 22 November.
Plan for 20 November unless the teaching team confirms otherwise.

## TA-feedback checklist

- [ ] Define and validate the reconstructed IEEE-CIS card identifier (Lycia).
- [ ] Compare results with and without card-history features (Gary, with Lycia).
- [ ] Pair KS/PSI with model performance by chronological window, especially on Sparkov.
- [ ] Name a concrete semi-supervised algorithm for M3 (Qiu Yu).
- [ ] Decide whether the M4 anomaly score is an input feature or combined with model output (Qiu Yu).
- [ ] Hide labels for all transactions in the last seven days in the delay experiment.
- [ ] Define Precision@k per transaction, or aggregate scores to card level.

## Questions for the next consultation

- Is a 2-code / 2-report contribution split acceptable, and how will GitHub contribution
  be assessed?
- What is the consultation-booking limit?
- Do the agreed dataset roles and the IEEE-CIS split (labelled training file only,
  chronological split) look reasonable?

## Collaboration workflow

- Each member works on a named branch, links the work to an issue, and opens a pull
  request into `main` reviewed by a teammate.
- Contribution history is visible to the teaching team and may inform the peer-evaluation
  adjustment, so commit your own work under your own account.
