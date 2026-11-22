# Investigation Capacity Optimization — Retrospective Analysis

Source run: `reports\tables\ml\chronological_dedup_withTime_463fcded` · chronological split · XGBoost__SMOTE

**Train-only policy-selection objective:** value_recall.
**Batch capacities:** 5.0, 10.0, 20.0, 50.0 alerts per 10,000 transactions.

## Holdout outcomes for policies selected on training OOF

| Allowance / 10k | Selected ranking | Alerts | Fraud cases caught | Case recall | Fraud-value recall | Scenario net value |
|---:|---|---:|---:|---:|---:|---:|
| 5 | score_times_amount | 42 | 13 | 12.15% | 62.98% | 4,376.59 |
| 10 | score_times_amount | 85 | 17 | 15.89% | 71.43% | 4,777.07 |
| 20 | score_times_amount | 170 | 28 | 26.17% | 83.84% | 5,255.58 |
| 50 | score_times_amount | 425 | 37 | 34.58% | 87.13% | 4,220.32 |

## Critical limitations

- These are review queues, NOT automatic declines or demonstrated prevented losses.
- Review caps are applied to a full batch; this assumes the entire batch score ranking is available, not real-time decisions.
- The winning ranking rule was chosen with training OOF labels. The holdout comparison is descriptive only.
- Financial numbers are illustrative assumptions in the original dataset amount units; currency is unverified.
- Model scores have not been calibrated. Score × amount is only a heuristic prioritization rule.
- Small holdout fraud counts and temporal change introduce considerable uncertainty.
