# Investigation Capacity Optimization — Retrospective Analysis

Source run: `reports\tables\ml\random_dedup_withTime_463fcded` · random split · XGBoost__class_weight

**Train-only policy-selection objective:** value_recall.
**Batch capacities:** 5.0, 10.0, 20.0, 50.0 alerts per 10,000 transactions.

## Holdout outcomes for policies selected on training OOF

| Allowance / 10k | Selected ranking | Alerts | Fraud cases caught | Case recall | Fraud-value recall | Scenario net value |
|---:|---|---:|---:|---:|---:|---:|
| 5 | score_times_amount | 42 | 27 | 19.01% | 67.85% | 6,533.59 |
| 10 | score_times_amount | 85 | 48 | 33.80% | 79.14% | 7,440.44 |
| 20 | score_times_amount | 170 | 60 | 42.25% | 85.57% | 7,654.91 |
| 50 | score_times_amount | 425 | 68 | 47.89% | 86.10% | 6,432.80 |

## Critical limitations

- These are review queues, NOT automatic declines or demonstrated prevented losses.
- Review caps are applied to a full batch; this assumes the entire batch score ranking is available, not real-time decisions.
- The winning ranking rule was chosen with training OOF labels. The holdout comparison is descriptive only.
- Financial numbers are illustrative assumptions in the original dataset amount units; currency is unverified.
- Model scores have not been calibrated. Score × amount is only a heuristic prioritization rule.
- Small holdout fraud counts and temporal change introduce considerable uncertainty.
