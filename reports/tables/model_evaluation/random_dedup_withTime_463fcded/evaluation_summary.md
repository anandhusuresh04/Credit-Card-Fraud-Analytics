# Model Evaluation — Locked Holdout

**Run:** `random` / `463fcded`. **Selected:** `XGBoost__class_weight`.

## Verified results

- Training-only pooled OOF AP: **0.8549**; locked threshold: **0.903320**.
- Holdout AP: **0.8112**; ROC-AUC: **0.9719**.
- Precision: **93.86%**; recall: **75.35%**; F1: **0.8359**.
- TP 107, FP 7, FN 35, TN 84969.
- Alerts: **114** (1.339 per 1,000 holdout transactions).
- Recorded fraud value flagged: **15,522.62**; missed: **4,355.26** (original source units).
- Legitimate transaction value flagged: **170.30** (potential friction exposure, not financial loss).

## Interpretation safeguards

- All holdout plots are **descriptive diagnostics**. Do not optimize model, thresholds, calibration, or alert policy using these labels.
- PR-AUC here is **average precision**, not trapezoidal PR-curve area.
- Score reliability is diagnostic; model probabilities have **not** been calibrated on independent training-only data.
- Amount flagged is not confirmed prevented loss. Operational effectiveness and customer friction are unknown.
- Differences between random and chronological holdouts are not paired causal comparisons.
- Existing bootstrap intervals omit retraining and temporal-shift uncertainty.
- Feature contributions cannot be inferred from this prediction-only artifact.
