# Model Evaluation — Locked Holdout

**Run:** `chronological` / `463fcded`. **Selected:** `XGBoost__SMOTE`.

## Verified results

- Training-only pooled OOF AP: **0.7641**; locked threshold: **0.715498**.
- Holdout AP: **0.7831**; ROC-AUC: **0.9830**.
- Precision: **82.98%**; recall: **72.90%**; F1: **0.7761**.
- TP 78, FP 16, FN 29, TN 84995.
- Alerts: **94** (1.104 per 1,000 holdout transactions).
- Recorded fraud value flagged: **10,372.83**; missed: **4,192.16** (original source units).
- Legitimate transaction value flagged: **29,406.78** (potential friction exposure, not financial loss).

## Interpretation safeguards

- All holdout plots are **descriptive diagnostics**. Do not optimize model, thresholds, calibration, or alert policy using these labels.
- PR-AUC here is **average precision**, not trapezoidal PR-curve area.
- Score reliability is diagnostic; model probabilities have **not** been calibrated on independent training-only data.
- Amount flagged is not confirmed prevented loss. Operational effectiveness and customer friction are unknown.
- Differences between random and chronological holdouts are not paired causal comparisons.
- Existing bootstrap intervals omit retraining and temporal-shift uncertainty.
- Feature contributions cannot be inferred from this prediction-only artifact.
