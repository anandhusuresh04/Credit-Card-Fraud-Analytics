# Credit Card Fraud — Business Decision Brief

## Observed facts (historical dataset)

- **284,807** transactions; **492** labeled frauds (**0.1727%** of transactions).
- Recorded fraudulent transaction amount: **60,127.97 source units** (**0.2390%** of total recorded transaction amount).
- Lowest amount quantile: **181** fraud cases (**36.79%** of fraud cases); recorded fraudulent amount **139.56**.
- Highest amount quantile: **85** fraud cases, but **78.11%** of total fraudulent amount.
- Zero-amount group: **27** fraud cases among **1,825** records; this is an association, not evidence of card testing.

## Decision implications

- Compare **case coverage** and **fraud-value coverage** as distinct objectives.
- Review workload must be considered before selecting a policy; amount bands alone are broad and expensive to investigate.
- Zero-amount cases warrant investigation of transaction semantics, not an automatic blocking rule.

## Scenario limitations

- Policy rows reflect **in-sample, label-informed retrospective counts**, not model generalization or a validated decision rule.
- The hypothetical net-value scenarios assume perfect reviews and a specified intervention-effectiveness fraction.
- Recorded fraudulent amount is **not observed financial loss, recoverable loss, or realized savings**.
- No customer IDs, merchant IDs, timestamp origin, or downstream intervention outcomes are available.
- Transaction dependence, duplicates, temporal drift, and changes in future behavior are unresolved.
- Currency is kept as **source units**; validate the original dataset documentation before interpreting as EUR.

## Provenance

- Input: `E:\DataAnalytics\Projects\Credit_Card_Fraud_Analytics\data\processed\cleaned.csv`
- SHA-256: `895dc3dad2840ac9e05c12d6442bd739d879c2d405e9b065efc0f1973be46a84`
- Created UTC: `2026-11-22T18:14:27.474722+00:00`

## Next required validation

Re-run the current ML pipeline with train-only preprocessing and a held-out chronological test. Evaluate alerts, case coverage, value coverage, and assumed cost on actual out-of-sample predictions.
