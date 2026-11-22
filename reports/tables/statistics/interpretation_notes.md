# Statistical analysis — interpretation and limitations

## What the data supports

- Fraud status is rare, so rate uncertainty and absolute group sizes matter.
- Zero-amount transactions: 1.479% observed fraud rate versus
  0.164% for positive-amount transactions; estimated
  risk ratio 9.00 (approx. 95% CI 6.12–13.23).
- Highest amount decile accounts for 78.11% of recorded fraud
  transaction value. Bootstrap interval: 72.25%–82.90%.
- A significant p-value does not imply a large predictive or business effect.
  Inspect effect sizes alongside adjusted p-values in hypothesis_tests.csv.

## Methodology

- Nine **exploratory** hypothesis tests, with Holm adjustment within one family.
- Two-sided Mann-Whitney tests use rank-biserial effect sizes. Ties are present.
- Two-sample Kolmogorov–Smirnov is included for descriptive shape differences;
  its asymptotic p-value is approximate because Amount has tied values.
- Amount deciles are defined on the FULL validated dataset for descriptive
  segmentation. They must NOT be treated as precomputed model features.
- The extreme-decile and five PCA components were identified in earlier EDA:
  do not describe them as pre-registered or independently validated discoveries.
- Wilson intervals and Katz risk-ratio intervals assume approximately independent
  observations; customer/merchant identities are unavailable to verify this.
- Fraud-amount bootstrap resamples **fraud records only**, holding the full-data
  decile boundaries fixed. This is a descriptive case-mix sensitivity interval,
  not a bank-loss confidence interval, prediction interval, or temporal forecast.

## Data and business limitations

- 'Amount' is shown in original source units; confirm the currency in dataset
  documentation before calling these euro-denominated losses.
- Recorded fraudulent transaction value is NOT confirmed unrecovered financial loss.
- No causal claims, card-testing confirmation, automatic-block recommendation,
  or generalization to new customers can be made from this dataset alone.
- The observation window is approximately 48 hours, not an annual or monthly sample.
- Rows may share underlying cards/merchants, and duplicates may reflect real activity.
- Statistical results are descriptive/exploratory and do not replace a separate
  leakage-free chronological ML validation.
- Sensitivity datasets compared: original_keep_duplicates, exact_duplicates_removed.

## Suggested next decisions

1. Investigate duplicate provenance and retain both versions for sensitivity.
2. Evaluate fraud **case recall and amount-weighted recall** on held-out model
   predictions, with realistic assumptions for reviews, recoveries and declines.
3. Repeat core findings on independent periods before defining operational rules.
