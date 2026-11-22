"""Statistical analysis of the ULB Credit Card Fraud dataset.

Run from the project root, after data_cleaner.py and analysis.py:
    python -m src.statistical_analysis
    python -m src.statistical_analysis --bootstrap 3000 --seed 42

Uses the validated dataset in ORIGINAL units, and optionally compares it
against a separately deduplicated dataset. Does not alter data or fit models.

Outputs (reports/tables/statistics):
    hypothesis_tests.csv                Exploratory tests with Holm correction
    segment_risk.csv                    Rates, Wilson intervals, rate ratios
    value_uncertainty.csv               Bootstrap intervals on fraud value shares
    duplicate_sensitivity.csv           Raw vs deduplicated descriptive comparison
    summary.json                        Machine-readable results and assumptions
    interpretation_notes.md             Human-readable cautious interpretation

Statistical caveats:
- Observational, anonymized data spanning ~48 hours; no causal inference.
- Transactions may be dependent (customer IDs absent), so conventional p-values
  and Wilson CIs rely on assumptions that cannot be fully verified.
- Five named PCA features and amount bands were identified in earlier EDA;
  this is a FOLLOW-UP EXPLORATORY analysis, not independent confirmation.
- Bootstrap over fraud observations describes case-mix sensitivity, not future
  temporal drift or model-performance uncertainty.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .data_cleaner import PROJECT_ROOT, PROCESSED_PATH, TARGET, validate_data
except ImportError:
    from data_cleaner import PROJECT_ROOT, PROCESSED_PATH, TARGET, validate_data

OUT_DIR = PROJECT_ROOT / "reports" / "tables" / "statistics"
DEDUP_PATH = PROJECT_ROOT / "data" / "processed" / "deduped.csv"
PCA_FEATURES = ("V14", "V17", "V12", "V10", "V7")
Z95 = float(stats.norm.ppf(0.975))


def read_validated(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Run python -m src.data_cleaner first.")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Empty dataset: {path}")
    validate_data(frame)
    if not frame[TARGET].eq(0).any() or not frame[TARGET].eq(1).any():
        raise ValueError("Both classes are needed for two-sample statistical comparisons")
    return frame


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson binomial proportion interval; returned on 0..1 scale."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Require n>0 and 0<=k<=n")
    z = float(stats.norm.ppf((1 + confidence) / 2))
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z / den * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def log_risk_ratio_interval(k1: int, n1: int, k0: int, n0: int,
                            confidence: float = 0.95) -> tuple[float, float, float]:
    """Katz log-relative-risk CI; Haldane correction for zero-event groups."""
    if min(n1, n0) <= 0 or not (0 <= k1 <= n1 and 0 <= k0 <= n0):
        raise ValueError("Invalid event and sample counts")
    a, b, c, d = float(k1), float(n1-k1), float(k0), float(n0-k0)
    if a == 0 or c == 0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    rate1, rate0 = a/(a+b), c/(c+d)
    rr = rate1/rate0
    se = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    z = float(stats.norm.ppf((1+confidence)/2))
    return float(rr), float(np.exp(np.log(rr)-z*se)), float(np.exp(np.log(rr)+z*se))


def _amount_bands(df: pd.DataFrame) -> pd.Series:
    bands = pd.qcut(df["Amount"], q=10, labels=False, duplicates="drop")
    if bands.isna().any():
        raise ValueError("Amount quantile binning produced null labels")
    return bands.astype(int)


def segment_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Two descriptive comparisons. The extreme-decile contrast is post-EDA."""
    n = len(df)
    zero = df["Amount"].eq(0)
    bands = _amount_bands(df)
    first, last = int(bands.min()), int(bands.max())
    parts = [
        ("zero_vs_positive", "Zero amount", zero, "Positive amount", ~zero),
        ("lowest_vs_highest_decile", f"Amount decile {first}", bands.eq(first),
         f"Amount decile {last}", bands.eq(last)),
    ]
    rows = []
    for comparison, group1, mask1, group0, mask0 in parts:
        n1, n0 = int(mask1.sum()), int(mask0.sum())
        k1, k0 = int(df.loc[mask1, TARGET].sum()), int(df.loc[mask0, TARGET].sum())
        if not n1 or not n0:
            raise ValueError(f"No records in one group of {comparison}")
        lo1, hi1 = wilson_interval(k1, n1)
        lo0, hi0 = wilson_interval(k0, n0)
        rr, rrlo, rrhi = log_risk_ratio_interval(k1, n1, k0, n0)
        rows.append(dict(comparison=comparison, group=group1,
                         n=n1, fraud=k1, fraud_rate_pct=100*k1/n1,
                         fraud_rate_ci_low_pct=100*lo1, fraud_rate_ci_high_pct=100*hi1,
                         reference_group=group0, reference_n=n0, reference_fraud=k0,
                         reference_rate_pct=100*k0/n0,
                         reference_rate_ci_low_pct=100*lo0,
                         reference_rate_ci_high_pct=100*hi0,
                         risk_ratio=rr, risk_ratio_ci_low=rrlo,
                         risk_ratio_ci_high=rrhi,
                         difference_percentage_points=100*(k1/n1-k0/n0),
                         note=("Decile extremes selected after prior EDA; contrast is exploratory."
                               if comparison.startswith("lowest") else
                               "Zero-amount transactions may have multiple unobserved meanings.")))
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Familywise Holm-Bonferroni adjustment; retains NaN for undefined tests."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not valid.size:
        return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m-rank)*p[idx]))
        out[idx] = running
    return out


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Exploratory tests with one conservative Holm family across all comparisons."""
    f = df[TARGET].eq(1).to_numpy()
    amt = df["Amount"].to_numpy(dtype=float)
    a_f, a_l = amt[f], amt[~f]
    rows: list[dict] = []

    mw = stats.mannwhitneyu(a_f, a_l, alternative="two-sided", method="asymptotic")
    rank_biserial = 2 * float(mw.statistic)/(len(a_f)*len(a_l)) - 1
    rows.append(dict(test_id="amount_mann_whitney", feature="Amount",
                     test="Mann-Whitney U (two-sided)", statistic=float(mw.statistic),
                     p_value=float(mw.pvalue), effect_size=rank_biserial,
                     effect_size_name="rank_biserial",
                     interpretation="Rank-based separation, not a test of mean difference"))

    # KS p-value is approximate for the ULB amount variable because of ties.
    ks = stats.ks_2samp(a_f, a_l, alternative="two-sided", method="asymp")
    rows.append(dict(test_id="amount_ks", feature="Amount",
                     test="Two-sample KS (asymptotic, tied values)",
                     statistic=float(ks.statistic), p_value=float(ks.pvalue),
                     effect_size=float(ks.statistic), effect_size_name="KS_D",
                     interpretation="Distribution-shape comparison; ties make p approximate"))

    bands = _amount_bands(df)
    contingency = pd.crosstab(bands, df[TARGET]).reindex(columns=[0,1], fill_value=0)
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency, correction=False)
    v = np.sqrt(chi2/(contingency.to_numpy().sum() *
                      min(contingency.shape[0]-1, contingency.shape[1]-1)))
    rows.append(dict(test_id="amount_decile_chi_square", feature="Amount deciles",
                     test="Pearson chi-square independence", statistic=float(chi2),
                     p_value=float(p_chi), effect_size=float(v), effect_size_name="Cramers_V",
                     interpretation=f"df={dof}; minimum expected cell={expected.min():.1f}"))

    zero = df["Amount"].eq(0)
    table = np.array([
        [int((zero & df[TARGET].eq(1)).sum()), int((zero & df[TARGET].eq(0)).sum())],
        [int((~zero & df[TARGET].eq(1)).sum()), int((~zero & df[TARGET].eq(0)).sum())]
    ], dtype=int)
    odds, p_fisher = stats.fisher_exact(table, alternative="two-sided")
    rows.append(dict(test_id="zero_amount_fisher", feature="Amount == 0",
                     test="Fisher exact (two-sided)", statistic=float(odds),
                     p_value=float(p_fisher), effect_size=float(odds),
                     effect_size_name="odds_ratio",
                     interpretation="Odds ratio is not a risk ratio; see segment_risk.csv"))

    for col in PCA_FEATURES:
        vals = df[col].to_numpy(dtype=float)
        fvals, lvals = vals[f], vals[~f]
        res = stats.mannwhitneyu(fvals, lvals, alternative="two-sided", method="asymptotic")
        effect = 2*float(res.statistic)/(len(fvals)*len(lvals)) - 1
        rows.append(dict(test_id=f"{col.lower()}_mann_whitney", feature=col,
                         test="Mann-Whitney U (two-sided)",
                         statistic=float(res.statistic), p_value=float(res.pvalue),
                         effect_size=effect, effect_size_name="rank_biserial",
                         interpretation="Selected from earlier EDA; exploratory, not independent confirmation"))

    out = pd.DataFrame(rows)
    out["p_value_holm"] = holm_adjust(out["p_value"].to_numpy())
    out["reject_holm_at_0_05"] = out["p_value_holm"] < 0.05
    out["multiple_testing_family"] = "9 exploratory comparisons (Holm)"
    return out


def value_uncertainty(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Within-fraud-record bootstrap for descriptive composition, not population loss."""
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap resamples for percentile intervals")
    bands = _amount_bands(df)
    fraud = df[TARGET].eq(1).to_numpy()
    amounts = df.loc[fraud, "Amount"].to_numpy(dtype=float)
    if (amounts < 0).any() or amounts.sum() <= 0:
        raise ValueError("Fraud amounts must have a positive total for value-share analysis")
    band_ids = bands.to_numpy()[fraud]
    highest = int(bands.max())
    rng = np.random.default_rng(seed)
    total_value = float(amounts.sum())
    point = float(amounts[band_ids == highest].sum()/total_value)
    out = np.empty(n_boot, dtype=float)
    # Memory bounded: each iteration resamples only ~492 fraud observations.
    for b in range(n_boot):
        idx = rng.integers(0, len(amounts), len(amounts))
        a = amounts[idx]
        denom = float(a.sum())
        out[b] = float(a[band_ids[idx] == highest].sum()/denom) if denom > 0 else np.nan
    valid = out[np.isfinite(out)]
    if len(valid) < 0.95*n_boot:
        raise RuntimeError("Too many invalid bootstrap resamples")
    lo, hi = np.percentile(valid, [2.5,97.5])
    return pd.DataFrame([dict(metric="highest_amount_decile_share_of_fraud_value",
                               estimate_pct=100*point, ci_low_pct=100*float(lo),
                               ci_high_pct=100*float(hi),
                               bootstrap_resamples=n_boot, seed=seed,
                               method="Fraud-row iid percentile bootstrap; full-data decile boundaries fixed",
                               note="Conditional case-mix uncertainty only; ignores dependencies and drift")])


def _summary_metrics(df: pd.DataFrame, name: str) -> dict:
    fraud_mask = df[TARGET].eq(1)
    fraud_amount = float(df.loc[fraud_mask,"Amount"].sum())
    bands = _amount_bands(df)
    last = int(bands.max())
    return dict(dataset=name, transactions=int(len(df)), fraud_cases=int(fraud_mask.sum()),
                fraud_rate_pct=100*float(fraud_mask.mean()),
                duplicate_rows=int(df.duplicated().sum()),
                total_transaction_amount=float(df["Amount"].sum()),
                recorded_fraud_amount=fraud_amount,
                highest_amount_decile_fraud_value_share_pct=(
                    100*float(df.loc[fraud_mask & bands.eq(last),"Amount"].sum())/fraud_amount
                    if fraud_amount > 0 else np.nan),
                zero_amount_fraud_rate_pct=(
                    100*float(df.loc[df["Amount"].eq(0),TARGET].mean())
                    if df["Amount"].eq(0).any() else np.nan))


def duplicate_sensitivity(original: pd.DataFrame, deduped: pd.DataFrame | None) -> pd.DataFrame:
    rows = [_summary_metrics(original, "original_keep_duplicates")]
    if deduped is not None:
        # Compare on identical raw fields, not row count alone. This enforces provenance.
        expected = original.drop_duplicates(keep="first").reset_index(drop=True)
        if not expected.equals(deduped.reset_index(drop=True)):
            raise ValueError("Deduplicated file is not an exact drop_duplicates view of input. "
                             "Regenerate it using data_cleaner.py --duplicates drop")
        rows.append(_summary_metrics(deduped, "exact_duplicates_removed"))
    return pd.DataFrame(rows)


def write_notes(out_dir: Path, tests: pd.DataFrame, risks: pd.DataFrame,
                value: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    z = risks.loc[risks["comparison"].eq("zero_vs_positive")].iloc[0]
    v = value.iloc[0]
    note = f"""# Statistical analysis — interpretation and limitations

## What the data supports

- Fraud status is rare, so rate uncertainty and absolute group sizes matter.
- Zero-amount transactions: {z['fraud_rate_pct']:.3f}% observed fraud rate versus
  {z['reference_rate_pct']:.3f}% for positive-amount transactions; estimated
  risk ratio {z['risk_ratio']:.2f} (approx. 95% CI {z['risk_ratio_ci_low']:.2f}–{z['risk_ratio_ci_high']:.2f}).
- Highest amount decile accounts for {v['estimate_pct']:.2f}% of recorded fraud
  transaction value. Bootstrap interval: {v['ci_low_pct']:.2f}%–{v['ci_high_pct']:.2f}%.
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
- Sensitivity datasets compared: {', '.join(sensitivity['dataset'].tolist())}.

## Suggested next decisions

1. Investigate duplicate provenance and retain both versions for sensitivity.
2. Evaluate fraud **case recall and amount-weighted recall** on held-out model
   predictions, with realistic assumptions for reviews, recoveries and declines.
3. Repeat core findings on independent periods before defining operational rules.
"""
    (out_dir/"interpretation_notes.md").write_text(note, encoding="utf-8")


def run(input_path: Path = PROCESSED_PATH, dedup_path: Path = DEDUP_PATH,
        out_dir: Path = OUT_DIR, bootstrap: int = 2000, seed: int = 42,
        compare_dedup: bool = True) -> dict:
    df = read_validated(Path(input_path))
    dedup = None
    if compare_dedup:
        if Path(dedup_path).is_file():
            dedup = read_validated(Path(dedup_path))
        else:
            print(f"[stats] Deduplicated dataset not found: {dedup_path}; skipping comparison")

    print(f"[stats] Input: {len(df):,} rows; fraud={int(df[TARGET].sum()):,}; "
          f"duplicate rows={int(df.duplicated().sum()):,}")
    tests = hypothesis_tests(df)
    risks = segment_risk(df)
    value = value_uncertainty(df, n_boot=bootstrap, seed=seed)
    sensitivity = duplicate_sensitivity(df, dedup)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"hypothesis_tests":tests, "segment_risk":risks,
              "value_uncertainty":value,"duplicate_sensitivity":sensitivity}
    for name, frame in tables.items():
        frame.to_csv(out_dir/f"{name}.csv", index=False, float_format="%.12g")

    summary = {"generated_utc":datetime.now(timezone.utc).isoformat(),
               "input_path":str(input_path),"dedup_path":str(dedup_path) if dedup is not None else None,
               "observations":len(df),"fraud_cases":int(df[TARGET].sum()),
               "hypothesis_tests":len(tests),"holm_significant_at_0_05":int(tests["reject_holm_at_0_05"].sum()),
               "bootstrap_resamples":bootstrap,"random_seed":seed,
               "analysis_type":"exploratory, observational; no causal inference",
               "currency_note":"Original dataset amount units; no assumption of recovered/unrecovered loss",
               "files":[f"{name}.csv" for name in tables]+["summary.json","interpretation_notes.md"]}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    write_notes(out_dir,tests,risks,value,sensitivity)
    print("\n[stats] Hypothesis tests (Holm adjusted):")
    print(tests[["feature","test","p_value_holm","effect_size_name","effect_size"]]
          .to_string(index=False,float_format=lambda x:f"{x:.5g}"))
    print("\n[stats] Segment comparisons:")
    print(risks[["comparison","fraud_rate_pct","reference_rate_pct","risk_ratio",
                 "risk_ratio_ci_low","risk_ratio_ci_high"]]
          .to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\n[stats] Fraud value uncertainty:")
    print(value[["estimate_pct","ci_low_pct","ci_high_pct"]]
          .to_string(index=False,float_format=lambda x:f"{x:.2f}"))
    print("\n[stats] Duplicate sensitivity:")
    print(sensitivity[["dataset","transactions","fraud_cases","fraud_rate_pct",
                       "recorded_fraud_amount"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(f"\n[stats] Wrote {len(tables)} CSV tables, summary.json, interpretation_notes.md -> {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Exploratory statistical analysis of ULB fraud data")
    p.add_argument("--input", type=Path, default=PROCESSED_PATH)
    p.add_argument("--deduped", type=Path, default=DEDUP_PATH)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-dedup", action="store_true", help="Skip duplicate-sensitivity comparison")
    args = p.parse_args(argv)
    run(args.input,args.deduped,args.out,args.bootstrap,args.seed,not args.skip_dedup)


if __name__ == "__main__":
    main()
