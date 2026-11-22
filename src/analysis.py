"""Reproducible descriptive, exploratory and statistical ULB fraud analysis.

Run after the improved cleaner:
    python -m src.data_cleaner
    python -m src.analysis

Input retains raw Amount and Time units (no global standardization).
Outputs: reports/tables/eda/*.csv and reports/tables/eda/summary.json.
Findings are observational; they do not establish the cause of fraud.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .data_cleaner import (
        PROJECT_ROOT, PROCESSED_PATH, TARGET, FEATURE_COLS,
        validate_data,
    )
except ImportError:
    from data_cleaner import (
        PROJECT_ROOT, PROCESSED_PATH, TARGET, FEATURE_COLS,
        validate_data,
    )

REPORT_DIR = PROJECT_ROOT / "reports" / "tables" / "eda"
PCA_COLS = [f"V{i}" for i in range(1, 29)]


def load_processed(path: str | Path = PROCESSED_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: python -m src.data_cleaner")
    df = pd.read_csv(path, low_memory=False)
    validate_data(df)
    return df


def _wilson_interval(positives: pd.Series | np.ndarray, totals: pd.Series | np.ndarray,
                     z: float = 1.959963984540054) -> tuple[np.ndarray, np.ndarray]:
    """Wilson 95% confidence intervals for binomial fraud proportions."""
    k = np.asarray(positives, dtype=float)
    n = np.asarray(totals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        den = 1 + z*z / n
        center = (p + z*z / (2*n)) / den
        half = z * np.sqrt((p*(1-p) + z*z/(4*n)) / n) / den
    return np.where(n > 0, center-half, np.nan), np.where(n > 0, center+half, np.nan)


def fraud_rate_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    out = pd.DataFrame({
        "class": [0, 1], "class_name": ["Legitimate", "Fraud"],
        "transaction_count": counts.to_numpy(dtype=int),
    })
    out["percent_of_transactions"] = out["transaction_count"] / len(df) * 100
    fraud = int(counts[1]); legit = int(counts[0])
    print(f"\n1. CLASS BALANCE: {fraud:,}/{len(df):,} fraud "
          f"({100 * fraud / len(df):.4f}%), legitimate:fraud = {legit/max(fraud,1):.1f}:1")
    return out


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full numeric feature summary; raw Amount and elapsed Time units."""
    cols = FEATURE_COLS
    desc = df[cols].describe(percentiles=[.01, .05, .25, .5, .75, .9, .95, .99]).T
    desc.index.name = "feature"
    desc = desc.reset_index()
    desc["skew"] = [float(df[c].skew()) for c in cols]
    desc["missing"] = [int(df[c].isna().sum()) for c in cols]
    print("\n2. FEATURE PROFILE (selected fields)")
    print(desc.loc[desc["feature"].isin(["Time", "Amount", "V14", "V17"]),
                   ["feature", "mean", "std", "50%", "95%", "max", "skew"]].round(3).to_string(index=False))
    return desc


def amount_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Raw amount levels, fraud exposure and concentration by class."""
    g = df.groupby(TARGET)["Amount"]
    out = g.agg(txn_count="count", total_amount="sum", mean_amount="mean",
                median_amount="median", min_amount="min", max_amount="max").reset_index()
    for p in (0.90, 0.95, 0.99):
        out[f"p{int(p*100)}_amount"] = out[TARGET].map(g.quantile(p))
    out["share_of_total_amount_pct"] = 100 * out["total_amount"] / df["Amount"].sum()
    out["class_name"] = out[TARGET].map({0: "Legitimate", 1: "Fraud"})
    print("\n3. AMOUNT BY CLASS (original dataset units)")
    print(out[["class_name", "txn_count", "total_amount", "mean_amount", "median_amount", "p95_amount"]]
          .round(2).to_string(index=False))
    return out


def correlation_with_fraud(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Point-biserial/Pearson correlations; association != causation."""
    corr = df[FEATURE_COLS + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    out = pd.DataFrame({"feature": corr.index, "pearson_r": corr.values})
    out["abs_r"] = out["pearson_r"].abs()
    out = out.sort_values("abs_r", ascending=False).reset_index(drop=True)
    print(f"\n4. TOP {top_n} ABSOLUTE CORRELATIONS WITH FRAUD")
    print(out.head(top_n)[["feature", "pearson_r"]].round(4).to_string(index=False))
    print("[analysis] Correlations are univariate associations; they do not prove nonlinear-model superiority.")
    return out


def fraud_by_amount_band(df: pd.DataFrame, q: int = 10) -> pd.DataFrame:
    """Count-based quantile bands with exposure, rates and Wilson intervals."""
    working = df[[TARGET, "Amount"]].copy()
    working["amount_band"] = pd.qcut(working["Amount"], q=q, labels=False, duplicates="drop")
    working["fraud_amount"] = working["Amount"].where(working[TARGET].eq(1), 0.0)
    out = working.groupby("amount_band", observed=True).agg(
        total_txns=(TARGET, "size"), fraud_txns=(TARGET, "sum"),
        min_amount=("Amount", "min"), max_amount=("Amount", "max"),
        median_amount=("Amount", "median"), total_amount=("Amount", "sum"),
        fraud_amount=("fraud_amount", "sum"),
    ).reset_index().rename(columns={"amount_band": "decile"})
    out["fraud_rate_pct"] = 100 * out["fraud_txns"] / out["total_txns"]
    out["share_of_fraud_cases_pct"] = 100 * out["fraud_txns"] / max(int(df[TARGET].sum()), 1)
    fraud_total = float(working["fraud_amount"].sum())
    out["share_of_fraud_value_pct"] = 100 * out["fraud_amount"] / fraud_total if fraud_total else np.nan
    lo, hi = _wilson_interval(out["fraud_txns"], out["total_txns"])
    out["fraud_rate_ci_low_pct"] = lo * 100
    out["fraud_rate_ci_high_pct"] = hi * 100
    print("\n5. FRAUD BY AMOUNT QUANTILE")
    print(out[["decile", "total_txns", "fraud_txns", "fraud_rate_pct", "fraud_amount",
               "share_of_fraud_value_pct"]].round(3).to_string(index=False))
    if len(out) != q:
        print(f"[analysis] Quantile ties resulted in {len(out)} bands instead of {q}.")
    return out


def fraud_by_time_interval(df: pd.DataFrame, hours: float = 4.0) -> pd.DataFrame:
    """Fixed elapsed-time windows; not wall-clock 'hour of day'."""
    if hours <= 0:
        raise ValueError("hours must be positive")
    w = df[["Time", "Amount", TARGET]].copy()
    seconds = float(hours * 3600)
    origin = float(w["Time"].min())
    w["segment"] = np.floor((w["Time"] - origin) / seconds).astype(int)
    w["fraud_amount"] = w["Amount"].where(w[TARGET].eq(1), 0.0)
    out = w.groupby("segment").agg(
        total_txns=(TARGET, "size"), fraud_txns=(TARGET, "sum"),
        transaction_amount=("Amount", "sum"), fraud_amount=("fraud_amount", "sum")
    ).reset_index()
    out["from_elapsed_hour"] = out["segment"] * hours
    out["to_elapsed_hour"] = (out["segment"] + 1) * hours
    out["fraud_rate_pct"] = 100 * out["fraud_txns"] / out["total_txns"]
    lo, hi = _wilson_interval(out["fraud_txns"], out["total_txns"])
    out["fraud_rate_ci_low_pct"] = 100 * lo
    out["fraud_rate_ci_high_pct"] = 100 * hi
    print(f"\n6. FRAUD BY {hours:g}-HOUR ELAPSED-TIME INTERVAL")
    print(out[["segment", "total_txns", "fraud_txns", "fraud_rate_pct", "fraud_amount"]]
          .round(3).to_string(index=False))
    return out


def zero_amount_analysis(df: pd.DataFrame) -> pd.DataFrame:
    w = df[[TARGET, "Amount"]].copy()
    w["amount_segment"] = np.where(w["Amount"].eq(0), "Zero amount", "Positive amount")
    out = w.groupby("amount_segment").agg(total_txns=(TARGET, "size"), fraud_txns=(TARGET, "sum")).reset_index()
    out["fraud_rate_pct"] = 100 * out["fraud_txns"] / out["total_txns"]
    lo, hi = _wilson_interval(out["fraud_txns"], out["total_txns"])
    out["fraud_rate_ci_low_pct"] = 100 * lo
    out["fraud_rate_ci_high_pct"] = 100 * hi
    print("\n7. ZERO-AMOUNT SEGMENT")
    print(out.round(4).to_string(index=False))
    return out


def feature_distributions_by_class(df: pd.DataFrame,
                                   features: tuple[str, ...] = ("V14", "V17", "V12", "V10", "V7")) -> pd.DataFrame:
    """Robust class-conditional summaries; PCA components are anonymized."""
    unknown = set(features) - set(FEATURE_COLS)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    rows = []
    for col in features:
        for label in (0, 1):
            s = df.loc[df[TARGET].eq(label), col]
            rows.append({"feature": col, "class": label, "count": len(s),
                         "mean": float(s.mean()), "median": float(s.median()),
                         "std": float(s.std()), "p05": float(s.quantile(.05)),
                         "p25": float(s.quantile(.25)), "p75": float(s.quantile(.75)),
                         "p95": float(s.quantile(.95))})
    out = pd.DataFrame(rows)
    print("\n8. SELECTED PCA FEATURE DISTRIBUTIONS BY CLASS")
    print(out[["feature", "class", "mean", "median", "p05", "p95"]].round(3).to_string(index=False))
    return out


def statistical_tests(df: pd.DataFrame, bands: pd.DataFrame | None = None) -> pd.DataFrame:
    """Exploratory, two-sided tests. P-values do NOT measure effect size.

    Amount differs between classes (Mann-Whitney and KS). The chi-square test
    checks independence of Class and amount quantile bands. These are three
    prespecified *descriptive* comparisons, not causal claims.
    """
    legit = df.loc[df[TARGET].eq(0), "Amount"].to_numpy()
    fraud = df.loc[df[TARGET].eq(1), "Amount"].to_numpy()
    mw = stats.mannwhitneyu(fraud, legit, alternative="two-sided", method="asymptotic")
    ks = stats.ks_2samp(fraud, legit, alternative="two-sided", method="auto")
    # Rank-biserial effect size, positive means fraud amounts tend to be larger.
    rank_biserial = 2 * float(mw.statistic) / (len(fraud) * len(legit)) - 1
    temp = df[["Amount", TARGET]].copy()
    temp["band"] = pd.qcut(temp["Amount"], q=10, labels=False, duplicates="drop")
    contingency = pd.crosstab(temp["band"], temp[TARGET]).reindex(columns=[0, 1], fill_value=0)
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency, correction=False)
    n = int(contingency.to_numpy().sum())
    cramers_v = float(np.sqrt(chi2 / (n * min(contingency.shape[0]-1, contingency.shape[1]-1))))
    out = pd.DataFrame([
        {"test": "Mann-Whitney U: Amount by Class", "statistic": float(mw.statistic),
         "p_value": float(mw.pvalue), "effect_size_name": "rank_biserial", "effect_size": rank_biserial,
         "notes": "Two-sided; ties present. Difference in distributions, not means."},
        {"test": "Two-sample KS: Amount by Class", "statistic": float(ks.statistic),
         "p_value": float(ks.pvalue), "effect_size_name": "KS_D", "effect_size": float(ks.statistic),
         "notes": "Two-sided; ties make classical continuous-null p-value approximate."},
        {"test": "Chi-square: Class vs Amount quantiles", "statistic": float(chi2),
         "p_value": float(p_chi), "effect_size_name": "Cramers_V", "effect_size": cramers_v,
         "notes": f"df={dof}; minimum expected count={expected.min():.2f}. Observational association."},
    ])
    print("\n9. EXPLORATORY STATISTICAL TESTS")
    print(out[["test", "statistic", "p_value", "effect_size_name", "effect_size"]]
          .to_string(index=False))
    return out


def run_all(df: pd.DataFrame, report_dir: str | Path | None = REPORT_DIR,
            time_bin_hours: float = 4.0) -> dict[str, pd.DataFrame]:
    validate_data(df)
    tables = {
        "class_balance": fraud_rate_summary(df),
        "feature_profile": describe_features(df),
        "amount_by_class": amount_by_class(df),
        "feature_correlations": correlation_with_fraud(df),
        "fraud_by_amount_band": fraud_by_amount_band(df),
        "fraud_by_time_interval": fraud_by_time_interval(df, hours=time_bin_hours),
        "zero_amount_analysis": zero_amount_analysis(df),
        "feature_by_class": feature_distributions_by_class(df),
        "statistical_tests": statistical_tests(df),
    }
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(report_dir / f"{name}.csv", index=False)
        total_fraud = float(df.loc[df[TARGET] == 1, "Amount"].sum())
        summary: dict[str, Any] = {
            "total_transactions": int(len(df)),
            "fraud_transactions": int(df[TARGET].sum()),
            "fraud_rate_pct": float(100 * df[TARGET].mean()),
            "total_transaction_amount": float(df["Amount"].sum()),
            "total_fraud_amount": total_fraud,
            "fraud_value_share_pct": float(100 * total_fraud / df["Amount"].sum()),
            "time_span_hours": float((df["Time"].max() - df["Time"].min()) / 3600),
            "units_note": "Amount is in source dataset units; confirm currency using dataset documentation.",
            "interpretation_note": "Associations and patterns do not establish causal fraud mechanisms.",
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[analysis] Saved {len(tables)} tables and summary.json -> {report_dir}")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--time-bin-hours", type=float, default=4.0)
    args = parser.parse_args()
    df = load_processed(args.input)
    run_all(df, report_dir=args.report_dir, time_bin_hours=args.time_bin_hours)


if __name__ == "__main__":
    main()
