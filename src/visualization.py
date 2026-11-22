"""Portfolio-ready, reproducible ULB fraud exploratory visualizations.

Run from the project root:
    python -m src.visualization
    python -m src.visualization --format png svg
    python -m src.visualization --input data/processed/deduped.csv \
        --tables reports/tables/eda_dedup --output reports/figures/eda_dedup

This module VISUALIZES EDA only. It does not load fitted ML models, infer
calendar time from elapsed seconds, or describe transaction amount as a
verified currency. Charts use raw Amount and Time units; not z-scores.

Requires: numpy, pandas, matplotlib. Optional: seaborn is not required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots reliably in notebooks, CI, and terminals
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "cleaned.csv"
DEFAULT_TABLES = ROOT / "reports" / "tables" / "eda"
DEFAULT_OUT = ROOT / "reports" / "figures" / "eda"

DARK = "#132238"
BLUE = "#3273a8"
TEAL = "#12958e"
RED = "#db5265"
ORANGE = "#efa34a"
GRAY = "#65768b"
LIGHT = "#e7edf2"
PCA_FEATURES = ("V14", "V17", "V12", "V10", "V7")


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 210,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.edgecolor": LIGHT,
        "axes.labelcolor": DARK, "text.color": DARK,
        "xtick.color": GRAY, "ytick.color": GRAY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def thousands(value, _pos=None):
    return f"{value:,.0f}"


def finish(fig, stem, output, formats):
    fig.subplots_adjust(bottom=0.15)
    for ext in formats:
        path = output / f"{stem}.{ext}"
        fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
        print(f"[viz] {path}")
    plt.close(fig)


def title(ax, heading, subtitle=None):
    ax.set_title(heading, loc="left", pad=22, color=DARK)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes,
                va="bottom", ha="left", fontsize=9, color=GRAY)


def footer(fig, note):
    fig.text(0.02, 0.015, note, fontsize=8, color=GRAY, va="bottom")


def read_table(directory, name):
    path = directory / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required EDA table not found: {path}\n"
            "Run: python -m src.analysis --report-dir <tables-path>")
    return pd.read_csv(path)


def validate_inputs(df, tables):
    needed = {"Class", "Amount", "Time", *PCA_FEATURES}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset lacks required fields: {sorted(missing)}")
    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")
    if df["Amount"].isna().any() or df["Time"].isna().any():
        raise ValueError("Missing Amount/Time; resolve quality errors first")
    bal = read_table(tables, "class_balance")
    expected = int(bal["transaction_count"].sum())
    if len(df) != expected:
        raise ValueError(
            f"Mismatched data and analytical tables: {len(df):,} data rows "
            f"versus {expected:,} table rows. Rerun src.analysis with the "
            "same dataset as --input, or use --tables for matching reports.")
    return bal


def plot_class_balance(balance, output, formats):
    balance = balance.sort_values("class")
    labels = balance["class_name"].tolist()
    counts = balance["transaction_count"].to_numpy(dtype=int)
    total = counts.sum()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].barh(labels[::-1], counts[::-1], color=[RED, TEAL], height=.47)
    axes[0].set_xlabel("Transactions (logarithmic scale)")
    axes[0].set_xscale("log")
    axes[0].set_xlim(.7, counts.max() * 3)
    for i, value in enumerate(counts[::-1]):
        axes[0].text(value * 1.07, i, f"{value:,}", va="center", fontweight="bold", fontsize=10)
    title(axes[0], "Fraud is exceptionally rare", "Absolute counts, displayed on a log scale")
    axes[1].axis("off")
    rate = 100 * counts[1] / total
    axes[1].text(.04, .68, f"{rate:.3f}%", color=RED, fontsize=36, fontweight="bold")
    axes[1].text(.04, .53, "of transactions are fraudulent", color=DARK, fontsize=12)
    axes[1].text(.04, .28, f"{counts[0] / max(counts[1], 1):,.0f} : 1", fontsize=25,
                 fontweight="bold", color=DARK)
    axes[1].text(.04, .17, "Legitimate-to-fraud transaction ratio", color=GRAY, fontsize=10)
    footer(fig, "Source: ULB credit card fraud dataset | Class = 1 denotes recorded fraud")
    finish(fig, "01_class_imbalance", output, formats)


def plot_amount_distribution(df, output, formats):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    # log1p is defined at 0 and preserves the original amount when inverted
    for label, klass, color in [("Legitimate", 0, TEAL), ("Fraud", 1, RED)]:
        amount = df.loc[df["Class"].eq(klass), "Amount"].to_numpy()
        axes[0].hist(np.log1p(amount), bins=np.linspace(0, np.log1p(df["Amount"].max()), 62),
                     density=True, histtype="step", linewidth=2.0, label=label, color=color)
        sorted_a = np.sort(amount)
        axes[1].plot(np.log1p(sorted_a), np.arange(1, len(sorted_a)+1) / len(sorted_a),
                     color=color, lw=2, label=label)
    tick_amount = np.array([0, 1, 10, 100, 1000, 10000])
    tick_amount = tick_amount[tick_amount <= df["Amount"].max()]
    for ax in axes:
        ax.set_xticks(np.log1p(tick_amount), [f"{x:,.0f}" for x in tick_amount])
        ax.set_xlabel("Transaction Amount (source units; log1p axis)")
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=.16)
    axes[0].set_ylabel("Density (within each class)")
    axes[1].set_ylabel("Cumulative share of class")
    title(axes[0], "Amount distribution by class", "Log-scale density; both classes normalized separately")
    title(axes[1], "Cumulative amount distribution", "Shows shifts in typical amounts and the upper tail")
    footer(fig, "Amounts remain in original units. A log1p axis accommodates zero-amount transactions.")
    finish(fig, "02_amount_distributions", output, formats)


def plot_amount_summary(table, output, formats):
    row = table.sort_values("Class")
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    names = row["class_name"].tolist()
    for ax, field, heading in [
        (axes[0], "mean_amount", "Mean transaction amount"),
        (axes[1], "median_amount", "Median transaction amount")]:
        vals = row[field].to_numpy()
        ax.bar(x, vals, color=[TEAL, RED], width=.48)
        ax.set_xticks(x, names)
        ax.set_ylabel("Amount (source units)")
        title(ax, heading)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*.04, f"{v:,.2f}", ha="center", fontweight="bold")
        ax.set_ylim(0, max(vals)*1.22)
    footer(fig, "Mean and median tell different stories when amounts are heavily skewed.")
    finish(fig, "03_mean_vs_median", output, formats)


def plot_amount_risk(table, output, formats):
    t = table.sort_values("decile")
    labels = t["decile"].astype(str).tolist()
    x = np.arange(len(t))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.7), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.1]})
    axes[0].bar(x, t["fraud_rate_pct"], color=RED, width=.7)
    lo = (t["fraud_rate_pct"] - t["fraud_rate_ci_low_pct"]).clip(lower=0)
    hi = (t["fraud_rate_ci_high_pct"] - t["fraud_rate_pct"]).clip(lower=0)
    axes[0].errorbar(x, t["fraud_rate_pct"], yerr=np.vstack([lo, hi]),
                     fmt="none", ecolor=DARK, capsize=3, alpha=.7)
    axes[0].set_ylabel("Fraud rate (%)")
    title(axes[0], "Fraud frequency varies by amount band", "Wilson 95% intervals; bands contain approximately equal transaction counts")
    axes[1].bar(x, t["share_of_fraud_value_pct"], color=BLUE, width=.7)
    axes[1].set_ylabel("Share of total fraudulent value (%)")
    axes[1].set_xlabel("Amount quantile band (lowest → highest)")
    axes[1].set_xticks(x, labels)
    title(axes[1], "Financial exposure concentrates in larger amounts")
    for ax in axes:
        ax.grid(axis="y", alpha=.16)
    footer(fig, "Equal-frequency bands may differ in row count because of tied amounts. No causal claims implied.")
    finish(fig, "04_fraud_rate_vs_fraud_value", output, formats)


def plot_amount_exposure(table, output, formats):
    t = table.sort_values("decile")
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10.7, 5.2))
    ax.bar(x, t["fraud_amount"], color=BLUE)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylabel("Recorded fraudulent transaction amount (source units)")
    ax.set_xlabel("Transaction amount quantile (0 = smallest)")
    ax.set_xticks(x, t["decile"].astype(str))
    title(ax, "Where is the fraudulent transaction value?", "Sum of fraud amounts, not confirmed unrecovered bank losses")
    ax.grid(axis="y", alpha=.16)
    footer(fig, "Summed amounts for Class = 1. Exposure is descriptive; preventable loss requires additional assumptions.")
    finish(fig, "05_fraud_amount_by_band", output, formats)


def plot_time_trends(table, output, formats):
    t = table.sort_values("segment")
    x = (t["from_elapsed_hour"] + t["to_elapsed_hour"]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.5), sharex=True)
    axes[0].plot(x, t["fraud_rate_pct"], "o-", color=RED, lw=2, ms=5)
    axes[0].fill_between(x, t["fraud_rate_ci_low_pct"].to_numpy(),
                         t["fraud_rate_ci_high_pct"].to_numpy(), color=RED, alpha=.12)
    axes[0].set_ylabel("Fraud rate (%)")
    title(axes[0], "Observed fraud rate by elapsed time", "Wilson 95% confidence intervals")
    width = np.median(np.diff(x)) * .30 if len(x) > 1 else 1.0
    axes[1].bar(x - width / 2, t["total_txns"], width=width, color=TEAL, label="All transactions")
    axr = axes[1].twinx()
    axr.plot(x, t["fraud_txns"], color=RED, marker="o", lw=1.7, label="Fraud transactions")
    axes[1].set_ylabel("Total transaction count", color=TEAL)
    axr.set_ylabel("Fraud transaction count", color=RED)
    axes[1].set_xlabel("Hours elapsed since first transaction (not clock hour)")
    title(axes[1], "Transaction volume and fraud counts", "Counts help contextualize changes in observed fraud rate")
    axes[1].yaxis.set_major_formatter(FuncFormatter(thousands))
    for ax in axes:
        ax.grid(axis="y", alpha=.16)
    footer(fig, "The dataset spans approximately two days. These are not local time-of-day patterns.")
    finish(fig, "06_temporal_patterns", output, formats)


def plot_zero_amount(table, output, formats):
    t = table.set_index("amount_segment")
    names = ["Positive amount", "Zero amount"]
    t = t.reindex(names)
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    v = t["fraud_rate_pct"].to_numpy()
    ax.bar(x, v, color=[TEAL, RED], width=.46)
    lo = (t["fraud_rate_pct"] - t["fraud_rate_ci_low_pct"]).clip(lower=0)
    hi = (t["fraud_rate_ci_high_pct"] - t["fraud_rate_pct"]).clip(lower=0)
    ax.errorbar(x, v, yerr=[lo.to_numpy(), hi.to_numpy()], fmt="none", color=DARK, capsize=4)
    ax.set_xticks(x, ["Positive-amount", "Zero-amount"])
    ax.set_ylabel("Fraud rate (%)")
    title(ax, "Zero-amount records have elevated observed fraud rates", "Wilson 95% intervals; investigate, do not automatically exclude")
    for i, value in enumerate(v):
        ax.text(i, value + float(hi.iloc[i]) + .05,
                f"{value:.2f}%  (n={int(t.iloc[i]['total_txns']):,})", ha="center", fontsize=10)
    ax.set_ylim(0, max(v+hi.to_numpy()) * 1.25)
    footer(fig, "Unknown transaction semantics prevent attributing zero amounts to a specific business process.")
    finish(fig, "07_zero_amount_risk", output, formats)



def plot_pca_features(df, output, formats):
    """Plot class-conditional PCA feature ranges without label overlap."""

    fig, axes = plt.subplots(
        nrows=len(PCA_FEATURES),
        ncols=1,
        figsize=(12, 17),
        sharex=False
    )

    for ax, name in zip(axes, PCA_FEATURES):
        for klass, yc, color in [
            (0, 1, TEAL),
            (1, 0, RED)
        ]:
            values = df.loc[df["Class"].eq(klass), name]
            p05, median, p95 = values.quantile(
                [0.05, 0.50, 0.95]
            ).to_numpy()

            ax.plot(
                [p05, p95],
                [yc, yc],
                color=color,
                lw=5,
                solid_capstyle="round"
            )

            ax.scatter(
                median,
                yc,
                color="white",
                edgecolor=color,
                s=75,
                zorder=3,
                linewidth=2
            )

        ax.set_yticks([1, 0], ["Legitimate", "Fraud"])
        ax.axvline(0, color=LIGHT, lw=1, ls="--")
        ax.set_title(name, loc="left", weight="bold", pad=14)
        ax.set_ylim(-0.65, 1.65)
        ax.grid(axis="x", alpha=0.14)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        "Class-conditional PCA feature ranges",
        fontsize=17,
        fontweight="bold",
        x=0.08,
        y=0.98,
        ha="left"
    )

    fig.subplots_adjust(
        left=0.17,
        right=0.96,
        top=0.92,
        bottom=0.09,
        hspace=0.65
    )

    fig.supxlabel(
        "Anonymized PCA component value",
        fontsize=12,
        y=0.055
    )

    footer(
        fig,
        "Line = 5th–95th percentile; ring = median. "
        "PCA component semantics are not available."
    )

    finish(fig, "08_pca_class_comparison", output, formats)


def plot_correlations(table, output, formats, top_n=12):
    top = table.assign(abs_r=lambda x: x["pearson_r"].abs()).nlargest(top_n, "abs_r")
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.3, 6.5))
    ax.barh(top["feature"], top["pearson_r"], color=np.where(top["pearson_r"] < 0, RED, TEAL))
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_xlabel("Pearson correlation with fraud label")
    title(ax, "Largest univariate associations with fraud", "Correlation does not imply causation or model-specific feature importance")
    ax.grid(axis="x", alpha=.13)
    footer(fig, "Signed Pearson r for anonymized features; feature meaning is not known.")
    finish(fig, "09_top_feature_correlations", output, formats)


def plot_executive_snapshot(df, bands, balance, output, formats):
    n = len(df)
    f = int(df["Class"].sum())
    fraud_value = df.loc[df["Class"].eq(1), "Amount"].sum()
    last = bands.sort_values("decile").iloc[-1]
    first = bands.sort_values("decile").iloc[0]
    fig = plt.figure(figsize=(12.8, 6.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[.8, 1.3], hspace=.30, wspace=.6)
    cards = [("TOTAL TRANSACTIONS", f"{n:,}"), ("FRAUD CASES", f"{f:,}"),
             ("FRAUD RATE", f"{f/n*100:.3f}%"),
             ("RECORDED FRAUD VALUE", f"{fraud_value:,.2f}")]
    for i, (lab, val) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        ax.text(0, .74, lab, fontsize=9, weight="bold", color=GRAY)
        ax.text(0, .30, val, fontsize=21, weight="bold", color=DARK)
        ax.plot([0, .98], [.08, .08], color=LIGHT, linewidth=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax = fig.add_subplot(gs[1, :])
    cats = ["Lowest amount band", "Highest amount band"]
    x = np.array([0, 1])
    case_share = np.array([first["share_of_fraud_cases_pct"], last["share_of_fraud_cases_pct"]])
    value_share = np.array([first["share_of_fraud_value_pct"], last["share_of_fraud_value_pct"]])
    ax.bar(x - .18, case_share, width=.35, label="Share of fraud cases", color=RED)
    ax.bar(x + .18, value_share, width=.35, label="Share of fraudulent value", color=BLUE)
    ax.set_xticks(x, cats)
    ax.set_ylabel("Share of all fraud (%)")
    ax.legend(loc="upper left", ncol=2)
    title(ax, "The key business trade-off", "Highest case concentration and highest value exposure occur in different amount bands")
    ax.grid(axis="y", alpha=.14)
    footer(fig, "Portfolio summary | Source: validated original raw-unit ULB dataset | Figures describe observations, not prevented losses")
    finish(fig, "00_executive_snapshot", output, formats)


def generate(data_path=DEFAULT_DATA, tables_path=DEFAULT_TABLES,
             output_path=DEFAULT_OUT, formats=("png",)):
    data_path, tables_path, output_path = map(Path, (data_path, tables_path, output_path))
    formats = tuple(dict.fromkeys(formats))
    if not formats or any(f not in {"png", "svg", "pdf"} for f in formats):
        raise ValueError("Supported formats: png, svg, pdf")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset missing: {data_path}. Run the cleaner first.")
    setup_style()
    df = pd.read_csv(data_path)
    balance = validate_inputs(df, tables_path)
    bands = read_table(tables_path, "fraud_by_amount_band")
    amount = read_table(tables_path, "amount_by_class")
    time = read_table(tables_path, "fraud_by_time_interval")
    zero = read_table(tables_path, "zero_amount_analysis")
    corr = read_table(tables_path, "feature_correlations")
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[viz] Source data: {data_path} | {len(df):,} rows")
    plot_executive_snapshot(df, bands, balance, output_path, formats)
    plot_class_balance(balance, output_path, formats)
    plot_amount_distribution(df, output_path, formats)
    plot_amount_summary(amount, output_path, formats)
    plot_amount_risk(bands, output_path, formats)
    plot_amount_exposure(bands, output_path, formats)
    plot_time_trends(time, output_path, formats)
    plot_zero_amount(zero, output_path, formats)
    plot_pca_features(df, output_path, formats)
    plot_correlations(corr, output_path, formats)
    manifest = {
        "data_path": str(data_path), "tables_path": str(tables_path),
        "n_rows": int(len(df)), "n_fraud": int(df["Class"].sum()),
        "format": list(formats), "chart_count": 10,
        "notes": ["Source transaction amount units are not independently verified as EUR.",
                  "Time is elapsed seconds, not clock hour.",
                  "Charts use the specified dataset and matching analysis tables.",
                  "No machine-learning evaluation plots are generated by this module."],
    }
    (output_path / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[viz] Completed 10 charts in {output_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--format", nargs="+", choices=["png", "svg", "pdf"], default=["png"])
    args = parser.parse_args()
    generate(args.input, args.tables, args.output, args.format)


if __name__ == "__main__":
    main()
