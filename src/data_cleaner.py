"""Validate and prepare the ULB credit-card fraud dataset without ML leakage.

Run from the project root (or any working directory):
    python -m src.data_cleaner
    python -m src.data_cleaner --duplicates drop

The output data/processed/cleaned.csv retains the ORIGINAL units for Time
(seconds since first transaction) and Amount. Model scaling belongs exclusively
in an sklearn/imblearn Pipeline fitted on training folds (see ml_models.py).

Default duplicate policy is 'keep': exact matching rows are not necessarily
erroneous transactions, because the public dataset has no unique transaction ID.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
QUALITY_DIR = PROJECT_ROOT / "reports" / "tables" / "data_quality"
TARGET = "Class"
EXPECTED_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", TARGET]
FEATURE_COLS = [c for c in EXPECTED_COLS if c != TARGET]


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing raw dataset: {path}\n"
            "Get creditcard.csv from the ULB credit-card fraud dataset and put it "
            "under data/raw/, or pass --raw-path."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"[clean] Read {len(df):,} rows x {len(df.columns)} columns from {path}")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Fail on missing/extra fields, duplicate column names, bad types or labels."""
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names detected")
    missing = sorted(set(EXPECTED_COLS) - set(df.columns))
    extra = sorted(set(df.columns) - set(EXPECTED_COLS))
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {missing}; unexpected: {extra}")
    non_numeric = [c for c in EXPECTED_COLS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric columns: {non_numeric}; inspect raw CSV")
    # The numeric 0/1 requirement is checked after the finite/missing checks.
    print("[clean] Schema OK: 31 expected numeric columns")


def _python_value(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def quality_profile(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return a summary, per-column profile, and exact-duplicate-by-class table.

    This profiles the data but does not change or delete observations.
    """
    validate_schema(df)
    numeric = df[EXPECTED_COLS]
    is_finite = np.isfinite(numeric.to_numpy(dtype=np.float64))
    nonfinite_by_col = (~is_finite).sum(axis=0)
    missing_by_col = numeric.isna().sum()
    dup = df.duplicated(keep="first")
    duplicate_by_class = (
        df.loc[dup].groupby(TARGET, dropna=False).size()
        .reindex([0, 1], fill_value=0).rename("duplicate_rows").reset_index()
    )
    column_profile = pd.DataFrame({
        "column": EXPECTED_COLS,
        "dtype": [str(df[c].dtype) for c in EXPECTED_COLS],
        "missing": missing_by_col.to_numpy(dtype=np.int64),
        "non_finite": nonfinite_by_col.astype(np.int64),
        "distinct_values": [int(df[c].nunique(dropna=True)) for c in EXPECTED_COLS],
        "min": [float(df[c].min()) for c in EXPECTED_COLS],
        "median": [float(df[c].median()) for c in EXPECTED_COLS],
        "max": [float(df[c].max()) for c in EXPECTED_COLS],
    })
    counts = df[TARGET].value_counts(dropna=False)
    total = len(df)
    amount = df["Amount"]
    summary = {
        "rows": total,
        "columns": len(df.columns),
        "missing_cells": int(numeric.isna().sum().sum()),
        "non_finite_cells": int((~is_finite).sum()),
        "duplicate_rows_excluding_first": int(dup.sum()),
        "duplicate_legitimate_rows": int(((df[TARGET] == 0) & dup).sum()),
        "duplicate_fraud_rows": int(((df[TARGET] == 1) & dup).sum()),
        "legitimate_rows": int(counts.get(0, 0)),
        "fraud_rows": int(counts.get(1, 0)),
        "fraud_rate": float((df[TARGET] == 1).mean()),
        "negative_amount_rows": int((amount < 0).sum()),
        "zero_amount_rows": int((amount == 0).sum()),
        "negative_time_rows": int((df["Time"] < 0).sum()),
        "time_min_seconds": float(df["Time"].min()),
        "time_max_seconds": float(df["Time"].max()),
        "total_transaction_amount": float(amount.sum()),
        "fraud_transaction_amount": float(df.loc[df[TARGET] == 1, "Amount"].sum()),
        "note": "Duplicates are reported, not assumed to be data errors; no transaction ID exists.",
    }
    return summary, column_profile, duplicate_by_class


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Validate integrity; return a summary instead of silently fixing records."""
    summary, _, _ = quality_profile(df)
    failures = []
    if summary["missing_cells"]:
        failures.append(f"{summary['missing_cells']} missing cells")
    if summary["non_finite_cells"]:
        failures.append(f"{summary['non_finite_cells']} non-finite cells")
    labels = set(df[TARGET].unique().tolist())
    if not labels.issubset({0, 1}) or df[TARGET].isna().any():
        failures.append(f"target values must be 0/1; got {labels}")
    if summary["negative_amount_rows"]:
        failures.append(f"{summary['negative_amount_rows']} negative Amount values")
    if summary["negative_time_rows"]:
        failures.append(f"{summary['negative_time_rows']} negative Time values")
    if failures:
        raise ValueError("Data validation failed: " + "; ".join(failures))
    print(f"[clean] Validated: {summary['fraud_rows']:,} fraud rows "
          f"({100 * summary['fraud_rate']:.4f}%), "
          f"{summary['duplicate_rows_excluding_first']:,} duplicate rows, "
          f"{summary['zero_amount_rows']:,} zero-amount rows")
    return summary


def clean_data(df: pd.DataFrame, duplicates: str = "keep") -> pd.DataFrame:
    """Validate, reorder columns, optionally drop exact duplicates.

    Unlike the old code, this NEVER fits a scaler to the whole dataset.
    Original units are retained for truthful amount/time and value analyses.
    """
    validate_data(df)
    if duplicates not in {"keep", "drop"}:
        raise ValueError("duplicates must be either 'keep' or 'drop'")
    out = df.loc[:, EXPECTED_COLS].copy()
    if duplicates == "drop":
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
    return out


def scale_time_and_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated compatibility entry point: refuse leakage-prone global fitting."""
    raise RuntimeError(
        "Full-dataset scaling is disabled. Use the training-fold fitted scaler "
        "inside src.ml_models.make_preprocessor instead."
    )


def save_processed(df: pd.DataFrame, path: str | Path = PROCESSED_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[clean] Saved validated raw-unit data: {path} ({len(df):,} rows)")
    return path


def save_quality_reports(
    before: pd.DataFrame,
    after: pd.DataFrame,
    duplicates: str,
    directory: str | Path = QUALITY_DIR,
) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    before_summary, profile, duplicate_breakdown = quality_profile(before)
    after_summary, _, _ = quality_profile(after)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "duplicate_policy": duplicates,
        "original_units_retained": True,
        "before": before_summary,
        "after": after_summary,
        "rows_removed": int(len(before) - len(after)),
        "scaler_fitted": False,
    }
    paths = {
        "json": directory / "quality_summary.json",
        "columns": directory / "column_profile.csv",
        "duplicates": directory / "duplicate_breakdown.csv",
    }
    paths["json"].write_text(json.dumps(report, indent=2, default=_python_value), encoding="utf-8")
    profile.to_csv(paths["columns"], index=False)
    duplicate_breakdown.to_csv(paths["duplicates"], index=False)
    return paths


def run(
    raw_path: str | Path = RAW_PATH,
    processed_path: str | Path = PROCESSED_PATH,
    duplicates: str = "keep",
    report_dir: str | Path = QUALITY_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_raw(raw_path)
    cleaned = clean_data(raw, duplicates=duplicates)
    save_processed(cleaned, processed_path)
    report_paths = save_quality_reports(raw, cleaned, duplicates, report_dir)
    print("[clean] Quality reports:", *(str(p) for p in report_paths.values()), sep="\n  ")
    summary, _, _ = quality_profile(cleaned)
    return cleaned, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--duplicates", choices=["keep", "drop"], default="keep")
    parser.add_argument("--report-dir", type=Path, default=QUALITY_DIR)
    args = parser.parse_args()
    run(args.raw_path, args.output, args.duplicates, args.report_dir)


if __name__ == "__main__":
    main()
