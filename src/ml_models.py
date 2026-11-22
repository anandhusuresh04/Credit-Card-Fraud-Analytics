"""Reproducible, leakage-controlled fraud-detection experiments.

Run from the project root:
    python -m src.ml_models --split random
    python -m src.ml_models --split chronological
    python -m src.ml_models --split both --models LogisticRegression
    python -m src.ml_models --split random --keep-duplicates  # sensitivity ONLY

Default experiment: deduplicate exact identical records, use the raw CSV,
select MODEL and decision threshold using training-only out-of-fold (OOF)
predictions, then assess the locked policy on the untouched holdout.

The financial analysis is a SCENARIO, not measured bank savings. Amount units
are those of the source dataset; the CSV itself provides no currency metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
V_COLS = [f"V{i}" for i in range(1, 29)]
REQUIRED = ["Time", *V_COLS, "Amount", "Class"]
MODELS = ("LogisticRegression", "RandomForest", "XGBoost")
STRATEGIES = ("baseline", "SMOTE", "class_weight")


@dataclass
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    mode: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path, *, keep_duplicates: bool) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    if df.columns.duplicated().any() or set(df.columns) != set(REQUIRED) or len(df.columns) != len(REQUIRED):
        raise ValueError("Dataset must have exactly the 31 ULB columns (no extras or duplicates)")
    for col in REQUIRED:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Non-numeric column: {col}")
    if df[REQUIRED].isna().any().any() or not np.isfinite(df[REQUIRED].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite values present")
    if not df["Class"].isin([0, 1]).all() or not (df["Class"] == df["Class"].astype(int)).all():
        raise ValueError("Class must contain only 0/1")
    if (df["Amount"] < 0).any() or (df["Time"] < 0).any():
        raise ValueError("Negative Amount or Time values")
    df = df.reset_index().rename(columns={"index": "original_row"})
    duplicate = df[REQUIRED].duplicated(keep="first")
    info = {"input_rows": len(df), "exact_duplicate_rows": int(duplicate.sum()),
            "duplicate_fraud_rows": int(df.loc[duplicate, "Class"].sum()),
            "duplicate_handling": "keep" if keep_duplicates else "drop_before_split"}
    if not keep_duplicates:
        df = df.loc[~duplicate].copy()
    return df.reset_index(drop=True), info


def make_split(df: pd.DataFrame, mode: str, test_size: float, drop_time: bool) -> SplitData:
    if mode == "random":
        tr, te = train_test_split(df, test_size=test_size, stratify=df["Class"], random_state=SEED)
    elif mode == "chronological":
        ordered = df.sort_values(["Time", "original_row"], kind="stable")
        cut = int(len(ordered) * (1 - test_size))
        tr, te = ordered.iloc[:cut], ordered.iloc[cut:]
        # A split that cuts through identical timestamps cannot justify a strict
        # "future" evaluation. Drop the boundary timestamp from train.
        boundary = te["Time"].iloc[0]
        tr = tr.loc[tr["Time"] < boundary]
    else:
        raise ValueError(f"Unknown split {mode}")
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    if tr["Class"].nunique() != 2 or te["Class"].nunique() != 2:
        raise ValueError("Train and test each need positive and negative labels")
    features = V_COLS + ([] if drop_time else ["Time"]) + ["Amount"]
    return SplitData(tr, te, features, mode)


def make_pipeline(model: str, strategy: str, X: pd.DataFrame, y: np.ndarray):
    if model not in MODELS or strategy not in STRATEGIES:
        raise ValueError("Unsupported model or imbalance strategy")
    scale = [c for c in ("Time", "Amount") if c in X.columns]
    prep = ColumnTransformer([("scaled", StandardScaler(), scale)], remainder="passthrough")
    weights = strategy == "class_weight"
    if model == "LogisticRegression":
        clf = LogisticRegression(max_iter=3000, random_state=SEED,
                                 class_weight="balanced" if weights else None)
    elif model == "RandomForest":
        clf = RandomForestClassifier(n_estimators=120, min_samples_leaf=1,
                                     random_state=SEED, n_jobs=-1,
                                     class_weight="balanced" if weights else None)
    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost with pip install xgboost") from exc
        ratio = float(np.count_nonzero(y == 0) / max(np.count_nonzero(y == 1), 1))
        clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=.07,
                            subsample=0.9, colsample_bytree=.9, eval_metric="logloss",
                            tree_method="hist", random_state=SEED, n_jobs=-1,
                            scale_pos_weight=ratio if weights else 1.0)
    steps = [("preprocess", prep)]
    if strategy == "SMOTE":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ResamplingPipeline
        except ImportError as exc:
            raise ImportError("Install imbalanced-learn with pip install imbalanced-learn") from exc
        # SMOTE only sees the fitting partition, inside the pipeline.
        n_pos = int(np.count_nonzero(y == 1))
        if n_pos < 2:
            raise ValueError("SMOTE requires at least two fraud cases in each fit")
        steps += [("smote", SMOTE(sampling_strategy=.3,
                                   k_neighbors=min(5, n_pos - 1), random_state=SEED))]
        steps += [("classifier", clf)]
        return ResamplingPipeline(steps)
    return Pipeline(steps + [("classifier", clf)])


def fold_indices(sp: SplitData, n_folds: int):
    X, y = sp.train[sp.features], sp.train["Class"].to_numpy(dtype=int)
    if sp.mode == "random":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        yield from splitter.split(X, y)
    else:
        # Expanding-window validation: every training timestamp precedes each
        # validation timestamp. Some earliest training rows have no OOF score.
        splitter = TimeSeriesSplit(n_splits=n_folds)
        for a, b in splitter.split(X):
            # Drop the timestamp tied with validation start from training.
            limit = float(sp.train.iloc[b[0]]["Time"])
            a = a[sp.train.iloc[a]["Time"].to_numpy() < limit]
            yield a, b


def out_of_fold(sp: SplitData, model: str, strategy: str, folds: int):
    X = sp.train[sp.features]
    y = sp.train["Class"].to_numpy(dtype=int)
    oof = np.full(len(X), np.nan, dtype=float)
    fold_rows = []
    for i, (a, b) in enumerate(fold_indices(sp, folds), start=1):
        if len(a) == 0 or np.unique(y[a]).size < 2 or np.unique(y[b]).size < 2:
            raise ValueError(f"Fold {i} lacks both classes; reduce --cv-folds or change split")
        if sp.mode == "chronological" and not (sp.train.iloc[a]["Time"].max() < sp.train.iloc[b]["Time"].min()):
            raise AssertionError("Chronological validation violates time ordering")
        fitted = make_pipeline(model, strategy, X.iloc[a], y[a])
        fitted.fit(X.iloc[a], y[a])
        oof[b] = fitted.predict_proba(X.iloc[b])[:, 1]
        fold_rows.append({"fold": i, "train_rows": len(a), "validation_rows": len(b),
                          "train_fraud": int(y[a].sum()), "validation_fraud": int(y[b].sum()),
                          "fold_ap": average_precision_score(y[b], oof[b])})
    mask = np.isfinite(oof)
    if mask.sum() < 1 or len(np.unique(y[mask])) < 2:
        raise ValueError("Not enough scored OOF data to select a model")
    return oof, pd.DataFrame(fold_rows)


def threshold_table(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
                    cost_per_review: float, effectiveness: float,
                    friction_per_legit: float) -> pd.DataFrame:
    """Exact decision points (scores >= threshold); include flag none/all.

    Scenario: intercepted fraudulent value * effectiveness, minus the cost of
    every review, minus additional friction for flagged legitimate payments.
    It is NOT measured financial savings.
    """
    if len(y) != len(scores) or len(y) != len(amounts):
        raise ValueError("Mismatched threshold arrays")
    order = np.argsort(-scores, kind="stable")
    s, yy, aa = scores[order], y[order], amounts[order]
    # At the end of each equal-score group, all rows at that score are flagged.
    ends = np.flatnonzero(np.r_[s[:-1] != s[1:], True])
    cum_tp = np.cumsum(yy == 1)
    cum_fp = np.cumsum(yy == 0)
    cum_amount = np.cumsum(aa * (yy == 1))
    positives = int(y.sum())
    total_fraud_value = float(amounts[y == 1].sum())
    rows = [{"threshold": float(np.nextafter(s[0], np.inf)), "alerts": 0,
             "tp": 0, "fp": 0, "fn": positives, "precision": 0., "recall": 0., "f1": 0.,
             "fraud_value_recall": 0., "scenario_net_value": 0.}]
    for idx in ends:
        tp, fp, caught = int(cum_tp[idx]), int(cum_fp[idx]), float(cum_amount[idx])
        count = tp + fp
        rows.append({"threshold": float(s[idx]), "alerts": count,
                     "tp": tp, "fp": fp, "fn": positives - tp,
                     "precision": tp / count if count else 0.,
                     "recall": tp / positives if positives else 0.,
                     "f1": 2 * tp / (2 * tp + fp + positives - tp) if (2 * tp + fp + positives - tp) else 0.,
                     "fraud_value_recall": caught / total_fraud_value if total_fraud_value else 0.,
                     "scenario_net_value": caught * effectiveness - count * cost_per_review - fp * friction_per_legit})
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame, objective: str, capacity_per_1000: float | None,
                     n_scored: int) -> tuple[float, dict]:
    feasible = curve
    if capacity_per_1000 is not None:
        feasible = curve[curve["alerts"] * 1000 <= capacity_per_1000 * n_scored]
    if feasible.empty:
        raise ValueError("No feasible threshold for capacity restriction")
    metric = "f1" if objective == "f1" else "scenario_net_value"
    # Conservative tie-break: fewer alerts, then higher threshold.
    winner = feasible.sort_values([metric, "alerts", "threshold"],
                                  ascending=[False, True, False]).iloc[0]
    return float(winner["threshold"]), winner.to_dict()


def metric_row(y: np.ndarray, score: np.ndarray, threshold: float,
               amount: np.ndarray, review_cost: float, effectiveness: float,
               friction: float) -> dict:
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    caught = float(amount[(y == 1) & pred].sum())
    missed = float(amount[(y == 1) & ~pred].sum())
    total = caught + missed
    cost = float((tp + fp) * review_cost + fp * friction)
    result = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
              "precision": float(precision_score(y, pred, zero_division=0)),
              "recall": float(recall_score(y, pred, zero_division=0)),
              "f1": float(f1_score(y, pred, zero_division=0)),
              "pr_auc": float(average_precision_score(y, score)),
              "roc_auc": float(roc_auc_score(y, score)),
              "alerts": int(pred.sum()), "alerts_per_1000": float(pred.mean() * 1000),
              "flagged_fraud_amount": caught, "missed_fraud_amount": missed,
              "fraud_value_recall": caught / total if total else None,
              "scenario_review_and_friction_cost": cost,
              "scenario_net_value": caught * effectiveness - cost,
              "threshold": float(threshold)}
    if np.isfinite(score).all() and np.all((score >= 0) & (score <= 1)):
        result["brier_score"] = float(brier_score_loss(y, score))
    else:
        result["brier_score"] = None
    return result


def bootstrap_ci(y: np.ndarray, score: np.ndarray, threshold: float,
                 amount: np.ndarray, n_boot: int, seed: int = SEED) -> dict:
    """Uncertainty from row resampling only, NOT retraining or temporal drift."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values = {"pr_auc": [], "recall": [], "precision": [], "fraud_value_recall": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(n, size=n)
        yy, ss, aa = y[idx], score[idx], amount[idx]
        if yy.sum() == 0 or yy.sum() == n:
            continue
        p = ss >= threshold
        values["pr_auc"].append(average_precision_score(yy, ss))
        values["recall"].append(recall_score(yy, p, zero_division=0))
        values["precision"].append(precision_score(yy, p, zero_division=0))
        val = aa[yy == 1].sum()
        if val > 0:
            values["fraud_value_recall"].append(float(aa[(yy == 1) & p].sum() / val))
    out = {}
    for key, samples in values.items():
        lo, hi = np.percentile(samples, [2.5, 97.5]) if samples else (np.nan, np.nan)
        out[f"{key}_ci_low"] = float(lo)
        out[f"{key}_ci_high"] = float(hi)
    return out


def package_versions() -> dict:
    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__}
    for name in ("xgboost", "imblearn"):
        try:
            module = __import__(name)
            versions[name] = module.__version__
        except ImportError:
            versions[name] = "not installed"
    return versions


def run_one(mode: str, args) -> None:
    raw = Path(args.input)
    df, quality = load_data(raw, keep_duplicates=args.keep_duplicates)
    sp = make_split(df, mode, args.test_size, args.drop_time)
    y_train = sp.train["Class"].to_numpy(dtype=int)
    y_test = sp.test["Class"].to_numpy(dtype=int)
    a_train = sp.train["Amount"].to_numpy(dtype=float)
    a_test = sp.test["Amount"].to_numpy(dtype=float)
    X_train, X_test = sp.train[sp.features], sp.test[sp.features]
    candidates = []
    fold_tables = []
    oof_scores = {}
    if args.keep_duplicates:
        print("[ml] WARNING: duplicates retained: random train/test overlap may inflate results")
    print(f"[ml] {mode} | train={len(y_train):,} ({y_train.sum()} fraud) | "
          f"test={len(y_test):,} ({y_test.sum()} fraud) | "
          f"duplicates={'retained' if args.keep_duplicates else 'removed'}")
    for model in args.models:
        for strategy in args.strategies:
            key = f"{model}__{strategy}"
            start = time.perf_counter()
            oof, folds = out_of_fold(sp, model, strategy, args.cv_folds)
            eligible = np.isfinite(oof)
            curve = threshold_table(y_train[eligible], oof[eligible], a_train[eligible],
                                    args.review_cost, args.effectiveness, args.friction_cost)
            threshold, operating = choose_threshold(curve, args.threshold_objective,
                                                   args.max_alerts_per_1000, eligible.sum())
            ap = float(average_precision_score(y_train[eligible], oof[eligible]))
            candidates.append({"key": key, "model": model, "strategy": strategy,
                               "cv_oof_ap": ap, "cv_folds_mean_ap": float(folds.fold_ap.mean()),
                               "cv_folds_std_ap": float(folds.fold_ap.std(ddof=1)),
                               "oof_rows": int(eligible.sum()),
                               "threshold": threshold,
                               "cv_selected_f1": float(operating["f1"]),
                               "cv_selected_alerts_per_1000": float(operating["alerts"] * 1000 / eligible.sum()),
                               "cv_selected_scenario_net_value": float(operating["scenario_net_value"]),
                               "cv_seconds": round(time.perf_counter() - start, 2)})
            folds.insert(0, "key", key)
            fold_tables.append(folds)
            oof_scores[key] = oof
            print(f"[ml] CV {key}: pooled-OOF AP={ap:.4f}, threshold={threshold:.6g}, "
                  f"F1={operating['f1']:.4f}, OOF rows={eligible.sum():,}")
    cv = pd.DataFrame(candidates)
    # Select only on training CV. Never use holdout labels for selection.
    selected = cv.sort_values(["cv_oof_ap", "key"], ascending=[False, True]).iloc[0]
    key = str(selected["key"])
    print(f"[ml] LOCKED winner by pooled OOF AP: {key} (AP {selected['cv_oof_ap']:.4f})")
    final = make_pipeline(str(selected["model"]), str(selected["strategy"]), X_train, y_train)
    started = time.perf_counter()
    final.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    score = final.predict_proba(X_test)[:, 1]
    threshold = float(selected["threshold"])
    holdout = metric_row(y_test, score, threshold, a_test,
                         args.review_cost, args.effectiveness, args.friction_cost)
    holdout.update(bootstrap_ci(y_test, score, threshold, a_test, args.n_boot))
    holdout.update({"selected_model": key, "split": mode, "train_rows": len(y_train),
                    "test_rows": len(y_test), "test_fraud": int(y_test.sum()),
                    "fit_seconds": fit_time})

    run_settings = {"models": args.models, "strategies": args.strategies,
                    "test_size": args.test_size, "cv_folds": args.cv_folds,
                    "objective": args.threshold_objective,
                    "capacity": args.max_alerts_per_1000,
                    "review_cost": args.review_cost,
                    "effectiveness": args.effectiveness,
                    "friction": args.friction_cost,
                    "input_sha256": sha256(raw)}
    run_id = hashlib.sha256(json.dumps(run_settings, sort_keys=True).encode()).hexdigest()[:8]
    stamp = (f"{mode}_{'dedup' if not args.keep_duplicates else 'keepdup'}_"
             f"{'noTime' if args.drop_time else 'withTime'}_{run_id}")
    output = Path(args.output) / stamp
    model_dir = Path(args.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    cv.sort_values("cv_oof_ap", ascending=False).to_csv(output / "cv_model_selection.csv", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(output / "cv_fold_results.csv", index=False)
    # Scored OOF rows only; training labels are NEVER written alongside test labels in this file.
    oof_df = sp.train[["original_row", "Time", "Amount", "Class"]].copy()
    for model_key, oof in oof_scores.items():
        oof_df[f"oof_score__{model_key}"] = oof
    oof_df.to_csv(output / "training_oof_scores.csv", index=False)
    pred_df = sp.test[["original_row", "Time", "Amount", "Class"]].copy()
    pred_df["score"] = score
    pred_df["predicted_fraud"] = (score >= threshold).astype(int)
    pred_df["error_type"] = np.select(
        [(pred_df.Class == 1) & (pred_df.predicted_fraud == 1),
         (pred_df.Class == 1) & (pred_df.predicted_fraud == 0),
         (pred_df.Class == 0) & (pred_df.predicted_fraud == 1)],
        ["TP", "FN", "FP"], default="TN")
    pred_df.to_csv(output / "holdout_predictions.csv", index=False)
    # Threshold sensitivity on OOF predictions ONLY (not on holdout).
    eligible = np.isfinite(oof_scores[key])
    threshold_table(y_train[eligible], oof_scores[key][eligible], a_train[eligible],
                    args.review_cost, args.effectiveness, args.friction_cost).to_csv(
        output / "training_threshold_curve.csv", index=False)
    pd.DataFrame([holdout]).to_csv(output / "holdout_metrics.csv", index=False)
    # A constant-prior classifier represents the no-signal baseline. It is
    # descriptive only and never competes for the model-selection decision.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_score = dummy.predict_proba(X_test)[:, 1]
    pd.DataFrame([metric_row(y_test, dummy_score, .5, a_test,
                             args.review_cost, args.effectiveness,
                             args.friction_cost)]).to_csv(
        output / "dummy_baseline_metrics.csv", index=False)
    artifact = model_dir / f"best_model_{stamp}.joblib"
    joblib.dump({"pipeline": final, "threshold": threshold, "model": key,
                 "input_features": sp.features, "selection_criterion": "cv_oof_ap",
                 "threshold_objective": args.threshold_objective,
                 "amount_units": "source dataset units, currency not independently verified",
                 "scenario": {"review_cost": args.review_cost,
                              "effectiveness": args.effectiveness,
                              "friction_cost": args.friction_cost},
                 "metadata_path": str(output / "run_metadata.json")}, artifact)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "raw_csv_path": str(raw), "raw_csv_sha256": sha256(raw), "run_id": run_id,
                "data_quality": quality, "split": mode, "random_state": SEED,
                "test_size": args.test_size, "cv_folds": args.cv_folds,
                "cv_policy": "StratifiedKFold" if mode == "random" else "expanding TimeSeriesSplit",
                "train_rows": len(y_train), "test_rows": len(y_test),
                "drop_time": args.drop_time, "features": sp.features,
                "model_choices": args.models, "strategies": args.strategies,
                "selection": "highest pooled training out-of-fold average precision",
                "selected_model": key, "selected_threshold": threshold,
                "threshold_objective": args.threshold_objective,
                "capacity_alerts_per_1000": args.max_alerts_per_1000,
                "scenario_assumptions": {"cost_per_review": args.review_cost,
                                         "intervention_effectiveness": args.effectiveness,
                                         "legitimate_transaction_friction_cost": args.friction_cost},
                "bootstrap_resamples": args.n_boot,
                "limitations": ["Observations may be dependent and are anonymized",
                                "No real intervention outcomes or bank loss figures",
                                "Bootstrap excludes retraining variation and temporal drift",
                                "Chronological CV uses expanding windows; earlier data have no OOF predictions",
                                "No hyperparameter search or probability calibration",
                                "The holdout is evaluated once only for the CV-selected model"],
                "library_versions": package_versions()}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[ml] HOLDOUT AP={holdout['pr_auc']:.4f} F1={holdout['f1']:.4f} "
          f"P={holdout['precision']:.4f} R={holdout['recall']:.4f} "
          f"value_recall={holdout['fraud_value_recall']:.4f} "
          f"alerts/1000={holdout['alerts_per_1000']:.3f}")
    print(f"[ml] Saved reports -> {output} | model -> {artifact}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", default="data/raw/creditcard.csv")
    p.add_argument("--output", default="reports/tables/ml")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--split", choices=("random", "chronological", "both"), default="random")
    p.add_argument("--test-size", type=float, default=.30)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    p.add_argument("--keep-duplicates", action="store_true", help="sensitivity test ONLY; duplicates retained")
    p.add_argument("--drop-time", action="store_true")
    p.add_argument("--threshold-objective", choices=("f1", "scenario_net_value"), default="f1")
    p.add_argument("--max-alerts-per-1000", type=float, default=None)
    p.add_argument("--review-cost", type=float, default=5.0)
    p.add_argument("--effectiveness", type=float, default=0.5)
    p.add_argument("--friction-cost", type=float, default=0.0)
    p.add_argument("--n-boot", type=int, default=200)
    args = p.parse_args(argv)
    if not 0.05 <= args.test_size <= .5 or args.cv_folds < 2 or args.n_boot < 0:
        p.error("test size must be 0.05..0.5, CV folds >= 2, bootstrap >= 0")
    if args.review_cost < 0 or args.friction_cost < 0 or not 0 <= args.effectiveness <= 1:
        p.error("review/friction costs must be >=0; intervention effectiveness must be 0..1")
    if args.max_alerts_per_1000 is not None and not 0 <= args.max_alerts_per_1000 <= 1000:
        p.error("max alerts per 1000 must be in 0..1000")
    return args


def main(argv=None):
    args = parse_args(argv)
    modes = ("random", "chronological") if args.split == "both" else (args.split,)
    for mode in modes:
        run_one(mode, args)


if __name__ == "__main__":
    main()
