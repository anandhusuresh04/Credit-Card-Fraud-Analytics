"""Validated, read-only access to existing fraud-analytics report artifacts.

No model fitting, threshold tuning, or use of holdout labels for policy selection.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

RUN_REQUIRED = ("run_metadata.json", "holdout_metrics.csv")
OPT_REQUIRED = ("holdout_policy_comparison.csv", "training_selected_policies.csv")

class DataIssue(ValueError):
    pass


def read_csv(root: Path, relative: str, required=(), *, optional=False) -> pd.DataFrame | None:
    path = (root / relative)
    if not path.is_file():
        if optional: return None
        raise DataIssue(f"Missing report: {relative}. Generate it with the corresponding src module.")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DataIssue(f"Cannot read {relative}: {exc}") from exc
    if frame.empty:
        raise DataIssue(f"Empty report: {relative}")
    absent = set(required) - set(frame.columns)
    if absent:
        raise DataIssue(f"{relative} lacks expected columns: {sorted(absent)}")
    return frame


def read_json(root: Path, relative: str, *, optional=False):
    file = root / relative
    if not file.is_file():
        if optional:return None
        raise DataIssue(f"Missing report: {relative}")
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataIssue(f"Cannot parse {relative}: {exc}") from exc


def discover_runs(root: Path):
    base = root / 'reports/tables/ml'
    if not base.is_dir(): return []
    return sorted([p.name for p in base.iterdir() if p.is_dir() and all((p/f).is_file() for f in RUN_REQUIRED)])


def get_run(root: Path, run: str):
    if run not in discover_runs(root):
        raise DataIssue(f"ML run not found or incomplete: {run}")
    prefix=f'reports/tables/ml/{run}/'
    meta=read_json(root, prefix+'run_metadata.json')
    df=read_csv(root,prefix+'holdout_metrics.csv',
        {'selected_model','split','test_rows','test_fraud','tp','fp','fn','tn','pr_auc','precision','recall','f1','fraud_value_recall','threshold','alerts_per_1000'})
    if len(df)!=1: raise DataIssue('Exactly one holdout_metrics row required per experiment')
    m=df.iloc[0]
    if str(meta.get('selected_model'))!=str(m.selected_model) or str(meta.get('split'))!=str(m.split):
        raise DataIssue('Run metadata does not match holdout metrics')
    if int(m.tp)+int(m.fn)!=int(m.test_fraud) or int(m.tp)+int(m.fp)+int(m.fn)+int(m.tn)!=int(m.test_rows):
        raise DataIssue('Confusion matrix does not reconcile with holdout population')
    expected_precision=float(m.tp)/(float(m.tp)+float(m.fp)) if (m.tp+m.fp) else 0.0
    expected_recall=float(m.tp)/(float(m.tp)+float(m.fn)) if (m.tp+m.fn) else 0.0
    expected_f1=2*expected_precision*expected_recall/(expected_precision+expected_recall) if expected_precision+expected_recall else 0.0
    for key, expected in (('precision', expected_precision),('recall', expected_recall),('f1', expected_f1)):
        if not np.isclose(float(m[key]),expected,atol=1e-6):
            raise DataIssue(f'{key} does not reconcile with confusion counts')
    for c in ('precision','recall','f1','pr_auc','fraud_value_recall'):
        if not np.isfinite(float(m[c])) or not 0<=float(m[c])<=1:
            raise DataIssue(f'Invalid {c} in holdout metrics')
    return m,meta


def get_optimization(root: Path, run: str):
    base=f'reports/tables/investigation_optimization/{run}/'
    allp=read_csv(root,base+'holdout_policy_comparison.csv',
        {'policy','capacity_per_10000','transactions','alerts','tp','fp','case_recall_pct','value_recall_pct','fraud_value_flagged','legitimate_value_flagged','scenario_net_value','fraud_value_total'})
    chosen=read_csv(root,base+'training_selected_policies.csv',
        {'capacity_per_10000','selected_policy','training_objective'})
    if allp.duplicated(['capacity_per_10000','policy']).any() or chosen.capacity_per_10000.duplicated().any():
        raise DataIssue('Duplicated capacity/policy observations')
    if set(allp.policy)!={'fraud_score','score_times_amount'}:
        raise DataIssue('Expected both predefined ranking policies')
    for cap, part in allp.groupby('capacity_per_10000'):
        if len(part)!=2 or cap not in set(chosen.capacity_per_10000):
            raise DataIssue('Expected two policies and one training selection at every capacity')
        if part.transactions.nunique()!=1 or part.fraud_value_total.nunique()!=1:
            raise DataIssue('Policy rows do not refer to the same holdout')
        if any(part.alerts > np.floor(part.transactions*cap/10000+1e-9)):
            raise DataIssue('Reported reviews exceed configured capacity')
        selected=chosen.loc[chosen.capacity_per_10000==cap,'selected_policy'].iloc[0]
        if selected not in set(part.policy): raise DataIssue('Unknown training-selected policy')
    if set(chosen.capacity_per_10000)!=set(allp.capacity_per_10000):
        raise DataIssue('Training selections and holdout capacity grids differ')
    return allp.sort_values(['capacity_per_10000','policy']), chosen.sort_values('capacity_per_10000')


def scenario_economics(row, *, review_cost: float, effectiveness: float, friction_cost: float) -> dict:
    """Pure counterfactual accounting on fixed observed alerts; never reselects a policy."""
    if any(not np.isfinite(x) for x in (review_cost,effectiveness,friction_cost)) or review_cost<0 or friction_cost<0 or not 0<=effectiveness<=1:
        raise DataIssue('Invalid scenario assumptions')
    gross=float(row['fraud_value_flagged'])*effectiveness
    cost=float(row['alerts'])*review_cost+float(row['fp'])*friction_cost
    return {'gross':gross,'cost':cost,'net':gross-cost,'breakeven_effectiveness':cost/float(row['fraud_value_flagged']) if float(row['fraud_value_flagged'])>0 else np.nan}


def verify_business_consistency(kpi: pd.Series, amount: pd.DataFrame):
    expected={'transactions','fraud_cases','recorded_fraud_value','fraud_rate_pct'}
    if not expected.issubset(kpi.index): raise DataIssue('Business KPI columns missing')
    cols={'transactions','fraud_cases','fraud_amount'}
    if not cols.issubset(amount.columns):raise DataIssue('Amount exposure columns missing')
    if int(amount.transactions.sum())!=int(kpi.transactions) or int(amount.fraud_cases.sum())!=int(kpi.fraud_cases):
        raise DataIssue('Amount bands do not reconcile to business KPI totals')
    if not np.isclose(amount.fraud_amount.sum(),float(kpi.recorded_fraud_value),atol=.011):
        raise DataIssue('Amount bands do not reconcile to recorded fraud value')


def fnum(v, decimals=0):
    return f'{float(v):,.{decimals}f}'
