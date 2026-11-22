"""Training-selected batch investigation allocation for ULB credit-card fraud.

Uses ONLY the already-selected ML model's training out-of-fold (OOF) scores
for policy selection. Holdout labels are never used to choose a policy.

Policy: rank the entire evaluated batch and take top floor(N * capacity/10000)
transactions. This is batch queue allocation, NOT streaming point-in-time
thresholding; batch size and all scores must be available before review.

Usage:
 python -m src.investigation_optimization --run-dir reports/tables/ml/chronological_dedup_withTime_463fcded
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POLICIES = ('fraud_score', 'score_times_amount')
DEFAULT_CAPACITIES = (5., 10., 20., 50.)
COLS = {'original_row', 'Class', 'Amount'}


def parse_capacities(s: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in s.split(','))
    if not vals or any(not np.isfinite(x) or x <= 0 or x > 10000 for x in vals):
        raise ValueError('Capacities must be finite, positive, <=10000 per 10,000')
    if len(set(vals)) != len(vals):
        raise ValueError('Capacity points must be unique')
    return tuple(sorted(vals))


def validated_frame(df: pd.DataFrame, score_col: str, name: str) -> pd.DataFrame:
    missing = (COLS | {score_col}) - set(df.columns)
    if missing:
        raise ValueError(f'{name}: required columns missing: {sorted(missing)}')
    d = df.copy()
    if d.empty or d[['original_row','Class','Amount',score_col]].isna().any().any():
        raise ValueError(f'{name}: empty or missing values')
    if d['original_row'].duplicated().any():
        raise ValueError(f'{name}: duplicate original_row IDs')
    for col in ['Class','Amount',score_col]:
        d[col] = pd.to_numeric(d[col],errors='raise')
    if not np.isfinite(d[['Class','Amount',score_col]].to_numpy(dtype=float)).all():
        raise ValueError(f'{name}: nonfinite data')
    if not set(d.Class.unique()).issubset({0,1}):
        raise ValueError(f'{name}: Class must be binary')
    if (d.Amount < 0).any() or ((d[score_col] < 0)|(d[score_col] > 1)).any():
        raise ValueError(f'{name}: invalid amount/score')
    if not (d.Class.eq(0).any() and d.Class.eq(1).any()):
        raise ValueError(f'{name}: both classes must be represented')
    return d


def allocations(frame: pd.DataFrame, score_col: str, capacity: float, policy: str) -> np.ndarray:
    """Deterministic top-k by score or amount-weighted score, ties by row ID."""
    if policy not in POLICIES:
        raise ValueError(f'Unknown policy {policy}')
    n = len(frame)
    k = math.floor((n * capacity / 10000.0) + 1e-10)
    score = frame[score_col].to_numpy(dtype=float)
    value = score if policy == 'fraud_score' else score * frame.Amount.to_numpy(dtype=float)
    # Stable deterministic tie-break independent of input row order.
    rid = frame.original_row.to_numpy()
    order = np.lexsort((rid, -value))
    chosen = np.zeros(n, dtype=bool)
    chosen[order[:k]] = True
    assert int(chosen.sum()) == k
    return chosen


def outcomes(df: pd.DataFrame, flag: np.ndarray, capacity: float, policy: str,
             review_cost: float, effectiveness: float, friction_cost: float) -> dict:
    y = df.Class.to_numpy(dtype=int)
    amt = df.Amount.to_numpy(dtype=float)
    count = int(len(df)); nfraud = int(y.sum()); nalert = int(flag.sum())
    tp = int((flag & (y==1)).sum()); fp = nalert-tp
    fraud_value = float(amt[y==1].sum()); caught = float(amt[flag & (y==1)].sum())
    legit_flagged = float(amt[flag & (y==0)].sum())
    cost = nalert*review_cost + fp*friction_cost
    benefit = caught*effectiveness
    # Expected prevented amount is a scenario, not realized savings.
    return dict(policy=policy, capacity_per_10000=capacity, transactions=count,
        allowed_alerts=math.floor((count*capacity/10000)+1e-10), alerts=nalert,
        alerts_per_10000=10000*nalert/count, tp=tp, fp=fp, fn=nfraud-tp,
        fraud_cases=nfraud, review_yield_pct=100*tp/nalert if nalert else 0.,
        case_recall_pct=100*tp/nfraud, fraud_value_total=fraud_value,
        fraud_value_flagged=caught, fraud_value_missed=fraud_value-caught,
        value_recall_pct=100*caught/fraud_value if fraud_value else 0.,
        legitimate_value_flagged=legit_flagged,
        review_cost_total=nalert*review_cost, friction_cost_total=fp*friction_cost,
        scenario_gross_benefit=benefit, scenario_net_value=benefit-cost,
        breakeven_effectiveness=cost/caught if caught > 0 else float('nan'),
        # Meaning of >1: 100% intervention effectiveness insufficient to break even.
        breakeven_feasible_at_100pct=bool(caught > 0 and cost <= caught))


def grid(df: pd.DataFrame, score_col: str, capacities, review_cost, effectiveness, friction_cost):
    rows=[]
    for cap in capacities:
        for policy in POLICIES:
            flag=allocations(df,score_col,cap,policy)
            rows.append(outcomes(df,flag,cap,policy,review_cost,effectiveness,friction_cost))
    return pd.DataFrame(rows)


def training_choice(train_grid: pd.DataFrame, objective: str) -> pd.DataFrame:
    """Only training OOF outcome metrics determine winner for each budget."""
    obj={'value_recall':'value_recall_pct','case_recall':'case_recall_pct',
         'net_value':'scenario_net_value'}[objective]
    chosen=[]
    for _, part in train_grid.groupby('capacity_per_10000',sort=True):
        # If ties, prefer the simpler fraud_score ranking; deterministic.
        part=part.assign(tie_priority=part.policy.map({'fraud_score':0,'score_times_amount':1}))
        winner=part.sort_values([obj,'tie_priority'],ascending=[False,True]).iloc[0]
        chosen.append({'capacity_per_10000':float(winner.capacity_per_10000),
                       'selected_policy':str(winner.policy),
                       'training_objective':objective,
                       'training_objective_value':float(winner[obj]),
                       'training_alerts':int(winner.alerts)})
    return pd.DataFrame(chosen)


def audit_model_selection(run_dir: Path, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel=pd.read_csv(run_dir/'cv_model_selection.csv')
    if not {'key','cv_oof_ap','threshold'}.issubset(sel):
        raise ValueError('Incomplete cv_model_selection.csv')
    if len(sel)==0 or sel.key.duplicated().any() or sel.cv_oof_ap.isna().any():
        raise ValueError('Invalid CV selection table')
    key=str(meta['selected_model'])
    if key != str(sel.sort_values(['cv_oof_ap','key'],ascending=[False,True]).iloc[0].key):
        raise ValueError('Model not selected from training OOF AP')
    selected=sel[sel.key==key]
    if len(selected)!=1 or not np.isclose(float(selected.iloc[0].threshold),float(meta['selected_threshold']),rtol=0,atol=1e-8):
        raise ValueError('Selected threshold differs from CV record')
    return sel,selected


def read_inputs(run_dir: Path):
    for name in ['run_metadata.json','training_oof_scores.csv','holdout_predictions.csv','cv_model_selection.csv']:
        if not (run_dir/name).is_file():
            raise FileNotFoundError(f'Missing {run_dir/name}')
    meta=json.loads((run_dir/'run_metadata.json').read_text(encoding='utf-8'))
    audit_model_selection(run_dir,meta)
    oof=pd.read_csv(run_dir/'training_oof_scores.csv')
    hold=pd.read_csv(run_dir/'holdout_predictions.csv')
    selected=str(meta['selected_model'])
    name='oof_score__'+selected
    if name not in oof.columns:
        raise ValueError(f'Missing selected-model training OOF scores: {name}')
    # Null OOF scores are expected ONLY for the unscored initial chronological window.
    if oof[name].notna().sum()==0:
        raise ValueError('No scored training OOF rows')
    train=validated_frame(oof.loc[oof[name].notna()],name,'training OOF')
    test=validated_frame(hold,'score','holdout')
    if len(oof)!=int(meta['train_rows']) or len(hold)!=int(meta['test_rows']):
        raise ValueError('Training or holdout count contradicts metadata')
    if int(meta['train_rows']) <= len(train)-1:
        raise ValueError('OOF rows exceed training population')
    if set(oof.original_row).intersection(set(hold.original_row)):
        raise ValueError('Training/holdout original_row overlap')
    if meta.get('split')=='chronological':
        if 'Time' not in oof or 'Time' not in hold:
            raise ValueError('Missing chronological timestamps')
        if float(oof.Time.max()) >= float(hold.Time.min()):
            raise ValueError('Chronological split has non-past training timestamps')
    if not {'predicted_fraud','error_type'}.issubset(hold):
        raise ValueError('Holdout saved decision labels missing')
    locked=(test.score.to_numpy()>=float(meta['selected_threshold'])).astype(int)
    if not np.array_equal(locked,hold.predicted_fraud.to_numpy(dtype=int)):
        raise ValueError('Saved decision contradicts locked threshold')
    if meta.get('split') == 'chronological' and not test.Time.is_monotonic_increasing:
        raise ValueError('Holdout time not increasing')
    actual=test.Class.to_numpy(dtype=int)
    labels=np.select([(actual==0)&(locked==0),(actual==0)&(locked==1),
                      (actual==1)&(locked==0),(actual==1)&(locked==1)],
                     ['TN','FP','FN','TP'],default='INVALID')
    if not np.array_equal(labels,hold.error_type.to_numpy()):
        raise ValueError('Saved error_type values contradict locked predictions')
    # This module should not run on an experiment whose saved holdout metrics
    # disagree with the row-level prediction source.
    if (run_dir/'holdout_metrics.csv').is_file():
        hmetrics=pd.read_csv(run_dir/'holdout_metrics.csv')
        if len(hmetrics)!=1:
            raise ValueError('Expected one holdout metrics row')
        r=hmetrics.iloc[0]
        observed={'tp':int(((actual==1)&(locked==1)).sum()),
                  'fp':int(((actual==0)&(locked==1)).sum()),
                  'fn':int(((actual==1)&(locked==0)).sum()),
                  'tn':int(((actual==0)&(locked==0)).sum()),
                  'alerts':int(locked.sum())}
        for k,v in observed.items():
            if k not in r or int(r[k])!=v:
                raise ValueError(f'Saved holdout {k} inconsistent with predictions')
    return meta,train,test,name


def chart_frontier(df: pd.DataFrame, chosen: pd.DataFrame, output: Path, formats):
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.3))
    colors={'fraud_score':'#0C8190','score_times_amount':'#C1812F'}
    labels={'fraud_score':'Fraud-score ranking','score_times_amount':'Score × amount ranking'}
    for policy in POLICIES:
        sub=df[df.policy==policy].sort_values('capacity_per_10000')
        axs[0].plot(sub.capacity_per_10000,sub.case_recall_pct,'o-',label=labels[policy],color=colors[policy],lw=2)
        axs[1].plot(sub.capacity_per_10000,sub.value_recall_pct,'o-',label=labels[policy],color=colors[policy],lw=2)
    for ax,title in zip(axs,['Fraud-case recall','Recorded fraud-value recall']):
        ax.set(title=title,xlabel='Review allowance per 10,000 transactions',ylabel='Coverage (%)',ylim=(0,105))
        ax.grid(alpha=.2)
    axs[0].legend(loc='lower right',fontsize=9)
    fig.suptitle('Holdout batch-queue coverage by review capacity',fontsize=15,fontweight='bold')
    fig.text(.5,.015,'Policies predeclared; selected policy per capacity chosen on training OOF only. Not streaming or realized savings.',ha='center',fontsize=9,color='#556779')
    fig.tight_layout(rect=[0,.065,1,.93])
    for ext in formats:fig.savefig(output/f'01_capacity_frontier.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def chart_economics(df: pd.DataFrame, output: Path, formats):
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.1))
    cols={'fraud_score':'#0C8190','score_times_amount':'#C1812F'}
    for policy in POLICIES:
        sub=df[df.policy==policy].sort_values('capacity_per_10000')
        axs[0].plot(sub.capacity_per_10000,sub.scenario_net_value,'o-',color=cols[policy],label=policy)
        axs[1].plot(sub.capacity_per_10000,sub.legitimate_value_flagged,'o-',color=cols[policy],label=policy)
    axs[0].axhline(0,color='black',lw=.8,ls='--')
    axs[0].set(title='Hypothetical net value',ylabel='Source dataset units')
    axs[1].set(title='Legitimate transaction value flagged',ylabel='Source dataset units')
    for ax in axs:
        ax.set_xlabel('Review allowance per 10,000 transactions')
        ax.grid(alpha=.2)
    axs[0].legend(['Fraud score','Score × amount'],fontsize=9)
    fig.suptitle('Scenario economics and legitimate-value exposure',fontsize=15,fontweight='bold')
    fig.text(.5,.015,'Assumed effectiveness and review/friction costs are hypothetical. Flagging is not equivalent to blocking.',ha='center',fontsize=9,color='#556779')
    fig.tight_layout(rect=[0,.065,1,.92])
    for ext in formats:fig.savefig(output/f'02_economics_exposure.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def run(args):
    run_dir=Path(args.run_dir)
    meta,train,hold,score_name=read_inputs(run_dir)
    caps=parse_capacities(args.capacities)
    for v,name in [(args.review_cost,'review cost'),(args.friction_cost,'friction cost')]:
        if not np.isfinite(v) or v<0:raise ValueError(name+' must be finite and nonnegative')
    if not np.isfinite(args.effectiveness) or not 0<=args.effectiveness<=1:
        raise ValueError('Effectiveness must be between 0 and 1')
    out=Path(args.output) if args.output else Path('reports/tables/investigation_optimization')/run_dir.name
    figdir=Path(args.figures) if args.figures else Path('reports/figures/investigation_optimization')/run_dir.name
    if out.resolve()==run_dir.resolve():
        raise ValueError('Never write optimization outputs into the source ML run directory')
    train_g=grid(train,score_name,caps,args.review_cost,args.effectiveness,args.friction_cost)
    selected=training_choice(train_g,args.objective)
    hold_g=grid(hold,'score',caps,args.review_cost,args.effectiveness,args.friction_cost)
    selected_hold=hold_g.merge(selected,on='capacity_per_10000',validate='many_to_one')
    selected_hold=selected_hold.loc[selected_hold.policy==selected_hold.selected_policy].drop(columns='selected_policy')
    if len(selected_hold)!=len(caps):raise RuntimeError('Selection merge failed')
    # Verify constraints and economic reconciliations before writing anything.
    for df in (train_g,hold_g):
        if (df.alerts>df.allowed_alerts).any() or (df.tp+df.fp!=df.alerts).any():
            raise RuntimeError('Capacity or confusion counts failed validation')
        if not np.allclose(df.fraud_value_flagged+df.fraud_value_missed,df.fraud_value_total):
            raise RuntimeError('Fraud value does not reconcile')
        if not np.allclose(df.scenario_net_value,df.scenario_gross_benefit-df.review_cost_total-df.friction_cost_total):
            raise RuntimeError('Scenario cost mismatch')
    out.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    train_g.to_csv(out/'training_oof_policy_grid.csv',index=False)
    selected.to_csv(out/'training_selected_policies.csv',index=False)
    hold_g.to_csv(out/'holdout_policy_comparison.csv',index=False)
    selected_hold.to_csv(out/'holdout_training_selected_policies.csv',index=False)
    m={'source_run_dir':str(run_dir),'source_split':meta['split'],
       'source_run_id':meta.get('run_id'),'source_model':meta['selected_model'],
       'source_dataset_sha256':meta.get('raw_csv_sha256'),
       'training_scored_oof_rows':int(len(train)),'training_unscored_rows':int(meta['train_rows'])-len(train),
       'holdout_rows':int(len(hold)),'training_objective':args.objective,
       'capacities_per_10000':caps,'scenario':{'review_cost':args.review_cost,
       'effectiveness':args.effectiveness,'false_positive_friction_cost':args.friction_cost},
       'policies':{'fraud_score':'rank saved selected-model score',
                   'score_times_amount':'rank saved selected-model score multiplied by Amount'},
       'methodology':['Policy family and capacity budgets are predeclared.',
          'Winning ranking strategy at each budget selected from training OOF rows ONLY.',
          'Both predeclared policies shown on holdout; do NOT pick a winner using holdout outcomes.',
          'For each evaluated batch, top floor(N*capacity/10000) is reviewed; tied ranks resolved by original_row.',
          'This batch policy uses all evaluation-batch scores; not an online threshold and not a point-in-time stream simulation.',
          'Scores may not be calibrated probabilities; multiplying score by amount is a heuristic, not expected recoverable loss.',
          'Training OOF period mix may be unstable; selected strategy is not proven optimal for the future.',
          'Model selection and policy comparison reuse OOF labels; training estimates may be optimistic.',
          'Test annotations are retrospective; this experiment does not justify retuning on test.',
          'Dataset currency, real costs, intervention outcomes, customer friction and repeated-customer dependence unknown.']}
    (out/'methodology.json').write_text(json.dumps(m,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Investigation Capacity Optimization — Retrospective Analysis','',
       f"Source run: `{run_dir}` · {meta['split']} split · {meta['selected_model']}",
       '',f'**Train-only policy-selection objective:** {args.objective}.',
       f'**Batch capacities:** {", ".join(str(v) for v in caps)} alerts per 10,000 transactions.',
       '', '## Holdout outcomes for policies selected on training OOF','',
       '| Allowance / 10k | Selected ranking | Alerts | Fraud cases caught | Case recall | Fraud-value recall | Scenario net value |',
       '|---:|---|---:|---:|---:|---:|---:|']
    for row in selected_hold.itertuples():
        lines.append(f'| {row.capacity_per_10000:g} | {row.policy} | {row.alerts:,} | {row.tp:,} | {row.case_recall_pct:.2f}% | {row.value_recall_pct:.2f}% | {row.scenario_net_value:,.2f} |')
    lines+=['','## Critical limitations','',
       '- These are review queues, NOT automatic declines or demonstrated prevented losses.',
       '- Review caps are applied to a full batch; this assumes the entire batch score ranking is available, not real-time decisions.',
       '- The winning ranking rule was chosen with training OOF labels. The holdout comparison is descriptive only.',
       '- Financial numbers are illustrative assumptions in the original dataset amount units; currency is unverified.',
       '- Model scores have not been calibrated. Score × amount is only a heuristic prioritization rule.',
       '- Small holdout fraud counts and temporal change introduce considerable uncertainty.']
    (out/'executive_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    chart_frontier(hold_g,selected_hold,figdir,args.format)
    chart_economics(hold_g,figdir,args.format)
    print(f'[optimization] PASS — {meta["split"]} / {meta["selected_model"]} | training OOF={len(train):,} | holdout={len(hold):,}')
    for row in selected_hold.itertuples():
        print(f'[optimization] {row.capacity_per_10000:g}/10k: {row.policy} | alerts={row.alerts:,} | cases={row.tp}/{row.fraud_cases} | value={row.value_recall_pct:.2f}% | scenario net={row.scenario_net_value:,.2f}')
    print(f'[optimization] 4 CSVs, methodology.json, executive_summary.md -> {out}')
    print(f'[optimization] 2 figures ({", ".join(args.format)}) -> {figdir}')
    return selected_hold


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--capacities',default='5,10,20,50',help='Review slots per 10,000 transactions')
    p.add_argument('--objective',choices=['value_recall','case_recall','net_value'],default='value_recall')
    p.add_argument('--review-cost',type=float,default=5.)
    p.add_argument('--effectiveness',type=float,default=.5)
    p.add_argument('--friction-cost',type=float,default=0.)
    p.add_argument('--output');p.add_argument('--figures')
    p.add_argument('--format',nargs='+',choices=['png','svg'],default=['png'])
    return run(p.parse_args(argv))

if __name__=='__main__':main()
