"""Artifact-only, holdout-safe fraud model evaluation.

Requires the run-specific output folder from ``python -m src.ml_models``.
Never fits a model, changes an operating threshold, or chooses a model using
holdout labels. All plots are retrospective diagnostics, not policy optimization.

Usage (from project root):
    python -m src.model_evaluation --run-dir reports/tables/ml/chronological_dedup_withTime_463fcded
    python -m src.model_evaluation --run-dir reports/tables/ml/chronological_dedup_withTime_463fcded --format png svg
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve, roc_curve)

NAVY, TEAL, RED, GREY, GOLD = "#14263E", "#087E8B", "#C94F50", "#66788A", "#C88A32"
plt.rcParams.update({"figure.dpi":125, "savefig.dpi":220, "font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "axes.grid":True,"grid.alpha":0.16,"axes.titleweight":"bold"})
REQUIRED_PRED = {"original_row", "Time", "Amount", "Class", "score", "predicted_fraud", "error_type"}
REQUIRED_METRIC = {"tn", "fp", "fn", "tp", "precision", "recall", "f1", "pr_auc", "roc_auc",
                   "brier_score", "threshold", "alerts", "alerts_per_1000", "flagged_fraud_amount",
                   "missed_fraud_amount", "fraud_value_recall", "selected_model", "split", "test_rows", "test_fraud"}


def read_artifacts(run_dir: Path):
    names = ["holdout_predictions.csv", "holdout_metrics.csv", "cv_model_selection.csv", "cv_fold_results.csv", "run_metadata.json"]
    absent = [s for s in names if not (run_dir / s).is_file()]
    if absent:
        raise FileNotFoundError(f"Incomplete ML run in {run_dir}: missing {', '.join(absent)}")
    predictions = pd.read_csv(run_dir / names[0])
    metrics = pd.read_csv(run_dir / names[1])
    selection = pd.read_csv(run_dir / names[2])
    folds = pd.read_csv(run_dir / names[3])
    meta = json.loads((run_dir / names[4]).read_text(encoding="utf-8"))
    return predictions, metrics, selection, folds, meta


def audit_run(predictions: pd.DataFrame, metrics: pd.DataFrame, selection: pd.DataFrame,
              folds: pd.DataFrame, meta: dict) -> dict:
    """Fail closed if observed predictions contradict the saved experiment."""
    if not REQUIRED_PRED.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(REQUIRED_PRED - set(predictions.columns))}")
    if not REQUIRED_METRIC.issubset(metrics.columns) or len(metrics) != 1:
        raise ValueError("Expected exactly one complete selected holdout metrics row")
    if not {"key", "cv_oof_ap", "threshold"}.issubset(selection.columns):
        raise ValueError("Incomplete training model-selection table")
    if not {"key", "fold", "fold_ap", "train_rows", "validation_rows"}.issubset(folds.columns):
        raise ValueError("Incomplete CV fold table")
    if len(predictions) == 0 or predictions.isna().any().any():
        raise ValueError("Empty predictions or missing prediction values")
    if predictions.original_row.duplicated().any():
        raise ValueError("Duplicate original_row identifiers in holdout")
    y = predictions["Class"].to_numpy(); p = predictions["predicted_fraud"].to_numpy()
    s = predictions["score"].to_numpy(dtype=float); a = predictions["Amount"].to_numpy(dtype=float)
    if not set(np.unique(y)).issubset({0,1}) or not set(np.unique(p)).issubset({0,1}):
        raise ValueError("Class and predictions must be binary 0/1")
    if not (np.isfinite(s).all() and np.isfinite(a).all() and (a >= 0).all() and ((s >= 0) & (s <= 1)).all()):
        raise ValueError("Nonfinite/negative amount or invalid probability score")
    row=metrics.iloc[0]; thr=float(meta["selected_threshold"])
    if not np.isclose(float(row["threshold"]),thr,atol=1e-8,rtol=0):
        raise ValueError("Threshold disagrees with run metadata")
    if not np.array_equal(p,(s>=thr).astype(int)):
        raise ValueError("Predicted labels do not match saved locked threshold")
    truth=np.select([(y==0)&(p==0),(y==0)&(p==1),(y==1)&(p==0),(y==1)&(p==1)],
                    ["TN","FP","FN","TP"],default="??")
    if not np.array_equal(truth,predictions.error_type.to_numpy()):
        raise ValueError("Invalid saved error_type labels")
    tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*tp/max(2*tp+fp+fn,1)
    total_fraud=float(a[y==1].sum()); flagged=float(a[(y==1)&(p==1)].sum())
    actual={"tn":tn,"fp":fp,"fn":fn,"tp":tp,"precision":prec,"recall":rec,"f1":f1,
            "pr_auc":float(average_precision_score(y,s)),"roc_auc":float(roc_auc_score(y,s)),
            "brier_score":float(brier_score_loss(y,s)),"alerts":int(p.sum()),
            "alerts_per_1000":float(p.mean()*1000),"flagged_fraud_amount":flagged,
            "missed_fraud_amount":total_fraud-flagged,"fraud_value_recall":flagged/total_fraud}
    for k,v in actual.items():
        if not np.isclose(float(row[k]),float(v),rtol=1e-6,atol=1e-5):
            raise ValueError(f"Saved {k}={row[k]} contradicts recalculated {v}")
    if int(row.test_rows)!=len(predictions) or int(row.test_fraud)!=int(y.sum()):
        raise ValueError("Saved test population differs from predictions")
    winner=str(selection.loc[selection.cv_oof_ap.idxmax(),"key"])
    if str(row.selected_model)!=winner or str(meta["selected_model"])!=winner:
        raise ValueError("Selected model must maximize training-only OOF AP")
    chosen=selection[selection.key==winner]
    if len(chosen)!=1 or not np.isclose(float(chosen.iloc[0].threshold),thr,rtol=0,atol=1e-8):
        raise ValueError("Selected training threshold differs from holdout threshold")
    if str(row.split)!=str(meta.get("split")):
        raise ValueError("Split name differs from run metadata")
    if str(row.split)=="chronological" and not predictions.Time.is_monotonic_increasing:
        raise ValueError("Chronological holdout must be time-ordered")
    if not set(folds.key).issubset(set(selection.key)):
        raise ValueError("Unknown model in CV folds")
    if folds.groupby("key").size().nunique()!=1:
        raise ValueError("Uneven fold counts among candidates")
    # Verify amount and rate reconcile, with no claims of bank loss reduction.
    actual.update({"selected_model":winner,"threshold":thr,"test_rows":len(y),
                   "test_frauds":int(y.sum()),"fraud_total_amount":total_fraud,
                   "false_positive_legitimate_value":float(a[(y==0)&(p==1)].sum()),
                   "holdout_prevalence":float(y.mean()),"run_id":str(meta.get("run_id","unknown")),
                   "split":str(meta["split"]),"cv_oof_ap":float(chosen.iloc[0].cv_oof_ap)})
    return actual


def save_fig(fig, out: Path, name: str, formats: Iterable[str]):
    for fmt in formats:
        fig.savefig(out/f"{name}.{fmt}",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def confusion_figure(audit, out, formats):
    """Compact row-normalized matrix; both annotations remain inside their cells."""
    m = np.array([[audit["tn"], audit["fp"]], [audit["fn"], audit["tp"]]])
    fig, ax = plt.subplots(figsize=(8.6, 6.5))
    ax.imshow(np.log1p(m), cmap="Blues", aspect="equal")
    for i in range(2):
        denominator = max(int(m[i].sum()), 1)
        for j in range(2):
            color = "white" if m[i, j] == m.max() else NAVY
            ax.text(j, i - .055, f"{m[i, j]:,}", ha="center", va="center",
                    fontsize=19, fontweight="bold", color=color)
            ax.text(j, i + .085, f"{m[i, j] / denominator:.2%}", ha="center",
                    va="center", fontsize=11, color=color)
    ax.set_xticks([0, 1], ["Predicted legitimate", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual legitimate", "Actual fraud"])
    ax.tick_params(axis="both", length=0, labelsize=11, pad=10)
    ax.set_title("Locked Holdout Decision — Confusion Matrix", pad=20, fontsize=15)
    ax.set_xlabel(f"Locked threshold: {audit['threshold']:.5f} (training OOF only)", labelpad=14)
    fig.text(.5, .035, "Percentages are normalized within each actual class.",
             color=GREY, fontsize=10, ha="center")
    ax.grid(False)
    fig.subplots_adjust(left=.18, right=.96, top=.88, bottom=.20)
    save_fig(fig, out, "01_confusion_matrix", formats)


def ranking_curves(df,audit,out,formats):
    y=df.Class.to_numpy(); s=df.score.to_numpy()
    precision,recall,_=precision_recall_curve(y,s)
    fpr,tpr,_=roc_curve(y,s)
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.2))
    axes[0].step(recall,precision,where="post",color=TEAL,lw=2,label=f"AP = {audit['pr_auc']:.3f}")
    axes[0].axhline(y.mean(),ls="--",color=GREY,label=f"Prevalence = {y.mean():.4%}")
    axes[0].set(xlabel="Recall",ylabel="Precision",title="Precision–Recall Ranking (All Thresholds)",xlim=(0,1.01),ylim=(0,1.02))
    axes[0].legend()
    axes[1].plot(fpr,tpr,color=TEAL,lw=2,label=f"ROC-AUC = {audit['roc_auc']:.3f}")
    axes[1].plot([0,1],[0,1],ls="--",color=GREY,label="Random ranking")
    axes[1].set(xlabel="False-positive rate",ylabel="True-positive rate",title="ROC Ranking (All Thresholds)",xlim=(0,1),ylim=(0,1.02))
    axes[1].legend();fig.suptitle("Holdout score-ranking diagnostics — threshold not retuned",fontsize=15,fontweight="bold")
    fig.tight_layout();save_fig(fig,out,"02_ranking_curves",formats)


def calibration_descriptive(df, out, formats):
    """Display score reliability and prevalence without hiding rare-event extremes.

    Fixed, prespecified score bins avoid collapsing all high scores into the
    final equal-mass decile. Empty bins are omitted. This is descriptive only;
    no threshold or probability mapping is learned on the holdout.
    """
    scores = df["score"].to_numpy(dtype=float)
    actual = df["Class"].to_numpy(dtype=int)
    edges = np.array([0, 1e-5, 1e-4, 1e-3, .01, .05, .10, .20, .40, .60, .80, 1.0])
    idx = np.searchsorted(edges, scores, side="right") - 1
    idx = np.clip(idx, 0, len(edges)-2)
    bins = (pd.DataFrame({"bin": idx, "score": scores, "fraud": actual})
              .groupby("bin").agg(mean_score=("score", "mean"),
                                  observed_rate=("fraud", "mean"),
                                  fraud_cases=("fraud", "sum"), n=("fraud", "size")))
    # Wilson intervals are displayed to avoid treating sparse high-score bins
    # as estimates of equivalent precision to the very large low-score bins.
    from scipy.stats import norm
    z = norm.ppf(.975)
    n = bins["n"].to_numpy(dtype=float)
    p = bins["observed_rate"].to_numpy(dtype=float)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    lo, hi = np.maximum(0, center-half), np.minimum(1, center+half)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), gridspec_kw={"width_ratios": [1.10, 1]})
    ax = axes[0]
    # Symmetric-log axes retain zero yet separate 0.001, 0.01, 0.1 and 1.
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.6, label="Perfect calibration")
    ax.errorbar(bins.mean_score, bins.observed_rate,
                yerr=np.vstack((np.maximum(0, p-lo), np.maximum(0, hi-p))), fmt="none", ecolor=TEAL,
                elinewidth=1.2, capsize=3, alpha=.75, zorder=2)
    sizes = np.minimum(145, 24 + 1.2*np.sqrt(bins.n.to_numpy()))
    ax.scatter(bins.mean_score, bins.observed_rate, s=sizes, color=TEAL,
               edgecolor="white", lw=1.1, alpha=.86, zorder=3,
               label="Observed rate (marker size ∝ bin count)")
    ax.set_xscale("symlog", linthresh=.0001)
    ax.set_yscale("symlog", linthresh=.0001)
    ax.set_xlim(-.000015, 1.15)
    ax.set_ylim(-.000015, 1.15)
    ax.set_xticks([0, .001, .01, .1, 1], labels=["0", "0.001", "0.01", "0.1", "1"])
    ax.set_yticks([0, .001, .01, .1, 1], labels=["0", "0.001", "0.01", "0.1", "1"])
    ax.set(xlabel="Mean model score in fixed bin", ylabel="Observed fraud rate",
           title="Reliability by Fixed Score Band")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=GREY, ls="--", lw=1.6, label="Perfect calibration"),
                       Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                              markerfacecolor=TEAL, markeredgecolor="white",
                              label="Observed rate; size reflects bin count")],
              loc="upper left", fontsize=8, framealpha=.95)

    ax = axes[1]
    # Scores close to zero dominate legitimate examples. Logarithmic x-spacing
    # and y-counts reveal the rare high-score tail without omitting zero scores.
    hist_edges = np.r_[0, np.geomspace(1e-6, 1, 31)]
    for klass, label, color in [(0, "Legitimate", NAVY), (1, "Fraud", RED)]:
        group = scores[actual == klass]
        counts, _ = np.histogram(group, bins=hist_edges)
        ax.stairs(counts, hist_edges, color=color, lw=2, label=f"{label} (n={len(group):,})")
    ax.set_xscale("symlog", linthresh=.0001)
    ax.set_yscale("log")
    ax.set_xticks([0, .001, .01, .1, 1], labels=["0", "0.001", "0.01", "0.1", "1"])
    ax.set(xlabel="Model score (log-spaced bins)", ylabel="Transactions per bin (log scale)",
           title="Score Distribution by Actual Class")
    ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Descriptive Holdout Score Reliability — No Recalibration", fontsize=15, fontweight="bold")
    fig.text(.5, .018,
             "Fixed score bins; Wilson 95% intervals. Symlog axes expand rare low-score ranges. "
             "Do not tune calibration or thresholds on this holdout.",
             fontsize=9, color=GREY, ha="center")
    fig.tight_layout(rect=(0, .055, 1, .93), w_pad=2.8)
    save_fig(fig, out, "03_descriptive_reliability", formats)


def errors_figure(df, out, formats):
    """Directly show every flagged/missed case and the extreme-value outliers."""
    labels, colors = ["TP", "FN", "FP"], [TEAL, RED, GOLD]
    sub = df[df.error_type.isin(labels)]
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2))
    rng = np.random.default_rng(2026)  # visual jitter only; never alters data
    for k, (label, color) in enumerate(zip(labels, colors)):
        d = sub.loc[sub.error_type.eq(label)]
        if len(d) == 0:
            continue
        yy = np.full(len(d), k, dtype=float) + rng.uniform(-.14, .14, len(d))
        amount = d.Amount.to_numpy(dtype=float)
        x_amount = np.log1p(amount)
        axes[0].scatter(x_amount, yy, color=color, s=25, alpha=.58,
                        edgecolor="none", rasterized=True)
        q25, median, q75 = np.quantile(x_amount, [.25, .5, .75])
        axes[0].plot([q25, q75], [k, k], lw=7, color=color, solid_capstyle="round", zorder=4)
        axes[0].scatter([median], [k], marker="|", s=270, color="white", linewidths=2, zorder=5)
        scores = d.score.to_numpy(dtype=float)
        axes[1].scatter(scores, yy, color=color, s=25, alpha=.58,
                        edgecolor="none", rasterized=True)
        q25, median, q75 = np.quantile(scores, [.25, .5, .75])
        axes[1].plot([q25, q75], [k, k], lw=7, color=color, solid_capstyle="round", zorder=4)
        axes[1].scatter([median], [k], marker="|", s=270, color="white", linewidths=2, zorder=5)

    axes[0].set(xlabel="log(1 + transaction amount)",
                title="Transaction Amounts by Outcome")
    axes[1].set(xlabel="Predicted model score",
                title="Scores by Outcome", xlim=(-.03, 1.03))
    for ax in axes:
        ax.set_yticks(range(3), [f"{label} (n={sum(sub.error_type.eq(label))})" for label in labels])
        ax.set_ylim(2.6, -.6)
        ax.grid(axis="x", alpha=.19)
        ax.grid(axis="y", visible=False)
    fn = sub[sub.error_type.eq("FN")]
    fp = sub[sub.error_type.eq("FP")]
    facts = []
    if len(fn):
        facts.append(f"Largest missed fraud amount: {fn.Amount.max():,.2f}")
    if len(fp):
        facts.append(f"Largest legitimate false alert: {fp.Amount.max():,.2f}")
    fig.suptitle("Holdout Error Characteristics — Descriptive Only", fontsize=15, fontweight="bold")
    fig.text(.5, .055, "Each dot is a transaction; thick line = interquartile range; white tick = median.",
             ha="center", fontsize=9, color=GREY)
    fig.text(.5, .022, "   |   ".join(facts) + ". Amounts are recorded values, not realized losses.",
             ha="center", fontsize=9, color=NAVY)
    fig.tight_layout(rect=(0, .095, 1, .93), w_pad=3)
    save_fig(fig, out, "04_error_characteristics", formats)


def value_fig(df,audit,out,formats):
    fraud=df[df.Class.eq(1)];labels=["Flagged fraudulent value","Missed fraudulent value"]
    vals=[audit['flagged_fraud_amount'],audit['missed_fraud_amount']]
    fig,ax=plt.subplots(figsize=(8.5,5.1))
    bars=ax.bar(labels,vals,color=[TEAL,RED],width=.5)
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(vals)*.015,f"{v:,.2f}",ha="center",fontweight="bold")
    ax.set_ylim(0,max(vals)*1.17);ax.set_ylabel("Recorded transaction amount (original units)")
    ax.set_title(f"Fraud-value recall = {audit['fraud_value_recall']:.2%} | Case recall = {audit['recall']:.2%}")
    fig.text(.08,.02,"Amounts are not realized losses or proven savings. Intervention outcomes are unknown.",color=GREY,fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));save_fig(fig,out,"05_fraud_value",formats)


def cv_figure(selection,folds,audit,out,formats):
    d=selection.sort_values("cv_oof_ap",ascending=True)
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    c=[TEAL if k==audit['selected_model'] else NAVY for k in d.key]
    axes[0].barh(d.key,d.cv_oof_ap,color=c)
    for i,x in enumerate(d.cv_oof_ap):axes[0].text(x+.004,i,f"{x:.3f}",va="center",fontsize=9)
    axes[0].set(xlim=(0,1.05),xlabel="Pooled training OOF average precision",title="Selection uses training only")
    for key,g in folds.groupby('key'):
        color=TEAL if key==audit['selected_model'] else GREY
        axes[1].plot(g.fold,g.fold_ap,marker='o',alpha=.85 if key==audit['selected_model'] else .3,
                     lw=2 if key==audit['selected_model'] else 1,color=color)
    axes[1].set(xlabel="Validation fold",ylabel="Fold average precision",title="Temporal / fold variability")
    axes[1].set_xticks(sorted(folds.fold.unique()));fig.suptitle("Training Cross-Validation Evidence",fontsize=15,fontweight="bold")
    fig.tight_layout();save_fig(fig,out,"06_training_cv_evidence",formats)


def temporal_figure(df,out,formats):
    d=df.sort_values(['Time','original_row']).copy()
    # Equal-count chronological bins; must not mislabel as equal-duration windows.
    d['segment']=pd.qcut(np.arange(len(d)),q=8,labels=False)
    g=d.groupby('segment').agg(n=('Class','size'),frauds=('Class','sum'),time_min=('Time','min'),time_max=('Time','max'))
    g['rate']=g.frauds/g.n*100
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(g.index.astype(str),g.rate,color=TEAL,width=.65)
    for i,r in g.iterrows():ax.text(i,r.rate+.003,f"{int(r.frauds)}/{int(r.n)}",ha='center',fontsize=9)
    ax.set_ylim(0,max(g.rate)*1.25)
    ax.set(xlabel="Successive equal-count holdout segments (not equal hours)",ylabel="Fraud rate (%)",title="Holdout Prevalence by Transaction Order")
    fig.text(.08,.015,"Descriptive comparison only: short observation period; don't infer periodicity.",color=GREY,fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save_fig(fig,out,"07_holdout_temporal_segments",formats)
    return g.reset_index()


def reports(df,audit,selection,folds,meta,output:Path, temporal):
    output.mkdir(parents=True,exist_ok=True)
    fn=df[df.error_type.eq('FN')].sort_values('Amount',ascending=False)
    fp=df[df.error_type.eq('FP')].sort_values('Amount',ascending=False)
    cols=['original_row','Time','Amount','Class','score','predicted_fraud','error_type']
    fn[cols].head(30).to_csv(output/'top_missed_fraud.csv',index=False)
    fp[cols].head(30).to_csv(output/'false_positive_review.csv',index=False)
    temporal.to_csv(output/'holdout_temporal_segments.csv',index=False)
    selection.sort_values('cv_oof_ap',ascending=False).to_csv(output/'cv_ranked.csv',index=False)
    with (output/'evaluation_audit.json').open('w',encoding='utf-8') as f:
        json.dump({'audit_passed':True, 'run_id':audit['run_id'],
                   'selected_by':'training-only pooled OOF AP',
                   'metrics_recomputed_from_holdout':audit,
                   'holdout_not_used_for_threshold_or_model_selection':True,
                   'bootstrap_ci_source':'existing holdout_metrics.csv; not recomputed',
                   'limitations':meta.get('limitations',[])},f,indent=2)
    lines=["# Model Evaluation — Locked Holdout","",
           f"**Run:** `{audit['split']}` / `{audit['run_id']}`. **Selected:** `{audit['selected_model']}`.","",
           "## Verified results","",
           f"- Training-only pooled OOF AP: **{audit['cv_oof_ap']:.4f}**; locked threshold: **{audit['threshold']:.6f}**.",
           f"- Holdout AP: **{audit['pr_auc']:.4f}**; ROC-AUC: **{audit['roc_auc']:.4f}**.",
           f"- Precision: **{audit['precision']:.2%}**; recall: **{audit['recall']:.2%}**; F1: **{audit['f1']:.4f}**.",
           f"- TP {audit['tp']}, FP {audit['fp']}, FN {audit['fn']}, TN {audit['tn']}.",
           f"- Alerts: **{audit['alerts']:,}** ({audit['alerts_per_1000']:.3f} per 1,000 holdout transactions).",
           f"- Recorded fraud value flagged: **{audit['flagged_fraud_amount']:,.2f}**; missed: **{audit['missed_fraud_amount']:,.2f}** (original source units).",
           f"- Legitimate transaction value flagged: **{audit['false_positive_legitimate_value']:,.2f}** (potential friction exposure, not financial loss).",
           "","## Interpretation safeguards","",
           "- All holdout plots are **descriptive diagnostics**. Do not optimize model, thresholds, calibration, or alert policy using these labels.",
           "- PR-AUC here is **average precision**, not trapezoidal PR-curve area.",
           "- Score reliability is diagnostic; model probabilities have **not** been calibrated on independent training-only data.",
           "- Amount flagged is not confirmed prevented loss. Operational effectiveness and customer friction are unknown.",
           "- Differences between random and chronological holdouts are not paired causal comparisons.",
           "- Existing bootstrap intervals omit retraining and temporal-shift uncertainty.",
           "- Feature contributions cannot be inferred from this prediction-only artifact."]
    (output/'evaluation_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='Saved run-specific reports/tables/ml/<run-id> directory')
    parser.add_argument('--figures-dir',type=Path,default=None)
    parser.add_argument('--tables-dir',type=Path,default=None)
    parser.add_argument('--format',choices=['png','svg','pdf'],nargs='+',default=['png'])
    args=parser.parse_args(argv)
    preds,metrics,selection,folds,meta=read_artifacts(args.run_dir)
    audit=audit_run(preds,metrics,selection,folds,meta)
    fdir=args.figures_dir or Path('reports/figures/ml')/args.run_dir.name
    tdir=args.tables_dir or Path('reports/tables/model_evaluation')/args.run_dir.name
    fdir.mkdir(parents=True,exist_ok=True)
    confusion_figure(audit,fdir,args.format)
    ranking_curves(preds,audit,fdir,args.format)
    calibration_descriptive(preds,fdir,args.format)
    errors_figure(preds,fdir,args.format)
    value_fig(preds,audit,fdir,args.format)
    cv_figure(selection,folds,audit,fdir,args.format)
    temporal=temporal_figure(preds,fdir,args.format)
    reports(preds,audit,selection,folds,meta,tdir,temporal)
    print(f"[evaluation] PASS — {audit['selected_model']} / {audit['split']} / {audit['test_rows']:,} holdout rows")
    print(f"[evaluation] TP={audit['tp']} FP={audit['fp']} FN={audit['fn']} TN={audit['tn']} | AP={audit['pr_auc']:.4f} | F1={audit['f1']:.4f}")
    print(f"[evaluation] 7 figures ({', '.join(args.format)}) -> {fdir}")
    print(f"[evaluation] 6 audit/report files -> {tdir}")

if __name__=='__main__':main()
