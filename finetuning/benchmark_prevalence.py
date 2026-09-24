#!/usr/bin/env python3
"""
Prevalence-matched benchmarking (no GPU, no retraining).

BioSerenity-E1 (and most TUH-Seizure papers) report on a *class-balanced* test set
(~29% seizure), built by keeping seizure-containing records and undersampling the
background class. Our THUSZ eval is the natural, highly imbalanced distribution
(~6.7% seizure), which makes AUPRC look far lower even for the same model.

This script re-scores our SAVED probabilities (scores.npz) after subsampling the
non-seizure eval windows down to a target seizure prevalence, so the numbers are
directly comparable to the benchmark. AUROC is ~prevalence-invariant; AUPRC and
threshold metrics shift. We repeat the subsampling over many seeds -> mean +/- std.

Reuses probs only; no model, no GPU.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             f1_score, confusion_matrix)

ROOT = Path("/home/abdulh/scratch/EEGdiff_V2")
OUT = ROOT / "Benchmarking"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "frozen":   ROOT / "TUHSZ_Res/run_20260702_042547/scores.npz",
    "unfrozen": ROOT / "Binary_finetune_unfrozen/run_20260702_071240/scores.npz",
}

TARGET_PREVALENCES = [0.0672, 0.29, 0.49]  # natural, BioSerenity test, ~balanced
N_SEEDS = 200
# BioSerenity-E1 reference (TUH-Seizure, ~29% test prevalence)
REF = {"model": "BioSerenity-E1 (TUH-Seizure)", "auroc": 0.926, "auprc": 0.83, "prevalence": 0.29}


def threshold_metrics(labels, probs, thr):
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-9)          # recall / sensitivity
    spec = tn / (tn + fp + 1e-9)
    bacc = 0.5 * (sens + spec)
    f1_seiz = f1_score(labels, preds, pos_label=1, zero_division=0)
    wf1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return dict(sensitivity=sens, specificity=spec, balanced_acc=bacc,
                seizure_f1=f1_seiz, weighted_f1=wf1)


def best_bacc_threshold(labels, probs):
    best_t, best = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        m = threshold_metrics(labels, probs, t)
        if m["balanced_acc"] > best:
            best, best_t = m["balanced_acc"], t
    return best_t


def auprc(labels, probs):
    pc, rc, _ = precision_recall_curve(labels, probs)
    return auc(rc, pc)


def subsample(probs, labels, target_prev, rng):
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_pos = len(pos)
    n_neg = int(round(n_pos * (1 - target_prev) / target_prev))
    if n_neg <= len(neg):
        neg_sel = rng.choice(neg, size=n_neg, replace=False)
    else:                                   # need more neg than available: keep all
        neg_sel = neg
    idx = np.concatenate([pos, neg_sel])
    return probs[idx], labels[idx]


def evaluate_run(name, probs, labels):
    natural_prev = float((labels == 1).mean())
    out = {"run": name, "n_total": int(len(labels)),
           "n_seizure": int((labels == 1).sum()),
           "n_nonseizure": int((labels == 0).sum()),
           "natural_prevalence": natural_prev, "by_prevalence": {}}

    for tp in TARGET_PREVALENCES:
        is_natural = abs(tp - natural_prev) < 1e-3
        if is_natural:
            # deterministic on the full set
            roc = roc_auc_score(labels, probs)
            pr = auprc(labels, probs)
            t_star = best_bacc_threshold(labels, probs)
            m05 = threshold_metrics(labels, probs, 0.5)
            mbest = threshold_metrics(labels, probs, t_star)
            rec = {"target_prevalence": tp, "n_windows": int(len(labels)),
                   "n_neg_kept": int((labels == 0).sum()), "seeds": 1,
                   "auroc_mean": float(roc), "auroc_std": 0.0,
                   "auprc_mean": float(pr), "auprc_std": 0.0,
                   "auprc_baseline": tp,
                   "thr0.5": {k: float(v) for k, v in m05.items()},
                   "thr_best_bacc": {"threshold": float(t_star),
                                     **{k: float(v) for k, v in mbest.items()}}}
        else:
            rocs, prs = [], []
            m05_acc, mbest_acc, tstars = [], [], []
            n_kept = None
            for s in range(N_SEEDS):
                rng = np.random.default_rng(1000 + s)
                p, l = subsample(probs, labels, tp, rng)
                n_kept = int((l == 0).sum())
                rocs.append(roc_auc_score(l, p))
                prs.append(auprc(l, p))
                m05_acc.append(threshold_metrics(l, p, 0.5))
                t_star = best_bacc_threshold(l, p)
                tstars.append(t_star)
                mbest_acc.append(threshold_metrics(l, p, t_star))

            def agg(dlist, key):
                v = np.array([d[key] for d in dlist])
                return float(v.mean()), float(v.std())

            rec = {"target_prevalence": tp,
                   "n_windows": int(len(np.where(labels == 1)[0]) + n_kept),
                   "n_neg_kept": n_kept, "seeds": N_SEEDS,
                   "auroc_mean": float(np.mean(rocs)), "auroc_std": float(np.std(rocs)),
                   "auprc_mean": float(np.mean(prs)), "auprc_std": float(np.std(prs)),
                   "auprc_baseline": tp,
                   "thr0.5": {k: {"mean": agg(m05_acc, k)[0], "std": agg(m05_acc, k)[1]}
                              for k in m05_acc[0]},
                   "thr_best_bacc": {"threshold_mean": float(np.mean(tstars)),
                                     **{k: {"mean": agg(mbest_acc, k)[0], "std": agg(mbest_acc, k)[1]}
                                        for k in mbest_acc[0]}}}
        out["by_prevalence"][f"{tp:.4f}"] = rec
    return out


def fmt_line(rec):
    tp = rec["target_prevalence"]
    return (f"  prev={tp:5.2%} | n={rec['n_windows']:>7,} | "
            f"AUROC {rec['auroc_mean']:.4f}+/-{rec['auroc_std']:.4f} | "
            f"AUPRC {rec['auprc_mean']:.4f}+/-{rec['auprc_std']:.4f} "
            f"(baseline {rec['auprc_baseline']:.3f})")


results = {"reference": REF, "n_seeds": N_SEEDS, "runs": {}}
lines = ["PREVALENCE-MATCHED BENCHMARKING (THUSZ seizure, frozen probes reused)",
         f"Reference: {REF['model']} -> AUROC {REF['auroc']}, AUPRC {REF['auprc']} "
         f"(at {REF['prevalence']:.0%} prevalence)\n"]

for name, path in RUNS.items():
    d = np.load(path)
    res = evaluate_run(name, d["eval_probs"], d["eval_labels"])
    results["runs"][name] = res
    lines.append(f"[{name.upper()}]  (natural prevalence {res['natural_prevalence']:.2%})")
    for k in sorted(res["by_prevalence"]):
        rec = res["by_prevalence"][k]
        lines.append(fmt_line(rec))
        m = rec["thr0.5"]
        if rec["seeds"] == 1:
            lines.append(f"        @0.5: seizureF1 {m['seizure_f1']:.3f}  wF1 {m['weighted_f1']:.3f}  "
                         f"sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}  bAcc {m['balanced_acc']:.3f}")
        else:
            lines.append(f"        @0.5: seizureF1 {m['seizure_f1']['mean']:.3f}  wF1 {m['weighted_f1']['mean']:.3f}  "
                         f"sens {m['sensitivity']['mean']:.3f}  spec {m['specificity']['mean']:.3f}  "
                         f"bAcc {m['balanced_acc']['mean']:.3f}")
    lines.append("")

(OUT / "prevalence_benchmark.json").write_text(json.dumps(results, indent=2))
(OUT / "prevalence_benchmark.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved -> {OUT}/prevalence_benchmark.{{json,txt}}")
