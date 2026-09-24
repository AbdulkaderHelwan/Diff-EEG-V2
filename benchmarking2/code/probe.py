#!/usr/bin/env python3
"""Linear probe: DiffEEG vs EEGDM frozen representations on our datasets.

    python probe.py --dataset tuab5s
    python probe.py --dataset tusz

Protocol -- identical for both models so nothing but the representation can decide it:
  * backbone frozen; each model contributes its own pooled descriptor
    (DiffEEG 2,400 dims; EEGDM 2,560 dims -- deliberately matched, see extract_features.py)
  * same samples, same split, same standardisation (train mean/std)
  * same classifier family, same C grid, C chosen on VALIDATION only
  * metrics reported are the ones the DiffEEG paper reports for binary tasks
  * uncertainty from a 1,000-fold bootstrap of the test set (lbfgs is deterministic,
    so re-seeding the solver would give an artificial spread of exactly zero)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

B = Path("/home/abdulh/scratch/benchmarking2")
FEAT = B / "results" / "features"
MODELS = ["diffeeg", "eegdm", "biot", "labram", "cbramod"]
# Widened twice: DiffEEG selected the maximum at both 1.0 and 10.0, so its optimum was
# never bracketed. A boundary selection is reported as a warning below.
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
N_BOOT = 1000


def metrics(y, p, prob):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "accuracy":    float(accuracy_score(y, p)),
        "roc_auc":     float(roc_auc_score(y, prob)),
        "pr_auc":      float(average_precision_score(y, prob)),
        "f1_pos":      float(f1_score(y, p, zero_division=0)),
        "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
        "f1_macro":    float(f1_score(y, p, average="macro", zero_division=0)),
        "precision":   float(precision_score(y, p, zero_division=0)),
        "sensitivity": float(recall_score(y, p, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def run(model, ds):
    def load(sp):
        d = np.load(FEAT / f"{model}_{ds}_{sp}.npz")
        return d["X"], d["y"]
    Xtr, ytr = load("train"); Xva, yva = load("val"); Xte, yte = load("test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    val_auc = {}
    for C in C_GRID:
        clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
        clf.fit(Ztr, ytr)
        val_auc[C] = roc_auc_score(yva, clf.predict_proba(Zva)[:, 1])
    best_C = max(val_auc, key=val_auc.get)
    if best_C in (C_GRID[0], C_GRID[-1]):
        print(f"    [warn] {model}: best C={best_C} is at a grid boundary", flush=True)

    clf = LogisticRegression(max_iter=2000, C=best_C, class_weight="balanced")
    clf.fit(Ztr, ytr)
    prob = clf.predict_proba(Zte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    point = metrics(yte, pred, prob)

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[i])) < 2:
            continue
        boot.append(metrics(yte[i], pred[i], prob[i]))
    ci = {k: (float(np.percentile([b[k] for b in boot], 2.5)),
              float(np.percentile([b[k] for b in boot], 97.5))) for k in point}
    return {"model": model, "dim": int(Xtr.shape[1]), "best_C": best_C,
            "n": {"train": len(ytr), "val": len(yva), "test": len(yte)},
            "test_prevalence": float(yte.mean()),
            "val_auc_by_C": {str(k): float(v) for k, v in val_auc.items()},
            "metrics": point, "ci95": ci,
            "confusion": confusion_matrix(yte, pred, labels=[0, 1]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tuab5s", "tusz", "bonn", "siena"], required=True)
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of MODELS; default = all available")
    a = ap.parse_args()

    out = {"task": f"{a.dataset} linear probe", "C_grid": C_GRID, "n_boot": N_BOOT,
           "results": {}}
    keys = ["accuracy", "roc_auc", "pr_auc", "f1_weighted", "f1_macro", "sensitivity",
            "specificity"]
    print(f"\n=== {a.dataset}: frozen linear probe ===")
    print(f"{'model':<10}{'dim':>7}{'C':>8}  " + "".join(f"{k:>13}" for k in keys))
    print("-" * (25 + 13 * len(keys)))
    wanted = a.models.split(",") if a.models else MODELS
    for m in wanted:
        if not (FEAT / f"{m}_{a.dataset}_test.npz").exists():
            print(f"{m:<10} features missing"); continue
        r = run(m, a.dataset)
        out["results"][m] = r
        print(f"{m:<10}{r['dim']:>7}{r['best_C']:>8}  "
              + "".join(f"{r['metrics'][k]:>13.4f}" for k in keys))

    # pairwise contrasts against DiffEEG, with CI overlap flagged
    if "diffeeg" in out["results"]:
        d = out["results"]["diffeeg"]
        for other in [m for m in wanted if m != "diffeeg" and m in out["results"]]:
            o = out["results"][other]
            print(f"\n  DiffEEG - {other} (test prevalence {d['test_prevalence']:.3f}):")
            for k in keys:
                dl, dh = d["ci95"][k]; ol, oh = o["ci95"][k]
                sep = "disjoint" if (dl > oh or ol > dh) else "OVERLAP"
                print(f"    {k:<13} {d['metrics'][k] - o['metrics'][k]:+.4f}   "
                      f"DiffEEG [{dl:.3f},{dh:.3f}]  {other} [{ol:.3f},{oh:.3f}]  95% CI {sep}")

    tag = "" if not a.models else "_" + a.models.replace(",", "-")
    p = B / "results" / f"probe_{a.dataset}{tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  saved -> {p}")


if __name__ == "__main__":
    main()
