#!/usr/bin/env python3
"""Non-linear (MLP) head on the SAME cached frozen features used by probe.py.

This is NOT fine-tuning: the backbones are not in the loop and their weights are never
updated. It isolates one question the linear probe leaves open -- whether DiffEEG's
representation trails the baselines only because the read-out was restricted to a linear
map, or whether the gap survives a non-linear head of reasonable capacity.

Protocol matches probe.py exactly (same features, same splits, same standardisation,
same metrics, same 1,000-fold test bootstrap) so the two are directly comparable. The
head is identical for every model, and its width does not depend on the input dimension
beyond the first layer, so no model is favoured by capacity.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

B = Path("/home/abdulh/scratch/benchmarking2")
FEAT = B / "results" / "features"
MODELS = ["diffeeg", "eegdm", "biot", "labram", "cbramod"]
HIDDEN, DROPOUT = [512, 256], 0.3
EPOCHS, BATCH, PATIENCE = 60, 512, 8
LRS, WDS = [1e-3, 3e-4], [1e-4, 1e-2]      # small grid, selected on validation AUROC
N_BOOT = 1000


def metrics(y, p, prob):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, p)),
            "roc_auc": float(roc_auc_score(y, prob)),
            "pr_auc": float(average_precision_score(y, prob)),
            "f1_pos": float(f1_score(y, p, zero_division=0)),
            "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
            "precision": float(precision_score(y, p, zero_division=0)),
            "sensitivity": float(recall_score(y, p, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1))}


def head(d_in, dev):
    layers, d = [], d_in
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
        d = h
    layers.append(nn.Linear(d, 2))
    return nn.Sequential(*layers).to(dev)


def fit(Ztr, ytr, Zva, yva, lr, wd, dev, seed):
    torch.manual_seed(seed)
    net = head(Ztr.shape[1], dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    # class_weight="balanced" equivalent, matching probe.py's balanced logistic regression
    cw = torch.tensor([len(ytr) / (2 * (ytr == c).sum()) for c in (0, 1)],
                      dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xtr = torch.from_numpy(Ztr).float(); Ytr = torch.from_numpy(ytr).long()
    Xva = torch.from_numpy(Zva).float().to(dev)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.softmax(net(Xva), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net, best


def run(model, dev, ds):
    def load(sp):
        d = np.load(FEAT / f"{model}_{ds}_{sp}.npz"); return d["X"], d["y"]
    Xtr, ytr = load("train"); Xva, yva = load("val"); Xte, yte = load("test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    best = (-1, None, None)
    for lr in LRS:
        for wd in WDS:
            t0 = time.time()
            net, auc = fit(Ztr, ytr, Zva, yva, lr, wd, dev, seed=0)
            print(f"    lr={lr} wd={wd}: val AUROC {auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if auc > best[0]:
                best = (auc, (lr, wd), net)
    val_auc, (lr, wd), net = best
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(torch.from_numpy(Zte).float().to(dev)), 1)[:, 1].cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    point = metrics(yte, pred, prob)
    rng = np.random.default_rng(0); boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[i])) < 2: continue
        boot.append(metrics(yte[i], pred[i], prob[i]))
    ci = {k: (float(np.percentile([b[k] for b in boot], 2.5)),
              float(np.percentile([b[k] for b in boot], 97.5))) for k in point}
    return {"model": model, "dim": int(Xtr.shape[1]), "hidden": HIDDEN,
            "best_lr": lr, "best_wd": wd, "val_auc": float(val_auc),
            "n": {"train": len(ytr), "val": len(yva), "test": len(yte)},
            "test_prevalence": float(yte.mean()), "metrics": point, "ci95": ci,
            "confusion": confusion_matrix(yte, pred, labels=[0, 1]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--dataset", default="tuab5s",
                    choices=["tuab5s", "tusz", "bonn", "siena"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wanted = a.models.split(",") if a.models else MODELS
    keys = ["accuracy", "roc_auc", "pr_auc", "f1_weighted", "f1_macro", "sensitivity",
            "specificity"]
    out = {"task": f"{a.dataset} MLP head on frozen features", "hidden": HIDDEN,
           "note": "frozen backbones; head-only training, NOT fine-tuning", "results": {}}
    print(f"device: {dev}\n=== {a.dataset}: non-linear (MLP) head on frozen features ===")
    print(f"{'model':<10}{'dim':>7}  " + "".join(f"{k:>13}" for k in keys))
    print("-" * (17 + 13 * len(keys)))
    for m in wanted:
        if not (FEAT / f"{m}_{a.dataset}_test.npz").exists():
            print(f"{m:<10} features missing"); continue
        print(f"  [{m}]", flush=True)
        r = run(m, dev, a.dataset); out["results"][m] = r
        print(f"{m:<10}{r['dim']:>7}  " + "".join(f"{r['metrics'][k]:>13.4f}" for k in keys),
              flush=True)
    p = B / "results" / f"mlp_head_{a.dataset}.json"
    p.write_text(json.dumps(out, indent=2)); print(f"\n  saved -> {p}")


if __name__ == "__main__":
    main()
