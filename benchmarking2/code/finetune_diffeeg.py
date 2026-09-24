#!/usr/bin/env python3
"""DiffEEG fine-tuned with the paper's own recipe, on the benchmark's exact subset.

The linear probe showed DiffEEG's frozen 2,400-d descriptor trailing the baselines. That
descriptor is bit-for-bit the representation fewshot_tuab_rl.py's EEGClassifier consumes
(model_channels * sum(channel_multipliers) * 5 = 32 * 15 * 5 = 2400), so the probe was
not mis-extracting anything. What the probe omits is everything the paper's pipeline adds
on top: a non-linear head, partial unfreezing of the backbone, and the reinforced
decision layer. This script restores all three and evaluates on the identical split, so
the difference between this result and the probe isolates the adaptation recipe.

Deliberate deviation from fewshot_tuab_rl.py: that script selects its checkpoint on
TRAIN metrics, which cannot detect overfitting. We have a held-out validation split from
the benchmark, so selection is done on validation AUROC -- the same criterion the probe
and the MLP head use. Reporting a number selected on training performance alongside
baselines selected on validation would not be a fair comparison.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

B = Path("/home/abdulh/scratch/benchmarking2")
sys.path.insert(0, str(B / "code"))
sys.path.insert(0, "/home/abdulh/scratch")

import data_adapter as DA
from extract_features import get_splits
from fewshot_tuab_rl import (ARCH, DIFFUSION_CHECKPOINT, EEGClassifier, MemmapEEGDataset,
                             ReinforcedDecisionLayer, batch_f1, collate_fn, set_seed,
                             setup_backbone, unwrap_module)
from Diff_EEG_train import DeepEnhancedEEGDiffusionModel

EPOCHS, BATCH, PATIENCE = 60, 128, 10
HEAD_LR, BACKBONE_LR, WD = 5e-4, 1e-5, 1e-4
RL_WEIGHT, SEED, WORKERS = 0.02, 42, 4
N_BOOT = 1000
CKPT_EVERY = 2


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


def loader(triples, mean, std, shuffle):
    return DataLoader(MemmapEEGDataset(triples, mean, std), batch_size=BATCH,
                      shuffle=shuffle, num_workers=WORKERS, pin_memory=True,
                      collate_fn=collate_fn, persistent_workers=WORKERS > 0)


@torch.no_grad()
def predict(clf, rl, dl, dev):
    clf.eval(); rl.eval()
    P, Y = [], []
    for x, y in dl:
        x = x.to(dev, non_blocking=True)
        logits, feats = clf(x)
        _, probs, _, _ = rl(logits, feats, training=False)
        P.append(probs.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", choices=["on", "off"], default="on")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default="tuab5s",
                    choices=["tuab5s", "tusz", "bonn", "siena"])
    a = ap.parse_args()
    set_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"diffeeg_ft_{a.dataset}_rl-{a.rl}"
    outdir = Path(a.out) if a.out else (B / "results" / tag)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path, best_path = outdir / "ckpt.pth", outdir / "best.pth"

    # identical subset to the probe / MLP-head experiments
    sp = get_splits(a.dataset, DA)
    print({k: len(v) for k, v in sp.items()}, flush=True)
    norm = DA.DATASETS[a.dataset]["norm"]
    mean, std = np.load(norm / "mean.npy"), np.load(norm / "std.npy")
    dl_tr = loader(sp["train"], mean, std, True)
    dl_va = loader(sp["val"], mean, std, False)
    dl_te = loader(sp["test"], mean, std, False)

    bb = DeepEnhancedEEGDiffusionModel(**ARCH).to(dev)
    ck = torch.load(DIFFUSION_CHECKPOINT, map_location=dev)
    bb.load_state_dict(ck["model_state_dict"]); del ck
    clf = EEGClassifier(backbone=bb, num_classes=2, dropout=0.4).to(dev)
    setup_backbone(clf.backbone, "unfrozen")          # time_mlp + last two down blocks
    agg = ARCH["model_channels"] * sum(ARCH["channel_multipliers"]) * 5
    rl = ReinforcedDecisionLayer(input_dim=agg,
                                 rl_weight=(RL_WEIGHT if a.rl == "on" else 0.0)).to(dev)

    bb_p = [p for n, p in clf.named_parameters() if p.requires_grad and "backbone" in n]
    hd_p = [p for n, p in clf.named_parameters() if p.requires_grad and "backbone" not in n]
    opt = optim.AdamW([{"params": bb_p, "lr": BACKBONE_LR, "weight_decay": WD},
                       {"params": hd_p, "lr": HEAD_LR, "weight_decay": WD},
                       {"params": rl.parameters(), "lr": HEAD_LR, "weight_decay": WD}])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    ytr = np.array([l for _, _, l in sp["train"]])
    cw = torch.tensor(1.0 / np.maximum(np.bincount(ytr, minlength=2), 1),
                      dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw / cw.sum() * 2)

    start, best, bad, hist = 1, -1.0, 0, []
    if ckpt_path.exists():
        c = torch.load(ckpt_path, map_location=dev)
        clf.load_state_dict(c["clf"]); rl.load_state_dict(c["rl"])
        opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start, best, bad, hist = c["epoch"] + 1, c["best"], c["bad"], c.get("hist", [])
        print(f"resumed at epoch {start} (best val AUROC {best:.4f})", flush=True)
        del c

    for ep in range(start, EPOCHS + 1):
        clf.train(); rl.train(); clf.backbone.train()
        t0, losses = time.time(), []
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, feats = clf(x)
            adj, probs, act, logp = rl(logits, feats, training=True)
            loss = lossf(adj, y)
            if a.rl == "on":
                reward = batch_f1(act, y)
                base = unwrap_module(rl).update_baseline(reward)
                loss = loss + RL_WEIGHT * (-(reward.detach() - base) * logp.mean())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep}")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in list(clf.parameters()) + list(rl.parameters()) if p.requires_grad], 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); continue    # one bad batch must not poison the step
            opt.step(); losses.append(loss.item())
        sched.step()

        pv, yv = predict(clf, rl, dl_va, dev)
        vauc = roc_auc_score(yv, pv)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_auc": float(vauc)})
        print(f"  ep {ep:02d} loss {np.mean(losses):.4f}  val AUROC {vauc:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vauc > best:
            best, bad = vauc, 0
            torch.save({"clf": clf.state_dict(), "rl": rl.state_dict(),
                        "epoch": ep, "val_auc": float(vauc)}, best_path)
            print(f"  >> new best (epoch {ep})", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at epoch {ep}", flush=True); break
        if ep % CKPT_EVERY == 0:
            torch.save({"clf": clf.state_dict(), "rl": rl.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep, "best": best,
                        "bad": bad, "hist": hist}, ckpt_path)

    bs = torch.load(best_path, map_location=dev)
    clf.load_state_dict(bs["clf"]); rl.load_state_dict(bs["rl"])
    prob, yte = predict(clf, rl, dl_te, dev)
    pred = (prob >= 0.5).astype(int)
    point = metrics(yte, pred, prob)
    rng = np.random.default_rng(0); boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[i])) < 2: continue
        boot.append(metrics(yte[i], pred[i], prob[i]))
    ci = {k: (float(np.percentile([b[k] for b in boot], 2.5)),
              float(np.percentile([b[k] for b in boot], 97.5))) for k in point}

    res = {"model": "diffeeg", "dataset": a.dataset, "protocol": f"fine-tuned (unfrozen + RDL rl={a.rl})",
           "selection": "validation AUROC", "best_epoch": int(bs["epoch"]),
           "val_auc": float(bs["val_auc"]),
           "n": {k: len(v) for k, v in sp.items()},
           "test_prevalence": float(yte.mean()), "metrics": point, "ci95": ci,
           "confusion": confusion_matrix(yte, pred, labels=[0, 1]).tolist(), "history": hist}
    (outdir / "result.json").write_text(json.dumps(res, indent=2))
    ks = ["accuracy", "roc_auc", "pr_auc", "f1_weighted", "f1_macro", "sensitivity", "specificity"]
    print("\n" + "".join(f"{k:>13}" for k in ks))
    print("".join(f"{point[k]:>13.4f}" for k in ks))
    print(f"\nsaved -> {outdir/'result.json'}")


if __name__ == "__main__":
    main()
