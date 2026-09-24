#!/usr/bin/env python3
"""Supervised baselines trained from scratch on our splits.

EEGNet, EEG-Conformer and ST-Transformer are not foundation models: none of them
ships pretrained weights, so there is no frozen representation to probe. They are
therefore trained end-to-end on the task, which is how their authors use them.

That makes this a DIFFERENT protocol from probe.py, and the two must not be merged
into one ranking. It is worth being explicit about the direction of the asymmetry:
these models get full end-to-end training on the downstream task, whereas the
foundation models in probe.py are frozen and receive only a linear read-out. If a
from-scratch supervised model still scores lower, that is a meaningful result rather
than a handicapped comparison.

All three receive identical input -- the same 22 monopolar channels at 256 Hz with
channel-wise standardisation that DiffEEG consumes -- the same splits, the same
optimiser and schedule, and the same metrics and bootstrap as every other experiment
in this benchmark. Only the architecture differs.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

B = Path("/home/abdulh/scratch/benchmarking2")
sys.path.insert(0, str(B / "code")); sys.path.insert(0, str(B / "BIOT"))
import data_adapter as DA
from extract_features import get_splits
from finetune_baseline import ViewDataset, metrics          # reuse loader + metric set

N_BOOT, SEED = 1000, 0
EPOCHS, BATCH, PATIENCE, WARMUP = 50, 128, 8, 3
LR, WD = 1e-3, 1e-4
N_CH, N_TIMES, SFREQ = 22, 1280, 256


def build(name):
    if name == "eegnet":
        from braindecode.models import EEGNet
        return EEGNet(n_chans=N_CH, n_outputs=2, n_times=N_TIMES, sfreq=SFREQ)
    if name == "conformer":
        from braindecode.models import EEGConformer
        return EEGConformer(n_chans=N_CH, n_outputs=2, n_times=N_TIMES, sfreq=SFREQ)
    if name == "sttransformer":
        from model.st_transformer import STTransformer
        return STTransformer(emb_size=256, depth=4, n_classes=2,
                             channel_legnth=N_TIMES, n_channels=N_CH)
    raise ValueError(name)


class RawView(ViewDataset):
    """Same lazy loader, but the DiffEEG view (22 monopolar, z-scored) for every model."""
    def __init__(self, triples, ds):
        self.t, self.view, self.cfg = list(triples), "diffeeg", DA.DATASETS[ds]
        norm = self.cfg["norm"]
        self.mu = np.load(norm / "mean.npy").reshape(-1, 1).astype(np.float32)[:22]
        self.sd = np.load(norm / "std.npy").reshape(-1, 1).astype(np.float32)[:22]

    def __getitem__(self, i):
        f, r, lab = self.t[i]
        s = np.asarray(np.load(f, mmap_mode="r")[r], dtype=np.float32)
        if s.ndim == 2 and s.shape[0] != 22 and s.shape[1] == 22:
            s = s.T
        s = (s - self.mu) / (self.sd + 1e-6)
        return torch.from_numpy(s).float(), torch.tensor(int(lab))


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); P, Y = [], []
    for x, y in dl:
        p = torch.softmax(model(x.to(dev, non_blocking=True)), 1)[:, 1]
        P.append(p.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["eegnet", "conformer", "sttransformer"])
    ap.add_argument("--dataset", default="tusz", choices=["tusz", "tuab5s", "bonn", "siena"])
    a = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = B / "results" / f"supervised_{a.model}_{a.dataset}"; out.mkdir(parents=True, exist_ok=True)
    ckpt_p, best_p = out / "ckpt.pth", out / "best.pth"

    sp = get_splits(a.dataset, DA)
    print({k: len(v) for k, v in sp.items()}, flush=True)
    dls = {k: DataLoader(RawView(sp[k], a.dataset), batch_size=BATCH,
                         shuffle=(k == "train"), num_workers=6, pin_memory=True,
                         persistent_workers=True, drop_last=(k == "train"))
           for k in ("train", "val", "test")}

    model = build(a.model).to(dev)
    print(f"  {a.model}: {sum(p.numel() for p in model.parameters()):,} params "
          f"(trained from scratch, no pretrained weights)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ytr = np.array([l for _, _, l in sp["train"]])
    cw = torch.tensor(len(ytr) / (2 * np.maximum(np.bincount(ytr, minlength=2), 1)),
                      dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw)
    steps = len(dls["train"]); base_lr = LR

    def set_lr(ep, it):
        t = ep + it / steps
        f = t / max(WARMUP, 1e-8) if t < WARMUP else \
            0.5 * (1 + math.cos(math.pi * (t - WARMUP) / max(EPOCHS - WARMUP, 1e-8)))
        for g in opt.param_groups: g["lr"] = base_lr * f

    start, best, bad, hist = 0, -1.0, 0, []
    if ckpt_p.exists():
        c = torch.load(ckpt_p, map_location=dev)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        start, best, bad, hist = c["epoch"] + 1, c["best"], c["bad"], c.get("hist", [])
        print(f"resumed at epoch {start} (best val AUROC {best:.4f})", flush=True); del c

    for ep in range(start, EPOCHS):
        model.train(); t0, losses = time.time(), []
        for it, (x, y) in enumerate(dls["train"]):
            set_lr(ep, it)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep}")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); continue
            opt.step(); losses.append(loss.item())
        pv, yv = predict(model, dls["val"], dev)
        vauc = roc_auc_score(yv, pv)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_auc": float(vauc)})
        print(f"  ep {ep:02d} loss {np.mean(losses):.4f}  val AUROC {vauc:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vauc > best:
            best, bad = vauc, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": float(vauc)}, best_p)
            print(f"  >> new best (epoch {ep})", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at epoch {ep}", flush=True); break
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "best": best, "bad": bad, "hist": hist}, ckpt_p)

    bs = torch.load(best_p, map_location=dev); model.load_state_dict(bs["model"])
    prob, yte = predict(model, dls["test"], dev)
    pred = (prob >= 0.5).astype(int)
    point = metrics(yte, pred, prob)
    rng = np.random.default_rng(0); boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[i])) < 2: continue
        boot.append(metrics(yte[i], pred[i], prob[i]))
    ci = {k: (float(np.percentile([b[k] for b in boot], 2.5)),
              float(np.percentile([b[k] for b in boot], 97.5))) for k in point}
    res = {"model": a.model, "dataset": a.dataset,
           "protocol": "supervised, trained from scratch (no pretrained weights)",
           "params": int(sum(p.numel() for p in model.parameters())),
           "best_epoch": int(bs["epoch"]), "val_auc": float(bs["val_auc"]),
           "n": {k: len(v) for k, v in sp.items()},
           "test_prevalence": float(yte.mean()), "metrics": point, "ci95": ci,
           "confusion": confusion_matrix(yte, pred, labels=[0, 1]).tolist(), "history": hist}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    ks = ["accuracy", "roc_auc", "pr_auc", "f1_weighted", "f1_macro", "sensitivity", "specificity"]
    print("\n" + "".join(f"{k:>13}" for k in ks))
    print("".join(f"{point[k]:>13.4f}" for k in ks))
    print(f"\nsaved -> {out/'result.json'}")


if __name__ == "__main__":
    main()
