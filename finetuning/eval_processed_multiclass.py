#!/usr/bin/env python3
"""Evaluate the frozen EEGdiff V2 backbone on any processed root (binary or
multiclass) via a FRESHLY TRAINED head -- the "trained classifier" experiment.

Unlike eval_processed_binary.py (which zero-shot reuses a THUSZ-trained
classifier head, no adaptation at all), this tests the EMBEDDING itself on a
task the backbone was never trained for: extract frozen features for the
processed root's own official train/test split, fit a small MLP head on train,
report metrics on test. No CV needed -- the dataset supplies its own held-out
split (e.g. BCI IV-2a's T=train / E=test sessions). Works for 2-class (e.g.
Bonn normal/seizure) or N-class (e.g. BCI IV-2a) roots alike; ROC-AUC/PR-AUC
are additionally reported when there are exactly 2 classes.

Expects `<root>/{train,test}-<classname>/*_batch_*.npy`.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Diff_EEG_train_v2 import DeepEnhancedEEGDiffusionModel

DEFAULT_NORM = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/normalization"
DEFAULT_BACKBONE = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/best_EEGDIFF_V2.pth"
DEFAULT_OUT = "/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
PROBE = [50, 250, 500, 750, 950]
ARCH = dict(in_channels=22, model_channels=64, channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=3, time_emb_dim=768, dropout=0.1, attention_heads=8)


def discover_classes(root, split):
    classes = []
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and d.name.startswith(f"{split}-"):
            classes.append(d.name[len(split) + 1:])
    return classes


def load_split_raw(root, split, classes, max_per_class=None, seed=42):
    """Load raw (unnormalized) windows + labels for one split.

    With `max_per_class`, subsamples at the FILE level (randomly ordered batch
    files, stop once the cap is hit) -- never reads more than needed. Essential
    for datasets like sleep_edfx (~330GB full size): loading everything just to
    subsample in-memory would OOM before subsampling ever helps."""
    xs, ys = [], []
    rng = np.random.default_rng(seed)
    for ci, cname in enumerate(classes):
        d = Path(root) / f"{split}-{cname}"
        if not d.is_dir():
            continue
        files = sorted(d.glob("*_batch_*.npy"))
        if max_per_class is not None:
            files = list(rng.permutation(np.array(files, dtype=object)))
        n_loaded = 0
        for f in files:
            arr = np.load(f).astype(np.float32)
            if max_per_class is not None and n_loaded + len(arr) > max_per_class:
                arr = arr[:max_per_class - n_loaded]
            xs.append(arr)
            ys.append(np.full(len(arr), ci, dtype=np.int64))
            n_loaded += len(arr)
            if max_per_class is not None and n_loaded >= max_per_class:
                break
    if not xs:
        raise SystemExit(f"No batches found for split '{split}' under {root}")
    return np.concatenate(xs), np.concatenate(ys)


def normalize(arr, mean=None, std=None, per_window=False):
    """Normalize (N, C, T) windows either with fixed THUSZ stats or per-window
    per-channel z-scoring (each window normalized against its own mean/std
    across time -- robust to cross-dataset amplitude/scale differences)."""
    if per_window:
        m = arr.mean(axis=-1, keepdims=True)
        s = arr.std(axis=-1, keepdims=True)
        return (arr - m) / (s + 1e-6)
    return (arr - mean) / (std + 1e-6)


def _autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
            else torch.autocast("cpu", enabled=False))


@torch.no_grad()
def embed_all(backbone, X, device, batch_size=256):
    pool = nn.AdaptiveAvgPool1d(1)
    probe = torch.tensor(PROBE, device=device, dtype=torch.long)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        with _autocast(device):
            feats = []
            for ts in probe:
                t_emb = backbone.time_mlp(ts.expand(xb.shape[0]))
                h = backbone.init_conv(xb)
                outs = []
                for ml in backbone.down_blocks:
                    if len(ml) == 1 and isinstance(ml[0], nn.Conv1d):
                        h = ml[0](h)
                    else:
                        for blk in ml:
                            h = blk(h, t_emb) if "time_emb" in blk.forward.__code__.co_varnames else blk(h)
                        outs.append(pool(h).squeeze(-1))
                feats.append(torch.cat(outs, dim=1))
            e = torch.cat(feats, dim=1).float()
        out.append(e.cpu().numpy())
    return np.concatenate(out)


class Head(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=512, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, n_cls))

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", required=True)
    p.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE)
    p.add_argument("--norm-stats-dir", default=DEFAULT_NORM)
    p.add_argument("--out-root", default=DEFAULT_OUT)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--embed-batch-size", type=int, default=256,
                   help="batch for backbone embedding extraction; lower it for long "
                        "windows (self-attention is quadratic in sequence length)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classes", nargs="+", default=None,
                   help="restrict to this subset of classes (default: all discovered)")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="cap windows per class per split, subsampled at the FILE level "
                        "(never loads more than needed) -- for huge datasets like sleep_edfx")
    p.add_argument("--weight-power", type=float, default=1.0,
                   help="exponent on the inverse-frequency class weight (1.0=full, 0.5=sqrt-softened)")
    p.add_argument("--per-window-norm", action="store_true",
                   help="z-score each window against its OWN per-channel mean/std "
                        "(across time) instead of the fixed THUSZ normalization stats")
    p.add_argument("--pooled-random-split", action="store_true",
                   help="DIAGNOSTIC: pool the dataset's train+test windows and draw a "
                        "fresh random 80/20 split, ignoring the official (usually "
                        "subject-wise) split. Leaks subjects between train/test on "
                        "purpose -- only use to check whether poor official-split "
                        "performance is a cross-subject generalization issue.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {device}")

    mean = np.load(Path(args.norm_stats_dir) / "mean.npy").reshape(-1, 1).astype(np.float32)
    std = np.load(Path(args.norm_stats_dir) / "std.npy").reshape(-1, 1).astype(np.float32)

    classes = args.classes if args.classes else discover_classes(args.processed_root, "train")
    test_classes = discover_classes(args.processed_root, "test")
    if not args.classes:
        assert classes == test_classes or set(classes) == set(test_classes), \
            f"train/test class mismatch: {classes} vs {test_classes}"
    n_cls = len(classes)
    print(f"Classes ({n_cls}): {classes}")

    print("Loading raw data...")
    Xtr_raw, ytr = load_split_raw(args.processed_root, "train", classes, args.max_per_class, args.seed)
    Xte_raw, yte = load_split_raw(args.processed_root, "test", classes, args.max_per_class, args.seed)
    print(f"  train {Xtr_raw.shape}  test {Xte_raw.shape}")

    if args.pooled_random_split:
        print("  ** POOLED RANDOM SPLIT (diagnostic, ignores official subject-wise split) **")
        X_all = np.concatenate([Xtr_raw, Xte_raw])
        y_all = np.concatenate([ytr, yte])
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(X_all))
        n_test = int(round(0.2 * len(idx)))
        te_idx, tr_idx = idx[:n_test], idx[n_test:]
        Xtr_raw, ytr = X_all[tr_idx], y_all[tr_idx]
        Xte_raw, yte = X_all[te_idx], y_all[te_idx]
        print(f"  re-split -> train {Xtr_raw.shape}  test {Xte_raw.shape}")

    Xtr_raw = normalize(Xtr_raw, mean, std, per_window=args.per_window_norm)
    Xte_raw = normalize(Xte_raw, mean, std, per_window=args.per_window_norm)
    print(f"  normalization: {'per-window' if args.per_window_norm else args.norm_stats_dir}")

    print("Loading frozen backbone...")
    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device).eval()
    ckpt = torch.load(args.backbone_ckpt, map_location=device)
    backbone.load_state_dict(ckpt["model_state_dict"])
    for pa in backbone.parameters():
        pa.requires_grad = False

    print("Extracting embeddings...")
    Xtr = embed_all(backbone, Xtr_raw, device, batch_size=args.embed_batch_size)
    Xte = embed_all(backbone, Xte_raw, device, batch_size=args.embed_batch_size)
    print(f"  embeddings: train {Xtr.shape}  test {Xte.shape}")

    # carve a val split from train for early stopping
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(Xtr))
    n_val = max(1, int(round(args.val_frac * len(idx))))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    scaler = StandardScaler().fit(Xtr[tr_idx])
    Xtr_s = torch.tensor(scaler.transform(Xtr[tr_idx]), dtype=torch.float32)
    Xval_s = torch.tensor(scaler.transform(Xtr[val_idx]), dtype=torch.float32)
    Xte_s = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr[tr_idx], dtype=torch.long)
    yval, ytest = ytr[val_idx], yte

    counts = np.bincount(ytr[tr_idx], minlength=n_cls).astype(np.float32)
    w = (counts.sum() / (n_cls * np.maximum(counts, 1))) ** args.weight_power
    print(f"  class weights (power={args.weight_power}): "
          + ", ".join(f"{c}={wi:.2f}" for c, wi in zip(classes, w)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))

    model = Head(Xtr_s.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = len(Xtr_s)
    best_f1, best_state, patience = -1.0, None, 0
    print("Training head...")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            bidx = perm[i:i + args.batch_size]
            xb = Xtr_s[bidx].to(device); yb = ytr_t[bidx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xval_s.to(device)).argmax(1).cpu().numpy()
        vf1 = f1_score(yval, vp, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, patience = vf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop @ epoch {ep+1} (best val macroF1 {best_f1:.4f})")
                break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(Xte_s.to(device))
        test_probs = torch.softmax(test_logits.float(), dim=1).cpu().numpy()
        test_pred = test_logits.argmax(1).cpu().numpy()

    macro_f1 = f1_score(ytest, test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(ytest, test_pred, average="weighted", zero_division=0)
    acc = float((test_pred == ytest).mean())
    report = classification_report(ytest, test_pred, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(ytest, test_pred, labels=list(range(n_cls)))

    roc_auc = pr_auc = None
    if n_cls == 2:
        pos_prob = test_probs[:, 1]
        roc_auc = float(roc_auc_score(ytest, pos_prob))
        pr_auc = float(average_precision_score(ytest, pos_prob))
        print(f"\nBinary (positive class = '{classes[1]}'): ROC-AUC {roc_auc:.4f}  PR-AUC {pr_auc:.4f}")

    print(f"\nTest accuracy {acc:.4f}  macro-F1 {macro_f1:.4f}  weighted-F1 {weighted_f1:.4f}")
    print(report)
    print("Confusion matrix", classes, "\n", cm)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.dataset_name}_trainedhead_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "confusion_matrix.npy", cm)
    summary = dict(dataset=args.dataset_name, processed_root=args.processed_root,
                   backbone_ckpt=args.backbone_ckpt, classes=classes,
                   n_train=int(len(ytr)), n_test=int(len(ytest)),
                   best_val_macro_f1=float(best_f1), test_accuracy=acc,
                   test_macro_f1=float(macro_f1), test_weighted_f1=float(weighted_f1),
                   roc_auc=roc_auc, pr_auc=pr_auc, config=vars(args))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    roc_line = f"ROC-AUC {roc_auc:.4f}   PR-AUC {pr_auc:.4f}\n\n" if roc_auc is not None else ""
    (out_dir / "report.txt").write_text(
        f"{args.dataset_name} TRAINED-HEAD eval (frozen backbone embedding + freshly trained MLP head)\n"
        f"classes: {classes}\ntrain n={len(ytr)}  test n={len(ytest)}\n\n"
        f"Test accuracy {acc:.4f}   macro-F1 {macro_f1:.4f}   weighted-F1 {weighted_f1:.4f}\n"
        f"{roc_line}"
        f"{report}\nConfusion matrix {classes}:\n{cm}\n")
    print(f"\n✓ saved -> {out_dir}/")


if __name__ == "__main__":
    main()
