#!/usr/bin/env python3
"""FINETUNING experiment for any processed external dataset (binary or multiclass).

Companion to eval_processed_multiclass.py's "no finetuning" (frozen backbone +
freshly trained head) experiment. Here the backbone's last N encoder levels are
also unfrozen and trained (discriminative LR), same pattern as
finetune_binary_PW_unfreeze.py / finetune_tuab_PW.py / finetune_cv_subtypes.py.
No RL layer -- plain weighted CE, to isolate the effect of unfreezing.

Expects `<root>/{train,test}-<classname>/*_batch_*.npy` (dataset supplies its
own held-out, subject-disjoint test split -- no CV needed).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Diff_EEG_train_v2 import DeepEnhancedEEGDiffusionModel

DEFAULT_NORM = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/normalization"
DEFAULT_BACKBONE = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/best_EEGDIFF_V2.pth"
DEFAULT_OUT = "/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
PROBE = [50, 250, 500, 750, 950]
ARCH = dict(in_channels=22, model_channels=64, channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=3, time_emb_dim=768, dropout=0.1, attention_heads=8)


def discover_classes(root, split):
    return sorted(d.name[len(split) + 1:] for d in Path(root).iterdir()
                 if d.is_dir() and d.name.startswith(f"{split}-"))


def load_split_raw(root, split, classes, max_per_class=None, seed=42):
    """With max_per_class, subsamples at the FILE level (random file order,
    stop once the cap is hit) -- never reads more than needed. Essential for
    huge datasets like sleep_edfx (~330GB full size)."""
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
    return np.concatenate(xs), np.concatenate(ys)


def normalize(arr, mean=None, std=None, per_window=False):
    if per_window:
        m = arr.mean(axis=-1, keepdims=True)
        s = arr.std(axis=-1, keepdims=True)
        return (arr - m) / (s + 1e-6)
    return (arr - mean) / (std + 1e-6)


class EEGClassifier(nn.Module):
    def __init__(self, backbone, n_cls, feature_dim=256, dropout=0.4, grad_checkpoint=True):
        super().__init__()
        self.backbone = backbone
        self.grad_checkpoint = grad_checkpoint
        self.register_buffer("probe", torch.tensor(PROBE, dtype=torch.long))
        self.pool = nn.AdaptiveAvgPool1d(1)
        agg = backbone.model_channels * sum(backbone.channel_multipliers) * len(PROBE)
        self.classifier = nn.Sequential(
            nn.Linear(agg, feature_dim * 2), nn.BatchNorm1d(feature_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim), nn.BatchNorm1d(feature_dim), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(feature_dim, n_cls))

    def _run(self, block, h, t_emb, takes_time):
        trainable = any(p.requires_grad for p in block.parameters())
        if self.grad_checkpoint and trainable and self.training and torch.is_grad_enabled():
            return checkpoint(block, h, t_emb, use_reentrant=False) if takes_time else \
                   checkpoint(block, h, use_reentrant=False)
        return block(h, t_emb) if takes_time else block(h)

    def _encode(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x)
        outs = []
        for ml in self.backbone.down_blocks:
            if len(ml) == 1 and isinstance(ml[0], nn.Conv1d):
                h = self._run(ml[0], h, t_emb, False)
            else:
                for b in ml:
                    tt = "time_emb" in b.forward.__code__.co_varnames
                    h = self._run(b, h, t_emb, tt)
                outs.append(self.pool(h).squeeze(-1))
        return torch.cat(outs, dim=1)

    def forward(self, x):
        B = x.shape[0]
        feats = torch.cat([self._encode(x, ts.expand(B)) for ts in self.probe], dim=1)
        return self.classifier(feats)


def apply_unfreezing(backbone, mode):
    for p in backbone.parameters():
        p.requires_grad = False
    down = backbone.down_blocks
    is_level = lambda ml: not (len(ml) == 1 and isinstance(ml[0], nn.Conv1d))
    if mode == "none":
        pass
    elif mode in ("encoder_all", "all"):
        for ml in down:
            for p in ml.parameters(): p.requires_grad = True
        if mode == "all":
            # Full finetuning of everything the classifier's forward pass actually
            # touches. bottleneck_blocks/up_blocks are diffusion-decoder components
            # never used by EEGClassifier._encode, so unfreezing them would be inert.
            for p in backbone.init_conv.parameters(): p.requires_grad = True
            for p in backbone.time_mlp.parameters(): p.requires_grad = True
    else:
        need = 1 if mode == "last_level" else 2
        seen = 0
        for ml in reversed(down):
            for p in ml.parameters(): p.requires_grad = True
            if is_level(ml):
                seen += 1
                if seen >= need:
                    break
    return [p for p in backbone.parameters() if p.requires_grad]


def _autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
            else torch.autocast("cpu", enabled=False))


def run_epoch(classifier, loader, opt, sched, loss_fn, trainable_bb, device, train):
    classifier.train(train)
    classifier.backbone.eval()
    preds_all, lab_all = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            opt.zero_grad()
            with _autocast(device):
                logits = classifier(x)
                loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(list(classifier.classifier.parameters()) + trainable_bb, 1.0)
            opt.step()
            if sched:
                try: sched.step()
                except ValueError: pass
        else:
            with torch.no_grad(), _autocast(device):
                logits = classifier(x)
        preds_all.append(logits.argmax(1).detach().cpu().numpy())
        lab_all.append(y.cpu().numpy())
    p, l = np.concatenate(preds_all), np.concatenate(lab_all)
    return f1_score(l, p, average="macro", zero_division=0), p, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE)
    ap.add_argument("--norm-stats-dir", default=DEFAULT_NORM)
    ap.add_argument("--out-root", default=DEFAULT_OUT)
    ap.add_argument("--unfreeze", choices=["none", "last_level", "last_two_levels", "encoder_all", "all"],
                    default="last_two_levels")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="restrict to this subset of classes (default: all discovered)")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap windows per class per split, subsampled at the FILE level")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="exponent on the inverse-frequency class weight (1.0=full, 0.5=sqrt-softened)")
    ap.add_argument("--per-window-norm", action="store_true",
                    help="z-score each window against its OWN per-channel mean/std "
                         "instead of the fixed THUSZ normalization stats")
    ap.add_argument("--pooled-random-split", action="store_true",
                    help="DIAGNOSTIC: pool train+test and draw a fresh random 80/20 "
                         "split, ignoring the official (usually subject-wise) split")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {device} | unfreeze={args.unfreeze}")

    mean = np.load(Path(args.norm_stats_dir) / "mean.npy").reshape(-1, 1).astype(np.float32)
    std = np.load(Path(args.norm_stats_dir) / "std.npy").reshape(-1, 1).astype(np.float32)

    classes = args.classes if args.classes else discover_classes(args.processed_root, "train")
    n_cls = len(classes)
    print(f"Classes ({n_cls}): {classes}")

    print("Loading data...")
    Xtr_raw, ytr = load_split_raw(args.processed_root, "train", classes, args.max_per_class, args.seed)
    Xte_raw, yte = load_split_raw(args.processed_root, "test", classes, args.max_per_class, args.seed)
    print(f"  train {Xtr_raw.shape}  test {Xte_raw.shape}")

    if args.pooled_random_split:
        print("  ** POOLED RANDOM SPLIT (diagnostic) **")
        X_all = np.concatenate([Xtr_raw, Xte_raw])
        y_all = np.concatenate([ytr, yte])
        split_rng = np.random.default_rng(args.seed)
        idx0 = split_rng.permutation(len(X_all))
        n_test = int(round(0.2 * len(idx0)))
        te_idx0, tr_idx0 = idx0[:n_test], idx0[n_test:]
        Xtr_raw, ytr = X_all[tr_idx0], y_all[tr_idx0]
        Xte_raw, yte = X_all[te_idx0], y_all[te_idx0]
        print(f"  re-split -> train {Xtr_raw.shape}  test {Xte_raw.shape}")

    Xtr_raw = normalize(Xtr_raw, mean, std, per_window=args.per_window_norm)
    Xte_raw = normalize(Xte_raw, mean, std, per_window=args.per_window_norm)
    print(f"  normalization: {'per-window' if args.per_window_norm else args.norm_stats_dir}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(Xtr_raw))
    n_val = max(1, int(round(args.val_frac * len(idx))))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_raw[tr_idx]), torch.from_numpy(ytr[tr_idx])),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_raw[val_idx]), torch.from_numpy(ytr[val_idx])),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(Xte_raw), torch.from_numpy(yte)),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Loading backbone...")
    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device).eval()
    ckpt = torch.load(args.backbone_ckpt, map_location=device)
    backbone.load_state_dict(ckpt["model_state_dict"])
    trainable_bb = apply_unfreezing(backbone, args.unfreeze)
    n_bb = sum(p.numel() for p in trainable_bb)
    print(f"  trainable backbone params: {n_bb:,}")

    classifier = EEGClassifier(backbone, n_cls, grad_checkpoint=args.grad_checkpoint).to(device)

    counts = np.bincount(ytr[tr_idx], minlength=n_cls).astype(np.float32)
    w = (counts.sum() / (n_cls * np.maximum(counts, 1))) ** args.weight_power
    print(f"  class weights (power={args.weight_power}): "
          + ", ".join(f"{c}={wi:.2f}" for c, wi in zip(classes, w)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))

    groups = [{"params": list(classifier.classifier.parameters()), "lr": args.lr, "weight_decay": 1e-4}]
    max_lrs = [args.lr]
    if trainable_bb:
        groups.append({"params": trainable_bb, "lr": args.lr * args.backbone_lr_mult, "weight_decay": 1e-4})
        max_lrs.append(args.lr * args.backbone_lr_mult)
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, epochs=args.epochs,
                                                steps_per_epoch=max(1, len(train_loader)), pct_start=0.1)

    best_f1, best_state, patience = -1.0, None, 0
    print("Training...")
    for ep in range(args.epochs):
        run_epoch(classifier, train_loader, opt, sched, loss_fn, trainable_bb, device, train=True)
        vf1, _, _ = run_epoch(classifier, val_loader, None, None, loss_fn, trainable_bb, device, train=False)
        print(f"  epoch {ep+1:3d}/{args.epochs} | val macroF1 {vf1:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1, patience = vf1, 0
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop @ epoch {ep+1}")
                break

    classifier.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, test_pred, ytest = run_epoch(classifier, test_loader, None, None, loss_fn, trainable_bb, device, train=False)

    macro_f1 = f1_score(ytest, test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(ytest, test_pred, average="weighted", zero_division=0)
    acc = float((test_pred == ytest).mean())
    report = classification_report(ytest, test_pred, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(ytest, test_pred, labels=list(range(n_cls)))

    print(f"\nTest accuracy {acc:.4f}  macro-F1 {macro_f1:.4f}  weighted-F1 {weighted_f1:.4f}")
    print(report)
    print("Confusion matrix", classes, "\n", cm)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.dataset_name}_finetuned_{args.unfreeze}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "confusion_matrix.npy", cm)
    torch.save({"classifier": best_state, "arch": ARCH, "unfreeze": args.unfreeze, "classes": classes},
               out_dir / "best_classifier.pth")
    summary = dict(dataset=args.dataset_name, processed_root=args.processed_root, unfreeze=args.unfreeze,
                   classes=classes, n_train=int(len(ytr)), n_test=int(len(ytest)),
                   trainable_backbone_params=int(n_bb), best_val_macro_f1=float(best_f1),
                   test_accuracy=acc, test_macro_f1=float(macro_f1), test_weighted_f1=float(weighted_f1),
                   config=vars(args))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.txt").write_text(
        f"{args.dataset_name} FINETUNED eval (backbone unfreeze={args.unfreeze} + trained head)\n"
        f"classes: {classes}\ntrain n={len(ytr)}  test n={len(ytest)}\n"
        f"trainable backbone params: {n_bb:,}\n\n"
        f"Test accuracy {acc:.4f}   macro-F1 {macro_f1:.4f}   weighted-F1 {weighted_f1:.4f}\n\n"
        f"{report}\nConfusion matrix {classes}:\n{cm}\n")
    print(f"\n✓ saved -> {out_dir}/")


if __name__ == "__main__":
    main()
