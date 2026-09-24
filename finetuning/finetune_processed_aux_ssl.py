#!/usr/bin/env python3
"""Auxiliary-SSL fine-tuning: fine-tune the DiffEEG backbone for classification
while keeping its ORIGINAL self-supervised diffusion (denoising) objective as an
auxiliary regularizer.

Motivation: on out-of-distribution tasks (Sleep-EDFx, TUEV, motor imagery) the
frozen embedding is weak. Plain fine-tuning can overfit / drift the backbone off
its pretraining manifold. Adding the denoising loss back in during fine-tuning
keeps the representation grounded in the generative objective it was pretrained
on, which can improve generalization.

  total_loss = CrossEntropy(logits, y)  +  ssl_coef * MSE(pred_noise, noise)

where the SSL term is exactly the pretraining loss: add noise to the SAME batch
at a random timestep, run the full U-Net, and predict the noise. ssl_coef=0
recovers plain fine-tuning (the baseline of the sweep).

Reuses the working data loading + EEGClassifier from finetune_processed_multiclass.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Diff_EEG_train_v2 import DeepEnhancedEEGDiffusionModel, ImprovedDiffusionScheduler
from finetune_processed_multiclass import (
    ARCH, PROBE, DEFAULT_BACKBONE, DEFAULT_NORM, _autocast,
    discover_classes, load_split_raw, normalize, EEGClassifier, apply_unfreezing,
)


def ssl_denoise_loss(backbone, x, scheduler):
    """The pretraining objective on the current batch: predict the added noise."""
    t = scheduler.sample_random_timesteps(x.shape[0])
    noisy, noise = scheduler.add_noise(x, t)
    pred_noise = backbone(noisy, t)
    return F.mse_loss(pred_noise.float(), noise.float())


def evaluate(clf, loader, device):
    clf.eval()
    preds, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            with _autocast(device):
                logits = clf(x.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            labs.append(y.numpy())
    p, l = np.concatenate(preds), np.concatenate(labs)
    return f1_score(l, p, average="macro", zero_division=0), p, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE)
    ap.add_argument("--norm-stats-dir", default=DEFAULT_NORM)
    ap.add_argument("--out-root", default="/home/abdulh/scratch/EEGdiff_V2/Benchmarking")
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--unfreeze", default="all",
                    choices=["none", "last_level", "last_two_levels", "encoder_all", "all"])
    ap.add_argument("--ssl-coef", type=float, default=0.1,
                    help="weight of the auxiliary diffusion-denoising loss (0 = plain FT)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--weight-power", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | unfreeze={args.unfreeze} | ssl_coef={args.ssl_coef}")
    mean = np.load(Path(args.norm_stats_dir) / "mean.npy").reshape(-1, 1).astype(np.float32)
    std = np.load(Path(args.norm_stats_dir) / "std.npy").reshape(-1, 1).astype(np.float32)
    classes = args.classes or discover_classes(args.processed_root, "train")
    n_cls = len(classes)
    print(f"Classes ({n_cls}): {classes}")

    Xtr_raw, ytr = load_split_raw(args.processed_root, "train", classes, seed=args.seed)
    Xte_raw, yte = load_split_raw(args.processed_root, "test", classes, seed=args.seed)
    Xtr_raw = normalize(Xtr_raw, mean, std)
    Xte_raw = normalize(Xte_raw, mean, std)
    print(f"  train {Xtr_raw.shape}  test {Xte_raw.shape}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(Xtr_raw))
    n_val = max(1, int(round(args.val_frac * len(idx))))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def loader(X, y, shuffle, drop_last=False):
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                          batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, drop_last=drop_last,
                          pin_memory=(device.type == "cuda"))
    train_loader = loader(Xtr_raw[tr_idx], ytr[tr_idx], True, drop_last=True)
    val_loader = loader(Xtr_raw[val_idx], ytr[val_idx], False)
    test_loader = loader(Xte_raw, yte, False)

    print("Loading backbone...")
    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device).eval()
    backbone.load_state_dict(torch.load(args.backbone_ckpt, map_location=device)["model_state_dict"])
    trainable_bb = apply_unfreezing(backbone, args.unfreeze)
    print(f"  trainable backbone params: {sum(p.numel() for p in trainable_bb):,}")
    scheduler = ImprovedDiffusionScheduler(timesteps=1000, beta_schedule="cosine", device=device)

    clf = EEGClassifier(backbone, n_cls, grad_checkpoint=True).to(device)

    counts = np.bincount(ytr[tr_idx], minlength=n_cls).astype(np.float32)
    w = (counts.sum() / (n_cls * np.maximum(counts, 1))) ** args.weight_power
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    groups = [{"params": list(clf.classifier.parameters()), "lr": args.lr, "weight_decay": 1e-4}]
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
        clf.train(); clf.backbone.eval()
        ce_sum = ssl_sum = nb = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with _autocast(device):
                ce = loss_fn(clf(x), y)
                ssl = ssl_denoise_loss(backbone, x, scheduler) if args.ssl_coef > 0 else \
                      torch.zeros((), device=device)
                loss = ce + args.ssl_coef * ssl
            loss.backward()
            nn.utils.clip_grad_norm_(list(clf.classifier.parameters()) + trainable_bb, 1.0)
            opt.step()
            try: sched.step()
            except ValueError: pass
            ce_sum += float(ce); ssl_sum += float(ssl); nb += 1
        vf1, _, _ = evaluate(clf, val_loader, device)
        print(f"  epoch {ep+1:3d}/{args.epochs} | CE {ce_sum/nb:.4f} | SSL {ssl_sum/nb:.4f} "
              f"| val macroF1 {vf1:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1, patience = vf1, 0
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop @ epoch {ep+1}"); break

    clf.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, pred, ytest = evaluate(clf, test_loader, device)
    macro = f1_score(ytest, pred, average="macro", zero_division=0)
    wf1 = f1_score(ytest, pred, average="weighted", zero_division=0)
    acc = float((pred == ytest).mean())
    print(f"\n[ssl_coef={args.ssl_coef}] Test accuracy {acc:.4f}  macro-F1 {macro:.4f}  weighted-F1 {wf1:.4f}")
    print(classification_report(ytest, pred, target_names=classes, digits=4, zero_division=0))
    print("Confusion matrix", classes, "\n", confusion_matrix(ytest, pred))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_root) / f"{args.dataset_name}_auxssl_{args.ssl_coef}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(
        dataset=args.dataset_name, ssl_coef=args.ssl_coef, unfreeze=args.unfreeze,
        classes=classes, accuracy=acc, macro_f1=macro, weighted_f1=wf1,
        best_val_macro_f1=best_f1), indent=2))
    print(f"\n✓ saved -> {out}")


if __name__ == "__main__":
    main()
