#!/usr/bin/env python3
"""
Standalone EVAL for the binary seizure classifier.

Loads a `best_classifier.pth` checkpoint (saved by finetune_binary_PW.py) and
reports test-set metrics on the held-out EVAL split WITHOUT retraining:

  - ROC-AUC, PR-AUC (threshold-free)
  - classification_report at the default 0.5 threshold
  - the F1-OPTIMAL threshold swept on VAL, and eval metrics at that threshold

Model/data definitions are copied (not imported) from finetune_binary_PW.py,
because importing that module would launch a full training run.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             f1_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Import the v2 backbone (safe: defining the model class has no side effects)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Diff_EEG_train_v2 import DeepEnhancedEEGDiffusionModel

BASE = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all"
NORM_STATS_DIR = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/normalization"
DEFAULT_CKPT = "/home/abdulh/scratch/seizure_clf_20260629_084706/best_classifier.pth"
RES_ROOT = "/home/abdulh/scratch/EEGdiff_V2/TUHSZ_Res"

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=DEFAULT_CKPT)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--num-workers", type=int, default=8)
p.add_argument("--res-root", default=RES_ROOT)
args = p.parse_args()

out_dir = Path(args.res_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Results dir: {out_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# Normalization + dataset (mirrors finetune_binary_PW.py LazyEEGDataset)
# ----------------------------------------------------------------------
mean = np.load(Path(NORM_STATS_DIR) / "mean.npy").astype(np.float32)
std = np.load(Path(NORM_STATS_DIR) / "std.npy").astype(np.float32)


def gather_files(dir_label_pairs):
    out = []
    for directory, label in dir_label_pairs:
        for f in sorted(Path(directory).glob("*_batch_*.npy")):
            if "_labels" not in f.name:
                out.append((f, label))
    return out


class LazyEEGDataset(Dataset):
    def __init__(self, file_label_pairs, mean, std):
        self.mean = mean.reshape(-1, 1).astype(np.float32)
        self.std = std.reshape(-1, 1).astype(np.float32)
        self.files, self.labels_per_file, self.index = [], [], []
        for f, label in file_label_pairs:
            n = np.load(f, mmap_mode='r').shape[0]
            fid = len(self.files)
            self.files.append(f)
            self.labels_per_file.append(label)
            self.index.extend((fid, i) for i in range(n))
        self._cache = {}

    def _arr(self, fid):
        a = self._cache.get(fid)
        if a is None:
            a = np.load(self.files[fid], mmap_mode='r')
            self._cache[fid] = a
        return a

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fid, row = self.index[idx]
        x = np.asarray(self._arr(fid)[row], dtype=np.float32)
        x = (x - self.mean) / (self.std + 1e-6)
        return torch.from_numpy(x), int(self.labels_per_file[fid])


# ----------------------------------------------------------------------
# Models (must match finetune_binary_PW.py exactly for state_dict load)
# ----------------------------------------------------------------------
class EEGClassifier(nn.Module):
    def __init__(self, backbone, feature_dim=256, num_classes=2, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('probe_timesteps', torch.tensor([50, 250, 500, 750, 950], dtype=torch.long))
        self.pool = nn.AdaptiveAvgPool1d(1)
        multi_scale_per_t = backbone.model_channels * sum(backbone.channel_multipliers)
        aggregated_dim = multi_scale_per_t * len(self.probe_timesteps)
        self.classifier = nn.Sequential(
            nn.Linear(aggregated_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(feature_dim, num_classes),
        )

    def _encode_at_t(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x)
        level_outputs = []
        for module_list in self.backbone.down_blocks:
            if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
                h = module_list[0](h)
            else:
                for block in module_list:
                    if hasattr(block, 'forward') and 'time_emb' in block.forward.__code__.co_varnames:
                        h = block(h, t_emb)
                    else:
                        h = block(h)
                level_outputs.append(self.pool(h).squeeze(-1))
        return torch.cat(level_outputs, dim=1)

    def forward(self, x):
        B = x.shape[0]
        feats = [self._encode_at_t(x, ts.expand(B)) for ts in self.probe_timesteps]
        combined = torch.cat(feats, dim=1)
        logits = self.classifier(combined)
        return logits, combined


class ReinforcedDecisionLayer(nn.Module):
    def __init__(self, input_dim, rl_weight=0.1, momentum=0.9):
        super().__init__()
        self.rl_weight = rl_weight
        self.momentum = momentum
        hidden = max(64, input_dim // 4)
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
        self.register_buffer('baseline', torch.tensor(0.0))

    def forward(self, logits, features, training=False):
        adj = self.policy(features).squeeze(-1)
        adjusted = logits.clone()
        adjusted[:, 1] += adj
        adjusted[:, 0] -= adj
        probs = torch.softmax(adjusted, dim=1)[:, 1]
        return adjusted, probs, None, None


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def predict(classifier, rl_layer, loader, device):
    classifier.eval()
    rl_layer.eval()
    probs_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with _autocast(device):
            logits, features = classifier(x)
            _, probs, _, _ = rl_layer(logits, features, training=False)
        probs_all.extend(probs.float().cpu().numpy())
        labels_all.extend(y.numpy())
    return np.array(probs_all), np.array(labels_all)


def best_f1_threshold(labels, probs):
    """Sweep PR-curve thresholds, return (threshold, f1) maximizing seizure F1."""
    prec, rec, thr = precision_recall_curve(labels, probs)
    # precision_recall_curve returns len(thr) = len(prec) - 1
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


# ----------------------------------------------------------------------
# Build model from checkpoint's stored arch, load weights
# ----------------------------------------------------------------------
print(f"\nLoading checkpoint: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location=device)
ARCH = ckpt['arch']
print(f"  epoch={ckpt.get('epoch')}  best_dev_auc={ckpt.get('best_dev_auc'):.4f}")
print(f"  arch={ARCH}")

backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
backbone.eval()
agg_dim = ARCH['model_channels'] * sum(ARCH['channel_multipliers']) * 5
classifier = EEGClassifier(backbone, feature_dim=256, dropout=0.4).to(device)
rl_layer = ReinforcedDecisionLayer(input_dim=agg_dim, rl_weight=0.1).to(device)
classifier.load_state_dict(ckpt['classifier'])
rl_layer.load_state_dict(ckpt['rl_layer'])
print("  ✓ weights loaded\n")

# ----------------------------------------------------------------------
# Rebuild the SAME val split (file-level, seed 42) + eval set
# ----------------------------------------------------------------------
all_files = gather_files([
    (f"{BASE}/train-non-seizure", 0),
    (f"{BASE}/train-seizure",     1),
    (f"{BASE}/dev-non-seizure",   0),
    (f"{BASE}/dev-seizure",       1),
])
split_rng = random.Random(42)
by_class = {0: [], 1: []}
for f, l in all_files:
    by_class[l].append((f, l))
val_files = []
for l, items in by_class.items():
    split_rng.shuffle(items)
    k = max(1, int(round(0.05 * len(items))))
    val_files += items[:k]

eval_files = gather_files([
    (f"{BASE}/eval-non-seizure", 0),
    (f"{BASE}/eval-seizure",     1),
])

val_ds = LazyEEGDataset(val_files, mean, std)
eval_ds = LazyEEGDataset(eval_files, mean, std)
print(f"Val  samples: {len(val_ds):,}   Eval samples: {len(eval_ds):,}\n")

dl_kw = dict(batch_size=args.batch_size, shuffle=False,
             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
val_loader = DataLoader(val_ds, **dl_kw)
eval_loader = DataLoader(eval_ds, **dl_kw)

# ----------------------------------------------------------------------
# Predict + report
# ----------------------------------------------------------------------
print("Scoring val...")
val_probs, val_labels = predict(classifier, rl_layer, val_loader, device)
print("Scoring eval...")
eval_probs, eval_labels = predict(classifier, rl_layer, eval_loader, device)

# F1-optimal threshold chosen on VAL (no eval leakage), applied to eval
thr_star, val_f1_star = best_f1_threshold(val_labels, val_probs)
print(f"\nF1-optimal threshold (chosen on val): {thr_star:.4f}  (val seizure F1={val_f1_star:.4f})")

# Save raw scores so results are fully reproducible without rerunning the model
np.savez_compressed(out_dir / "scores.npz",
                    val_probs=val_probs, val_labels=val_labels,
                    eval_probs=eval_probs, eval_labels=eval_labels)

results = {
    "checkpoint": str(args.ckpt),
    "epoch": int(ckpt.get("epoch")) if ckpt.get("epoch") is not None else None,
    "best_dev_auc": float(ckpt.get("best_dev_auc")) if ckpt.get("best_dev_auc") is not None else None,
    "arch": ARCH,
    "f1_optimal_threshold": thr_star,
    "splits": {},
}
report_lines = [f"checkpoint: {args.ckpt}",
                f"epoch: {results['epoch']}  best_dev_auc: {results['best_dev_auc']}",
                f"F1-optimal threshold (from val): {thr_star:.4f}\n"]

for name, probs, labels in [("val", val_probs, val_labels), ("eval", eval_probs, eval_labels)]:
    roc = float(roc_auc_score(labels, probs))
    pc, rc, _ = precision_recall_curve(labels, probs)
    pr_auc = float(auc(rc, pc))
    preds_05 = (probs >= 0.5).astype(int)
    preds_t = (probs >= thr_star).astype(int)
    f1_05 = float(f1_score(labels, preds_05, pos_label=1))
    f1_t = float(f1_score(labels, preds_t, pos_label=1))
    rep_05 = classification_report(labels, preds_05, target_names=['Normal', 'Seizure'], digits=4)
    rep_t = classification_report(labels, preds_t, target_names=['Normal', 'Seizure'], digits=4)

    block = (f"\n{'='*60}\n{name.upper()} RESULTS\n{'='*60}\n"
             f"ROC-AUC: {roc:.4f}    PR-AUC: {pr_auc:.4f}\n"
             f"\n-- threshold = 0.50 (default) --\nseizure F1: {f1_05:.4f}\n{rep_05}\n"
             f"-- threshold = {thr_star:.4f} (F1-optimal, from val) --\nseizure F1: {f1_t:.4f}\n{rep_t}")
    print(block)
    report_lines.append(block)

    results["splits"][name] = {
        "roc_auc": roc, "pr_auc": pr_auc,
        "thr_0.5": {"seizure_f1": f1_05, "report": rep_05},
        "thr_optimal": {"seizure_f1": f1_t, "report": rep_t},
    }

(out_dir / "results.json").write_text(json.dumps(results, indent=2))
(out_dir / "results.txt").write_text("\n".join(report_lines) + "\n")
print(f"\n✓ Eval complete. Results saved to {out_dir}/")
