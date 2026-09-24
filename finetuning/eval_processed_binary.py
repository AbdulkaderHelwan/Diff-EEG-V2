#!/usr/bin/env python3
"""Evaluate a saved binary EEGdiff classifier on any TUAB-style processed root."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Diff_EEG_train_v2 import DeepEnhancedEEGDiffusionModel


DEFAULT_NORM = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/normalization"
DEFAULT_OUT = "/home/abdulh/scratch/EEGdiff_V2/External_Eval_Res"


def gather_files(root):
    pairs = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        name = d.name.lower()
        if any(x in name for x in ["normal", "background", "wake", "control"]):
            label = 0
        elif any(x in name for x in ["seizure", "event", "abnormal"]):
            label = 1
        else:
            continue
        for f in sorted(d.glob("*_batch_*.npy")):
            pairs.append((f, label))
    if not pairs:
        raise SystemExit(f"No binary batch files found under {root}")
    return pairs


class LazyEEGDataset(Dataset):
    def __init__(self, file_label_pairs, mean, std):
        self.mean = mean.reshape(-1, 1).astype(np.float32)
        self.std = std.reshape(-1, 1).astype(np.float32)
        self.files, self.labels, self.index, self._cache = [], [], [], {}
        for f, label in file_label_pairs:
            n = np.load(f, mmap_mode="r").shape[0]
            fid = len(self.files)
            self.files.append(f)
            self.labels.append(label)
            self.index.extend((fid, i) for i in range(n))

    def __len__(self):
        return len(self.index)

    def _arr(self, fid):
        if fid not in self._cache:
            self._cache[fid] = np.load(self.files[fid], mmap_mode="r")
        return self._cache[fid]

    def __getitem__(self, idx):
        fid, row = self.index[idx]
        x = np.asarray(self._arr(fid)[row], dtype=np.float32)
        x = (x - self.mean) / (self.std + 1e-6)
        return torch.from_numpy(x), int(self.labels[fid])


class EEGClassifier(nn.Module):
    def __init__(self, backbone, feature_dim=256, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("probe_timesteps", torch.tensor([50, 250, 500, 750, 950], dtype=torch.long))
        self.pool = nn.AdaptiveAvgPool1d(1)
        agg = backbone.model_channels * sum(backbone.channel_multipliers) * len(self.probe_timesteps)
        self.classifier = nn.Sequential(
            nn.Linear(agg, feature_dim * 2), nn.BatchNorm1d(feature_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim), nn.BatchNorm1d(feature_dim), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(feature_dim, 2),
        )

    def _encode_at_t(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x)
        outs = []
        for module_list in self.backbone.down_blocks:
            if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
                h = module_list[0](h)
            else:
                for block in module_list:
                    if hasattr(block, "forward") and "time_emb" in block.forward.__code__.co_varnames:
                        h = block(h, t_emb)
                    else:
                        h = block(h)
                outs.append(self.pool(h).squeeze(-1))
        return torch.cat(outs, dim=1)

    def forward(self, x):
        b = x.shape[0]
        feats = [self._encode_at_t(x, ts.expand(b)) for ts in self.probe_timesteps]
        combined = torch.cat(feats, dim=1)
        return self.classifier(combined), combined


class ReinforcedDecisionLayer(nn.Module):
    def __init__(self, input_dim, rl_weight=0.1, momentum=0.9):
        super().__init__()
        hidden = max(64, input_dim // 4)
        self.policy = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
        self.register_buffer("baseline", torch.tensor(0.0))

    def forward(self, logits, features, training=False):
        adj = self.policy(features).squeeze(-1)
        adjusted = logits.clone()
        adjusted[:, 1] += adj
        adjusted[:, 0] -= adj
        return adjusted, torch.softmax(adjusted, dim=1)[:, 1], None, None


def compute_metrics(labels, probs, threshold):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + 1e-9)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_acc": float(0.5 * (recall + specificity)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--norm-stats-dir", default=DEFAULT_NORM)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-root", default=DEFAULT_OUT)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    arch = ckpt["arch"]
    backbone = DeepEnhancedEEGDiffusionModel(**arch).to(device).eval()
    classifier = EEGClassifier(backbone).to(device)
    rl_layer = ReinforcedDecisionLayer(arch["model_channels"] * sum(arch["channel_multipliers"]) * 5).to(device)
    classifier.load_state_dict(ckpt["classifier"])
    rl_layer.load_state_dict(ckpt["rl_layer"])

    mean = np.load(Path(args.norm_stats_dir) / "mean.npy").astype(np.float32)
    std = np.load(Path(args.norm_stats_dir) / "std.npy").astype(np.float32)
    ds = LazyEEGDataset(gather_files(args.processed_root), mean, std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    probs, labels = [], []
    classifier.eval()
    rl_layer.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, features = classifier(x)
            _, p1, _, _ = rl_layer(logits, features)
            probs.extend(p1.float().cpu().numpy())
            labels.extend(y.numpy())

    threshold = args.threshold
    if threshold is None:
        threshold = float(ckpt.get("best_threshold", 0.5))
    metrics = compute_metrics(labels, probs, threshold)

    out_dir = Path(args.out_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scores.npz", probs=np.asarray(probs), labels=np.asarray(labels))
    with open(out_dir / "results.json", "w") as fp:
        json.dump({"processed_root": args.processed_root, "ckpt": args.ckpt, "metrics": metrics}, fp, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

