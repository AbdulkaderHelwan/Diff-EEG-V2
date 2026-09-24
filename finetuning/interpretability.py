#!/usr/bin/env python3
"""Interpretability + embedding analysis for the DiffEEG paper.

Produces, from the REAL pretrained backbone and the finetuned TUEV 4-class
classifier:
  1. t-SNE of the 4800-d embedding (frozen pretrained vs. task-finetuned),
     coloured by clinical event class -> shows the embedding organises
     clinically meaningful categories.
  2. Gradient-based saliency for the 'epileptiform' class: a channel x time
     map and a per-channel importance vector -> which electrodes/timepoints
     drive the epileptiform decision (rendered later as a 10-20 topomap).

All outputs are saved as .npy so the paper figures are built from real data.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Diff_EEG_train_v2 import DeepEnhancedEEGDiffusionModel

PROBE = [50, 250, 500, 750, 950]
ROOT = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUEV_v2.0.1/processed_TUEV_4class"
CKPT = "/home/abdulh/scratch/EEGdiff_V2/Benchmarking/tuev_4class_finetuned_all_RL_20260708_072026/best_classifier.pth"
FROZEN = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/best_EEGDIFF_V2.pth"
CLASSES = ["artf", "bckg", "epileptiform", "eyem"]
OUT = Path("/home/abdulh/scratch/EEGdiff_V2/paper/analysis")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EEGClassifier(nn.Module):
    def __init__(self, backbone, n_cls, feature_dim=256, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("probe", torch.tensor(PROBE, dtype=torch.long))
        self.pool = nn.AdaptiveAvgPool1d(1)
        agg = backbone.model_channels * sum(backbone.channel_multipliers) * len(PROBE)
        self.classifier = nn.Sequential(
            nn.Linear(agg, feature_dim * 2), nn.BatchNorm1d(feature_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim), nn.BatchNorm1d(feature_dim), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(feature_dim, n_cls))

    def _encode(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x)
        outs = []
        for ml in self.backbone.down_blocks:
            if len(ml) == 1 and isinstance(ml[0], nn.Conv1d):
                h = ml[0](h)
            else:
                for b in ml:
                    tt = "time_emb" in b.forward.__code__.co_varnames
                    h = b(h, t_emb) if tt else b(h)
                outs.append(self.pool(h).squeeze(-1))
        return torch.cat(outs, dim=1)

    def forward(self, x):
        B = x.shape[0]
        feats = torch.cat([self._encode(x, ts.expand(B)) for ts in self.probe], dim=1)
        return self.classifier(feats), feats


def load_test():
    xs, ys = [], []
    for ci, c in enumerate(CLASSES):
        for f in sorted((Path(ROOT) / f"test-{c}").glob("*_batch_*.npy")):
            a = np.load(f).astype(np.float32)
            xs.append(a); ys.append(np.full(len(a), ci, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def load_backbone_from(state, arch):
    m = DeepEnhancedEEGDiffusionModel(**arch).to(DEV)
    # accept several checkpoint layouts
    sd = state
    for key in ("model_state_dict", "state_dict", "model", "ema"):
        if isinstance(state, dict) and key in state and hasattr(state[key], "keys"):
            sd = state[key]; break
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  backbone loaded (missing {len(missing)}, unexpected {len(unexpected)})")
    return m


def embed(model, X, mean, std, bs=64):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = (X[i:i + bs] - mean) / (std + 1e-6)
            x = torch.from_numpy(x).float().to(DEV)
            _, f = model(x)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats)


def saliency(model, X, y, target, mean, std, bs=32):
    """Mean |d logit_target / d input| over windows of class `target`."""
    model.eval()
    idx = np.where(y == target)[0]
    acc = np.zeros((22, 1280), dtype=np.float64)
    n = 0
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        x = (X[sel] - mean) / (std + 1e-6)
        x = torch.from_numpy(x).float().to(DEV).requires_grad_(True)
        logits, _ = model(x)
        model.zero_grad()
        logits[:, target].sum().backward()
        g = x.grad.detach().abs().cpu().numpy()  # (b,22,1280)
        acc += g.sum(axis=0)
        n += len(sel)
    return acc / max(n, 1)


def load_root(root, classes, splits=("train", "test")):
    xs, ys = [], []
    for ci, c in enumerate(classes):
        for sp in splits:
            dd = Path(root) / f"{sp}-{c}"
            if not dd.is_dir():
                continue
            for f in sorted(dd.glob("*_batch_*.npy")):
                a = np.load(f).astype(np.float32)
                xs.append(a); ys.append(np.full(len(a), ci, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def knn_sep(emb, y, k=10):
    """Balanced-accuracy 5-fold kNN in embedding space (imbalance-robust)."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(emb)
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, Z, y, cv=cv, scoring="balanced_accuracy").mean())


def main():
    print("Loading TUEV test data...")
    X, y = load_test()
    print(f"  {len(X)} windows, classes {np.bincount(y)}")
    mean = np.load(Path(ROOT) / "normalization/mean.npy").reshape(1, -1, 1).astype(np.float32)
    std = np.load(Path(ROOT) / "normalization/std.npy").reshape(1, -1, 1).astype(np.float32)

    arch = torch.load(CKPT, map_location="cpu")["arch"]

    # --- finetuned classifier (backbone + head) ---
    print("Loading finetuned TUEV classifier...")
    ck = torch.load(CKPT, map_location=DEV)
    clf = EEGClassifier(DeepEnhancedEEGDiffusionModel(**arch).to(DEV), n_cls=len(CLASSES)).to(DEV)
    clf.load_state_dict(ck["classifier"])

    # --- frozen pretrained backbone (for the 'pretraining already separates' claim) ---
    print("Loading frozen pretrained backbone...")
    frozen_bb = load_backbone_from(torch.load(FROZEN, map_location=DEV), arch)
    frozen = EEGClassifier(frozen_bb, n_cls=len(CLASSES)).to(DEV)  # random head; only feats used

    print("Extracting embeddings (finetuned)...")
    emb_ft = embed(clf, X, mean, std)
    print("Extracting embeddings (frozen)...")
    emb_fz = embed(frozen, X, mean, std)

    print("t-SNE...")
    ts_ft = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb_ft)
    ts_fz = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb_fz)

    print("Saliency for 'epileptiform' (finetuned)...")
    epi = CLASSES.index("epileptiform")
    sal = saliency(clf, X, y, epi, mean, std)  # (22,1280)

    # kNN separability in embedding space (quantitative)
    knn_ft = knn_sep(emb_ft, y)
    knn_fz = knn_sep(emb_fz, y)
    print(f"kNN separability TUEV frozen={knn_fz:.3f}  finetuned={knn_ft:.3f}")

    # --- Bonn seizure vs normal on the frozen embedding (clean 2-class case) ---
    print("Bonn: frozen embeddings + t-SNE...")
    BONN = "/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/bonn"
    Xb, yb = load_root(BONN, ["normal", "seizure"])
    mb = np.load("/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/normalization/mean.npy").reshape(1, -1, 1).astype(np.float32)
    sb = np.load("/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/normalization/std.npy").reshape(1, -1, 1).astype(np.float32)
    emb_b = embed(frozen, Xb, mb, sb)
    ts_b = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb_b)
    knn_b = knn_sep(emb_b, yb)
    print(f"Bonn: {len(Xb)} windows, kNN separability (frozen) = {knn_b:.3f}")

    np.savez(OUT / "interp.npz",
             y=y, classes=np.array(CLASSES),
             tsne_finetuned=ts_ft, tsne_frozen=ts_fz,
             saliency_map=sal, sal_channel=sal.mean(axis=1),
             sal_time=sal.mean(axis=0),
             knn_tuev_frozen=knn_fz, knn_tuev_finetuned=knn_ft,
             bonn_tsne=ts_b, bonn_y=yb, bonn_classes=np.array(["normal", "seizure"]),
             knn_bonn_frozen=knn_b,
             emb_frozen=emb_fz, emb_finetuned=emb_ft, emb_bonn=emb_b)
    print(f"Saved -> {OUT/'interp.npz'}")
    # quick text summary
    ch = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8",
          "T3","T4","T5","T6","A1","A2","FZ","CZ","PZ","ROC"]
    order = np.argsort(-sal.mean(axis=1))
    print("Top-8 channels by epileptiform saliency:",
          [(ch[i], round(float(sal.mean(axis=1)[i]), 4)) for i in order[:8]])


if __name__ == "__main__":
    main()
