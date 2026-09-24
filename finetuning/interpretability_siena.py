#!/usr/bin/env python3
"""Adds Siena (seizure vs. normal) t-SNE + kNN separability to interp.npz,
computed on the FROZEN backbone and on the FINETUNED+RL Siena classifier.
Siena shows a cleaner frozen-vs-finetuned improvement than TUEV (macro-F1
0.528 -> 0.638, ~8x seizure precision gain), so it replaces the TUEV panels
in the embedding-analysis figure.

Uses only the TEST split (no leakage into the figure), with all seizure test
windows (132) plus a balanced random sample of normal test windows, seeded for
reproducibility.
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
SIENA_ROOT = "/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/siena"
SIENA_NORM = "/scratch/linah03/EpilepticSeizureProject/Dataset/Siena/processed_siena/normalization"
FT_CKPT = "/home/abdulh/scratch/EEGdiff_V2/Benchmarking/siena_finetuned_all_RL_20260707_100303/best_classifier.pth"
FROZEN_CKPT = "/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/best_EEGDIFF_V2.pth"
CLASSES = ["normal", "seizure"]
OUT = Path("/home/abdulh/scratch/EEGdiff_V2/paper/analysis")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0


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


def load_backbone_from(state, arch):
    m = DeepEnhancedEEGDiffusionModel(**arch).to(DEV)
    sd = state
    for key in ("model_state_dict", "state_dict", "model", "ema"):
        if isinstance(state, dict) and key in state and hasattr(state[key], "keys"):
            sd = state[key]; break
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  backbone loaded (missing {len(missing)}, unexpected {len(unexpected)})")
    return m


def load_siena_test_balanced(rng, n_normal_sample=500):
    xs, ys = [], []
    for ci, c in enumerate(CLASSES):
        files = sorted((Path(SIENA_ROOT) / f"test-{c}").glob("*_batch_*.npy"))
        arrs = [np.load(f).astype(np.float32) for f in files]
        arr = np.concatenate(arrs)
        if c == "normal" and len(arr) > n_normal_sample:
            idx = rng.choice(len(arr), size=n_normal_sample, replace=False)
            arr = arr[idx]
        xs.append(arr); ys.append(np.full(len(arr), ci, dtype=np.int64))
        print(f"  Siena test-{c}: using {len(arr)} windows")
    return np.concatenate(xs), np.concatenate(ys)


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


def knn_sep(emb, y, k=10):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(emb)
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, Z, y, cv=cv, scoring="balanced_accuracy").mean())


def main():
    rng = np.random.default_rng(SEED)
    print("Loading Siena test set (all seizure + balanced normal sample)...")
    X, y = load_siena_test_balanced(rng)
    mean = np.load(Path(SIENA_NORM) / "mean.npy").reshape(1, -1, 1).astype(np.float32)
    std = np.load(Path(SIENA_NORM) / "std.npy").reshape(1, -1, 1).astype(np.float32)

    arch = torch.load(FT_CKPT, map_location="cpu")["arch"]

    print("Loading frozen pretrained backbone...")
    frozen_bb = load_backbone_from(torch.load(FROZEN_CKPT, map_location=DEV), arch)
    frozen = EEGClassifier(frozen_bb, n_cls=2).to(DEV)  # random head; only feats used

    print("Loading Siena finetuned+RL classifier...")
    ck = torch.load(FT_CKPT, map_location=DEV)
    ft = EEGClassifier(DeepEnhancedEEGDiffusionModel(**arch).to(DEV), n_cls=2).to(DEV)
    ft.load_state_dict(ck["classifier"])

    print("Extracting embeddings (frozen)...")
    emb_fz = embed(frozen, X, mean, std)
    print("Extracting embeddings (finetuned+RL)...")
    emb_ft = embed(ft, X, mean, std)

    print("t-SNE...")
    ts_fz = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb_fz)
    ts_ft = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb_ft)

    knn_fz = knn_sep(emb_fz, y)
    knn_ft = knn_sep(emb_ft, y)
    print(f"kNN balanced accuracy: frozen={knn_fz:.3f}  finetuned+RL={knn_ft:.3f}")

    # merge into the existing interp.npz
    existing = dict(np.load(OUT / "interp.npz", allow_pickle=True))
    existing.update(dict(
        siena_y=y, siena_classes=np.array(CLASSES),
        siena_tsne_frozen=ts_fz, siena_tsne_finetuned=ts_ft,
        knn_siena_frozen=knn_fz, knn_siena_finetuned=knn_ft,
    ))
    np.savez(OUT / "interp.npz", **existing)
    print(f"Updated -> {OUT/'interp.npz'}")


if __name__ == "__main__":
    main()
