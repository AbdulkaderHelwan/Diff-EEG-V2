#!/usr/bin/env python3
"""Zero-shot / parameter-free evaluation of the frozen DiffEEG embedding.

Uses a classifier with NO trained weights: a distance-weighted k-nearest-neighbor
classifier on the frozen 4800-d embedding, scored by 5-fold cross-validated
balanced accuracy. This is the SAME `knn_sep` protocol used for the t-SNE panels
(Fig. 5 / interp.npz), so the numbers are consistent across the paper.

Memory-bounded: raw windows are capped per class and freed immediately after
embedding. Reuses the exact backbone / normalization / embedding extraction of
eval_processed_multiclass.py.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_processed_multiclass import (
    DeepEnhancedEEGDiffusionModel, ARCH, DEFAULT_BACKBONE, DEFAULT_NORM,
    discover_classes, load_split_raw, normalize, embed_all,
)


def knn_sep(emb, y, k=10, seed=42):
    """Balanced-accuracy 5-fold CV kNN in embedding space (matches interp.py)."""
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, emb, y, cv=cv, scoring="balanced_accuracy").mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--backbone-ckpt", default=DEFAULT_BACKBONE)
    p.add_argument("--norm-stats-dir", default=DEFAULT_NORM)
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-per-class", type=int, default=2500,
                   help="cap raw windows loaded per class per split (memory bound)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default="/home/abdulh/scratch/EEGdiff_V2/Benchmarking")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = np.load(Path(args.norm_stats_dir) / "mean.npy").reshape(-1, 1).astype(np.float32)
    std = np.load(Path(args.norm_stats_dir) / "std.npy").reshape(-1, 1).astype(np.float32)
    classes = args.classes or discover_classes(args.processed_root, "train")
    print(f"[{args.dataset_name}] classes({len(classes)}): {classes}", flush=True)

    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device).eval()
    ckpt = torch.load(args.backbone_ckpt, map_location=device)
    backbone.load_state_dict(ckpt["model_state_dict"])

    # Pool train+test (capped), embed each split, free raw immediately.
    embs, ys = [], []
    for split in ("train", "test"):
        Xr, yr = load_split_raw(args.processed_root, split, classes,
                                max_per_class=args.max_per_class, seed=args.seed)
        E = embed_all(backbone, normalize(Xr, mean, std), device)
        del Xr
        embs.append(E)
        ys.append(yr)
        print(f"  {split}: embedded {E.shape}", flush=True)
    emb = np.concatenate(embs); y = np.concatenate(ys)
    del embs
    print(f"  pooled embedding {emb.shape}  (k={args.k}, 5-fold CV bal-acc)", flush=True)

    bal = knn_sep(emb, y, k=args.k, seed=args.seed)
    print(f"\n[{args.dataset_name}] zero-shot kNN (no training)  "
          f"balanced-acc {bal:.4f}   chance {1.0/len(classes):.4f}", flush=True)

    out = Path(args.out_root) / f"{args.dataset_name}_zeroshot_knn"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(
        dataset=args.dataset_name, classes=classes, k=args.k,
        n_pooled=int(len(y)), balanced_accuracy=float(bal),
        chance=1.0 / len(classes)), indent=2))
    print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
