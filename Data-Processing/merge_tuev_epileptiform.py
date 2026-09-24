#!/usr/bin/env python3
"""Merge TUEV's spsw+gped+pled into one 'epileptiform' class.

Follow-up to the 6-class TUEV experiments: spsw/gped/pled are clinically the
same family (epileptiform discharges) and turned out to be genuinely
inseparable at our embedding's temporal resolution -- the model's errors just
moved between these three rather than resolving (see PROJECT_LOG 5.6/5.7).
This produces a cleaner 4-class task (artf/bckg/eyem/epileptiform) from the
ALREADY-PROCESSED 6-class data -- no need to re-read raw EDFs.

Reuses the same normalization stats (computed globally across all windows,
still valid for any relabeling of the same underlying data).
"""
import argparse
from pathlib import Path

import numpy as np

DEFAULT_SRC = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUEV_v2.0.1/processed_TUEV"
DEFAULT_OUT = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUEV_v2.0.1/processed_TUEV_4class"
MERGE_INTO = {"epileptiform": ["spsw", "gped", "pled"]}
KEEP_AS_IS = ["artf", "bckg", "eyem"]


def copy_class(src_root, out_root, split, cname):
    src_dir = Path(src_root) / f"{split}-{cname}"
    out_dir = Path(out_root) / f"{split}-{cname}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob("*_batch_*.npy")):
        arr = np.load(f)
        np.save(out_dir / f.name, arr)
        n += len(arr)
    return n


def merge_classes(src_root, out_root, split, new_name, old_names, batch_size=2000):
    out_dir = Path(out_root) / f"{split}-{new_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = []
    idx = 0
    n_total = 0
    for cname in old_names:
        src_dir = Path(src_root) / f"{split}-{cname}"
        for f in sorted(src_dir.glob("*_batch_*.npy")):
            buf.append(np.load(f))
    if buf:
        full = np.concatenate(buf, axis=0)
        n_total = len(full)
        for i in range(0, n_total, batch_size):
            chunk = full[i:i + batch_size]
            np.save(out_dir / f"{new_name}_batch_{idx:04d}.npy", chunk)
            idx += 1
    return n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default=DEFAULT_SRC)
    ap.add_argument("--out-root", default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"Source: {args.src_root}\nOutput: {args.out_root}\n")
    for split in ["train", "test"]:
        for cname in KEEP_AS_IS:
            n = copy_class(args.src_root, args.out_root, split, cname)
            print(f"  {split}-{cname:14s} copied {n:5d} windows")
        for new_name, old_names in MERGE_INTO.items():
            n = merge_classes(args.src_root, args.out_root, split, new_name, old_names)
            print(f"  {split}-{new_name:14s} merged  {n:5d} windows (from {'+'.join(old_names)})")

    # reuse the same global normalization stats (still valid -- same underlying windows)
    src_norm = Path(args.src_root) / "normalization"
    out_norm = Path(args.out_root) / "normalization"
    out_norm.mkdir(parents=True, exist_ok=True)
    for fname in ["mean.npy", "std.npy"]:
        np.save(out_norm / fname, np.load(src_norm / fname))

    print(f"\n✓ saved -> {args.out_root}/")


if __name__ == "__main__":
    main()
