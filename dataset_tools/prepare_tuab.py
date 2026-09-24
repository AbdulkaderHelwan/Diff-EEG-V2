#!/usr/bin/env python3
"""Convert the TUH Abnormal (TUAB) corpus into 22xN float32 windows, with a
CONFIGURABLE window length (for the 5 s vs 10 s window-size ablation).

TUAB layout (TUH standard, recording-level labels):
    edf/train/{normal,abnormal}/01_tcp_ar/<patient>_..._t000.edf
    edf/eval/{normal,abnormal}/01_tcp_ar/...
The train/eval split is patient-disjoint by construction, so we use it directly
(train -> train, eval -> test). Every window of an abnormal recording is
abnormal (labels are per-recording).

Channels are the TCP referential montage, named 'EEG FP1-REF', 'EEG T3-REF', ...
We strip the 'EEG ' prefix and the '-REF'/'-LE' reference suffix, then map by
name onto the canonical 22-slot 10-20 frame (never averaged -- position-specific
filters). Source is 250 Hz, resampled to 256 Hz.

Writes <out>/{train,test}-{normal,abnormal}/*.npy plus
<out>/normalization/{mean,std}.npy (TUAB's OWN train-split stats).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

import mne
mne.set_log_level("ERROR")

TARGET_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
    "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "ROC",
]
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
LABELS = ["normal", "abnormal"]


def canonical_channel(name):
    """'EEG FP1-REF' -> 'FP1'. Strips the 'EEG' prefix and a '-REF'/'-LE'
    reference suffix before name matching."""
    c = name.upper().replace("EEG", "")
    c = re.split(r"-\s*(REF|LE)\b", c)[0]      # 'FP1-REF' -> 'FP1'
    c = re.sub(r"[^A-Z0-9]", "", c)
    return ALIASES.get(c, c)


def project_channels(data, names):
    out = np.zeros((len(TARGET_CHANNELS), data.shape[1]), dtype=np.float32)
    present = {}
    for i, name in enumerate(names):
        c = canonical_channel(name)
        if c in TARGET_CHANNELS and c not in present:
            present[c] = i
    for j, target in enumerate(TARGET_CHANNELS):
        if target in present:
            out[j] = data[present[target]]
    return out, sorted(set(TARGET_CHANNELS) - set(present)), present


def resample_full(data, sfreq, target_sfreq):
    if abs(sfreq - target_sfreq) < 1e-6:
        return data.astype(np.float32)
    up, down = int(target_sfreq), int(round(sfreq))
    g = np.gcd(up, down)
    return resample_poly(data, up // g, down // g, axis=-1).astype(np.float32)


class BatchWriter:
    def __init__(self, out_root, batch_size):
        self.out_root = Path(out_root)
        self.batch_size = batch_size
        self.buffers = defaultdict(list)
        self.counts = defaultdict(int)
        self.total = defaultdict(int)

    def add(self, split, label, arr):
        key = (split, label)
        self.buffers[key].append(arr)
        self.total[key] += 1
        if len(self.buffers[key]) >= self.batch_size:
            self.flush(key)

    def flush(self, key):
        rows = self.buffers[key]
        if not rows:
            return
        split, label = key
        d = self.out_root / f"{split}-{label}"
        d.mkdir(parents=True, exist_ok=True)
        idx = self.counts[key]
        np.save(d / f"{label}_batch_{idx:04d}.npy", np.stack(rows).astype(np.float32))
        self.counts[key] += 1
        self.buffers[key] = []

    def flush_all(self):
        for key in list(self.buffers):
            self.flush(key)


class StatsAccumulator:
    def __init__(self, n):
        self.sum = np.zeros(n, dtype=np.float64)
        self.sum_sq = np.zeros(n, dtype=np.float64)
        self.n = 0

    def update(self, seg):
        self.sum += seg.sum(axis=1)
        self.sum_sq += (seg.astype(np.float64) ** 2).sum(axis=1)
        self.n += seg.shape[1]

    def finalize(self):
        if self.n == 0:
            z = np.zeros_like(self.sum, dtype=np.float32)
            return z, np.ones_like(z)
        mean = self.sum / self.n
        var = np.maximum(self.sum_sq / self.n - mean ** 2, 1e-12)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root",
                    default="/scratch/linah03/EpilepticSeizureProject/Dataset/TUAB/edf")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--target-sfreq", type=float, default=256.0)
    ap.add_argument("--window-sec", type=float, default=5.0)
    ap.add_argument("--stride-sec", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--max-files-per-class", type=int, default=None,
                    help="optional cap on #recordings per (split,label) for speed")
    args = ap.parse_args()

    root = Path(args.input_root)
    out_root = Path(args.out_root)
    writer = BatchWriter(out_root, args.batch_size)
    acc = StatsAccumulator(len(TARGET_CHANNELS))
    win = int(round(args.window_sec * args.target_sfreq))
    stride = int(round(args.stride_sec * args.target_sfreq))
    stats = defaultdict(int)
    records = []

    for folder, split in [("train", "train"), ("eval", "test")]:
        for label in LABELS:
            edfs = sorted((root / folder / label).rglob("*.edf"))
            if args.max_files_per_class:
                edfs = edfs[:args.max_files_per_class]
            print(f"{folder}/{label}: {len(edfs)} recordings -> split '{split}'", flush=True)
            for k, edf in enumerate(edfs):
                try:
                    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
                except Exception as e:
                    print(f"  WARN {edf.name}: {e}"); stats["read_errors"] += 1; continue
                sfreq = float(raw.info["sfreq"])
                data = raw.get_data().astype(np.float32)
                names = raw.ch_names
                del raw
                projected, missing, present = project_channels(data, names)
                del data
                resampled = resample_full(projected, sfreq, args.target_sfreq)
                del projected
                n_used = 0
                for start in range(0, resampled.shape[1] - win + 1, stride):
                    chunk = resampled[:, start:start + win].copy()
                    if chunk.shape[1] != win:
                        continue
                    writer.add(split, label, chunk)
                    if split == "train":
                        acc.update(chunk)
                    stats[f"windows_{split}_{label}"] += 1
                    n_used += 1
                del resampled
                records.append(dict(file=edf.name, split=split, label=label,
                                    sfreq=sfreq, windows=n_used, missing=missing,
                                    n_present=len(present)))
                if (k + 1) % 200 == 0:
                    print(f"  {folder}/{label}: {k+1}/{len(edfs)} done "
                          f"({len(present)}/22 ch, missing {missing})", flush=True)

    writer.flush_all()
    mean, std = acc.finalize()
    norm_dir = out_root / "normalization"
    norm_dir.mkdir(parents=True, exist_ok=True)
    np.save(norm_dir / "mean.npy", mean)
    np.save(norm_dir / "std.npy", std)

    print("\nWindow counts:")
    for key in sorted(writer.total):
        print(f"  {key[0]}-{key[1]:9s}: {writer.total[key]:7d}")
    print(f"Normalization -> {norm_dir}  mean[:3]={np.round(mean[:3],7)} std[:3]={np.round(std[:3],7)}")

    report = dict(target_channels=TARGET_CHANNELS, aliases=ALIASES,
                  window_sec=args.window_sec, stride_sec=args.stride_sec,
                  target_sfreq=args.target_sfreq, labels=LABELS,
                  counts={f"{k[0]}-{k[1]}": int(v) for k, v in writer.total.items()},
                  stats={k: int(v) for k, v in stats.items()},
                  records=records)
    (out_root / "tuab_processing_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved -> {out_root}/")


if __name__ == "__main__":
    main()
