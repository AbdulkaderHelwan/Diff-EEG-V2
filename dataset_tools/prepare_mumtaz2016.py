#!/usr/bin/env python3
"""Convert the Mumtaz2016 MDD/healthy EEG dataset into 22x1280 float32 batches.

Mumtaz2016 ("MDD Patients and Healthy Controls EEG Data (New)", figshare
4244171): 64 subjects (30 healthy 'H', 34 depressed 'MDD'), each recorded in
three conditions -- EC (eyes closed), EO (eyes open), TASK (P300). Files are
EDF named '<H|MDD> S<subj> <EC|EO|TASK>.edf', already at 256 Hz, 19 EEG
electrodes in the 10-20 system referenced to linked ears ('EEG <name>-LE'),
plus a bipolar reference 'A2-A1' and (in the H files) two aux channels.

  Labels: healthy vs mdd  (binary MDD-detection)

Channel handling: the 19 scalp electrodes map by NAME onto our canonical
22-slot 10-20 frame (never averaged/interpolated -- filters are position
specific). The 'EEG ' prefix and '-LE' reference suffix are stripped before
matching; the bipolar reference ('A2-A1') and aux channels ('23A-23R',
'24A-24R') match nothing and are dropped. 19/22 slots fill; A1, A2 (no separate
scalp electrodes -- only the bipolar) and ROC are zero-filled.

No resampling is needed (source is already 256 Hz).

Split is SUBJECT-WISE and stratified by group: a fraction of the H subjects and
of the MDD subjects are held out for test, so no subject (and none of that
subject's EC/EO/TASK recordings) appears in both train and test. Note H S1 and
MDD S1 are different people, so the split key is (group, subject).

Writes <out>/{train,test}-<label>/<label>_batch_XXXX.npy plus
<out>/normalization/{mean,std}.npy (Mumtaz's OWN stats -- do not reuse THUSZ's).
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

import mne
mne.set_log_level("ERROR")

# Canonical 22-channel frame (same order as TUAB/THUSZ -- must match exactly).
TARGET_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
    "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "ROC",
]
ALIASES = {}  # Mumtaz already uses classic 10-20 names; no aliasing needed.

LABELS = {"H": "healthy", "MDD": "mdd"}


def canonical_channel(name):
    """'EEG Fp1-LE' -> 'FP1'. Strips the 'EEG' prefix and the '-LE' linked-ear
    reference suffix, then keeps alphanumerics. Bipolar/aux channels such as
    'A2-A1' or '23A-23R' have no '-LE' suffix and canonicalize to non-frame
    tokens ('A2A1', '23A23R'), so they are dropped -- the bipolar reference is
    never mistaken for the A2 scalp slot."""
    c = name.upper().replace("EEG", "").strip()
    c = re.split(r"-LE\b", c)[0].strip()      # 'FP1-LE' -> 'FP1'; 'A2-A1' unchanged
    c = re.sub(r"[^A-Z0-9]", "", c)
    return ALIASES.get(c, c)


def project_channels(data, names):
    """Map (n_src_channels, T) onto the 22-slot frame by NAME. First match wins."""
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
    """Streaming per-channel mean/std over the TRAIN split only."""
    def __init__(self, n_channels):
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sum_sq = np.zeros(n_channels, dtype=np.float64)
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


FNAME_RE = re.compile(r"^(H|MDD)\s+S(\d+)\s+(EC|EO|TASK)\.edf$", re.IGNORECASE)


def scan_files(eeg_dir, conditions):
    """Return list of (group, subj_num, condition, path), filtered to the
    requested conditions."""
    recs = []
    for f in sorted(Path(eeg_dir).glob("*.edf")):
        m = FNAME_RE.match(f.name)
        if not m:
            print(f"  skip unrecognized name: {f.name}")
            continue
        group, subj, cond = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        if cond not in conditions:
            continue
        recs.append((group, subj, cond, f))
    return recs


def choose_test_subjects(recs, test_frac, seed):
    """Stratified subject-wise: hold out the last ceil(test_frac*n) subjects of
    each group (deterministic given the sorted subject numbers)."""
    by_group = defaultdict(set)
    for group, subj, _, _ in recs:
        by_group[group].add(subj)
    test = set()
    for group, subjs in by_group.items():
        s = sorted(subjs)
        k = max(1, math.ceil(test_frac * len(s)))
        for subj in s[-k:]:
            test.add((group, subj))
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root",
                    default="/scratch/linah03/EpilepticSeizureProject/Dataset/external_raw/mumtaz2016")
    ap.add_argument("--out-root",
                    default="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/mumtaz2016")
    ap.add_argument("--target-sfreq", type=float, default=256.0)
    ap.add_argument("--window-sec", type=float, default=5.0)
    ap.add_argument("--stride-sec", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--conditions", nargs="+", default=["EC", "EO", "TASK"],
                    choices=["EC", "EO", "TASK"],
                    help="which recording conditions to include as windows")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of subjects per group held out for test")
    ap.add_argument("--seed", type=int, default=42)
    # Optional preprocessing to match the NeurIPT/CBraMod Mumtaz2016 pipeline
    # (notch + band-pass + global-average reference). Off by default so the
    # original run is reproduced exactly.
    ap.add_argument("--notch-freq", type=float, default=None,
                    help="apply a notch filter at this frequency, e.g. 50")
    ap.add_argument("--l-freq", type=float, default=None,
                    help="band-pass high-pass edge, e.g. 0.1")
    ap.add_argument("--h-freq", type=float, default=None,
                    help="band-pass low-pass edge, e.g. 30")
    ap.add_argument("--reref-average", action="store_true",
                    help="common-average reference over the present scalp electrodes")
    args = ap.parse_args()

    eeg_dir = Path(args.input_root)
    out_root = Path(args.out_root)
    writer = BatchWriter(out_root, args.batch_size)
    acc = StatsAccumulator(len(TARGET_CHANNELS))

    recs = scan_files(eeg_dir, set(c.upper() for c in args.conditions))
    n_subj = len({(g, s) for g, s, _, _ in recs})
    print(f"{len(recs)} recordings from {n_subj} subjects; conditions {args.conditions}")
    test_subjects = choose_test_subjects(recs, args.test_frac, args.seed)
    print(f"test subjects ({len(test_subjects)}): "
          f"{sorted((g, s) for g, s in test_subjects)}")

    win = int(round(args.window_sec * args.target_sfreq))
    stride = int(round(args.stride_sec * args.target_sfreq))
    stats = defaultdict(int)
    records = []

    for group, subj, cond, path in recs:
        split = "test" if (group, subj) in test_subjects else "train"
        label = LABELS[group]
        try:
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        except Exception as e:
            print(f"  WARN {path.name}: {e}")
            stats["read_errors"] += 1
            continue
        sfreq = float(raw.info["sfreq"])
        # Optional filtering, applied on the continuous recording before windowing.
        if args.notch_freq:
            raw.notch_filter(args.notch_freq, verbose=False)
        if args.l_freq is not None or args.h_freq is not None:
            raw.filter(args.l_freq, args.h_freq, verbose=False)
        data = raw.get_data().astype(np.float32)
        names = raw.ch_names
        del raw

        projected, missing, present = project_channels(data, names)
        del data
        if missing:
            stats["files_with_missing_channels"] += 1
        # Common-average reference over the electrodes actually present (a global
        # average reference computed on the real scalp channels, never the
        # zero-filled slots).
        if args.reref_average and present:
            idx = [j for j, t in enumerate(TARGET_CHANNELS) if t in present]
            projected[idx] -= projected[idx].mean(axis=0, keepdims=True)
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
            stats[f"windows_{label}"] += 1
            stats["windows_total"] += 1
            n_used += 1
        del resampled
        records.append(dict(group=group, subject=subj, condition=cond, file=path.name,
                            split=split, sfreq=sfreq, windows=n_used,
                            missing_channels=missing,
                            mapped_channels={k: names[v] for k, v in present.items()}))
        print(f"  {group:3s} S{subj:02d} {cond:4s} [{split:5s}] -> {n_used:4d} windows "
              f"({len(present)}/22 channels, missing {missing})", flush=True)

    writer.flush_all()

    mean, std = acc.finalize()
    norm_dir = out_root / "normalization"
    norm_dir.mkdir(parents=True, exist_ok=True)
    np.save(norm_dir / "mean.npy", mean)
    np.save(norm_dir / "std.npy", std)

    print("\nWindow counts:")
    for key in sorted(writer.total):
        print(f"  {key[0]}-{key[1]:8s}: {writer.total[key]:6d}")
    print(f"\nTOTAL windows: {stats['windows_total']}")
    print(f"Normalization (Mumtaz's OWN) -> {norm_dir}")
    print(f"  mean[:4] {np.round(mean[:4], 8)}")
    print(f"  std[:4]  {np.round(std[:4], 8)}")

    report = dict(target_channels=TARGET_CHANNELS, aliases=ALIASES,
                  target_sfreq=args.target_sfreq, window_sec=args.window_sec,
                  stride_sec=args.stride_sec, labels=LABELS,
                  preprocessing=dict(notch_freq=args.notch_freq, l_freq=args.l_freq,
                                     h_freq=args.h_freq, reref_average=args.reref_average),
                  conditions=args.conditions, test_frac=args.test_frac,
                  test_subjects=sorted(f"{g} S{s}" for g, s in test_subjects),
                  stats={k: int(v) for k, v in stats.items()},
                  counts={f"{k[0]}-{k[1]}": int(v) for k, v in writer.total.items()},
                  records=records)
    (out_root / "mumtaz2016_processing_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out_root}/")


if __name__ == "__main__":
    main()
