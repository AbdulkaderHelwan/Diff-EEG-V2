#!/usr/bin/env python3
"""Convert TUH EEG Events Corpus (TUEV) into THUSZ-style 22x1280 float32 batches.

Modeled on prepare_tuab.py's patient-wise approach: TUEV's own edf/{train,eval}
directories are ALREADY patient-disjoint (per AAREADME: "the evaluation set is
disjoint from the training set"), so we reuse that split directly rather than
building our own -- exactly like prepare_tuab.py does for TUAB.

6-class event labels (per AAREADME): spsw, gped, pled, eyem, artf, bckg.
Labels come from each session's .rec file: "channel,start_sec,end_sec,code"
where the channel index refers to a 22-way TCP BIPOLAR montage (FP1-F7, etc.)
-- NOT the monopolar channels we read from the EDF. We read the EDF's raw
MONOPOLAR/referential channels (EEG FP1-REF, EEG FP2-REF, ...) and reorder them
into our standard 22-channel frame (same order as TUAB/THUSZ), then assign each
5s window a SINGLE label by taking, across all bipolar-montage .rec entries
whose interval contains the window's center time, the highest-priority class
present (spsw > gped > pled > eyem > artf > bckg). Windows with no matching
annotation are skipped (not defaulted to background).

TUEV's native EDF sample rate is 250 Hz; resampled to 256 Hz / 1280 samples per
5s window to match the model's training spec, same as every other dataset here.
A dataset-specific mean/std is computed and saved (do NOT reuse THUSZ-V2 or any
other dataset's normalization stats -- raw-value scale varies across pipelines).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly

mne.set_log_level("WARNING")

TARGET_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
    "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "ROC",
]
LABEL_CODES = {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}
PRIORITY = ["spsw", "gped", "pled", "eyem", "artf", "bckg"]   # rarest/most-important first

DEFAULT_INPUT_ROOT = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUEV_v2.0.1/edf"
DEFAULT_OUTPUT_ROOT = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUEV_v2.0.1/processed_TUEV"


def canonical_channel(name):
    c = re.sub(r"[^A-Za-z0-9]", "", name.upper())
    return c.replace("EEG", "").replace("REF", "")


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
    return out, sorted(set(TARGET_CHANNELS) - set(present))


def resample_to_target(data, sfreq, target_sfreq, target_len):
    if abs(sfreq - target_sfreq) > 1e-6:
        up, down = int(target_sfreq), int(round(sfreq))
        data = resample_poly(data, up, down, axis=-1).astype(np.float32)
    if data.shape[-1] >= target_len:
        return data[..., :target_len]
    out = np.zeros((*data.shape[:-1], target_len), dtype=np.float32)
    out[..., :data.shape[-1]] = data
    return out


def parse_rec(rec_path):
    """Return list of (start_sec, end_sec, label_name) across all montage channels."""
    events = []
    for line in Path(rec_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 4:
            continue
        _ch, s0, s1, code = parts
        label = LABEL_CODES.get(int(float(code)))
        if label:
            events.append((float(s0), float(s1), label))
    return events


def merge_events(events, gap_tol=0.05):
    """Merge back-to-back/overlapping same-label intervals (across the 22
    montage channels) into consolidated (start, end, label) spans. TUEV's .rec
    entries are short (~1s) per-channel event fragments; a real annotated event
    is typically many contiguous fragments (often from multiple channels) at
    the same label, e.g. [0.4,1.4),[1.4,2.4),... all 'artf' -> one [0.4,6.4) span."""
    by_label = defaultdict(list)
    for s0, s1, lab in events:
        by_label[lab].append((s0, s1))
    spans = []
    for lab, ivs in by_label.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e + gap_tol:
                cur_e = max(cur_e, e)
            else:
                spans.append((cur_s, cur_e, lab))
                cur_s, cur_e = s, e
        spans.append((cur_s, cur_e, lab))
    return spans


def extract_event_windows(events, total_sec, window_sec):
    """TUEV's annotations are short, sparse EVENT markers (not continuous
    session-long labels like TUAB/THUSZ) -- a fixed 0,5,10,... grid rarely
    aligns with them, so tiling the whole recording throws almost everything
    away. Instead, extract windows driven by the events themselves: one
    `window_sec` window centered on each short span, or a non-overlapping
    tiling of `window_sec` steps across a long span. Returns (start_sec, label).
    Overlapping windows from different (lower-priority) spans at nearly the
    same time are de-duplicated, keeping the highest-priority label."""
    spans = merge_events(events, gap_tol=0.05)
    chosen = {}   # rounded start_sec -> (priority_rank, label)
    for s0, s1, lab in spans:
        rank = PRIORITY.index(lab)
        dur = s1 - s0
        if dur <= window_sec:
            starts = [max(0.0, min(s0 - (window_sec - dur) / 2, total_sec - window_sec))]
        else:
            starts = list(np.arange(s0, s1 - window_sec + 1e-6, window_sec))
        for start in starts:
            if start < 0 or start + window_sec > total_sec + 1e-6:
                continue
            key = round(start, 1)
            if key not in chosen or rank < chosen[key][0]:
                chosen[key] = (rank, lab)
    return [(start, lab) for start, (rank, lab) in sorted(chosen.items())]


class ChannelAccumulator:
    def __init__(self, n_channels):
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sum_sq = np.zeros(n_channels, dtype=np.float64)
        self.n = 0

    def update(self, segment):
        self.sum += segment.sum(axis=1)
        self.sum_sq += (segment ** 2).sum(axis=1)
        self.n += segment.shape[1]

    def finalize(self):
        if self.n == 0:
            z = np.zeros_like(self.sum, dtype=np.float32)
            return z, z
        mean = self.sum / self.n
        std = np.sqrt(np.maximum(self.sum_sq / self.n - mean ** 2, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32)


class BatchWriter:
    def __init__(self, out_root, batch_size):
        self.out_root = Path(out_root)
        self.batch_size = batch_size
        self.buffers = defaultdict(list)
        self.counts = defaultdict(int)

    def add(self, split, label, arr):
        key = (split, label)
        self.buffers[key].append(arr.astype(np.float32, copy=False))
        if len(self.buffers[key]) >= self.batch_size:
            self.flush(key)

    def flush(self, key):
        rows = self.buffers[key]
        if not rows:
            return
        split, label = key
        d = self.out_root / f"{split}-{label}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{label}_batch_{self.counts[key]:04d}.npy", np.stack(rows, axis=0))
        self.counts[key] += 1
        self.buffers[key] = []

    def close(self):
        for key in list(self.buffers):
            self.flush(key)


def process_session(edf_path, writer, accumulator, args, stats):
    rec_path = edf_path.with_suffix(".rec")
    if not rec_path.exists():
        stats["missing_rec"] += 1
        return
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        stats["read_errors"] += 1
        print(f"    WARN {edf_path.name}: {e}")
        return

    sfreq = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float32)
    total_sec = data.shape[1] / sfreq
    events = parse_rec(rec_path)
    event_windows = extract_event_windows(events, total_sec, args.window_sec)

    projected, missing = project_channels(data, raw.ch_names)
    if missing:
        stats["missing_channel_files"] += 1

    win_in = int(round(args.window_sec * sfreq))
    target_len = int(round(args.window_sec * args.target_sfreq))
    split = "eval" if "/eval/" in str(edf_path) else "train"

    for start_sec, label in event_windows:
        start = int(round(start_sec * sfreq))
        end = start + win_in
        if end > projected.shape[1]:
            continue
        chunk = resample_to_target(projected[:, start:end], sfreq, args.target_sfreq, target_len)
        writer.add(split, label, chunk)
        accumulator.update(chunk)
        stats[f"windows_{label}"] += 1
        stats["windows_total"] += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--stride-sec", type=float, default=5.0)
    p.add_argument("--target-sfreq", type=float, default=256.0)
    p.add_argument("--batch-size", type=int, default=2000)
    args = p.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    writer = BatchWriter(output_root, args.batch_size)
    accumulator = ChannelAccumulator(len(TARGET_CHANNELS))
    stats = defaultdict(int)

    for split in ["train", "eval"]:
        edf_files = sorted((input_root / split).glob("*/*.edf"))
        print(f"{split}: {len(edf_files)} sessions")
        for i, edf_path in enumerate(edf_files):
            process_session(edf_path, writer, accumulator, args, stats)
            if (i + 1) % 50 == 0:
                print(f"  {split}: {i+1}/{len(edf_files)} sessions | windows so far: {stats['windows_total']}")
    writer.close()

    mean, std = accumulator.finalize()
    norm_dir = output_root / "normalization"
    norm_dir.mkdir(parents=True, exist_ok=True)
    np.save(norm_dir / "mean.npy", mean)
    np.save(norm_dir / "std.npy", std)

    report = {
        "target_channels": TARGET_CHANNELS, "target_sfreq": args.target_sfreq,
        "window_sec": args.window_sec, "stride_sec": args.stride_sec,
        "classes": PRIORITY, "stats": dict(stats),
    }
    (output_root / "tuev_processing_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nTotal windows: {stats['windows_total']}")
    for c in PRIORITY:
        print(f"  {c}: {stats.get(f'windows_{c}', 0)}")
    print(f"Unlabeled (skipped): {stats['unlabeled_skipped']}")
    print(f"Missing .rec: {stats['missing_rec']}  Read errors: {stats['read_errors']}")
    print(f"\n✓ saved -> {output_root}/")


if __name__ == "__main__":
    main()
