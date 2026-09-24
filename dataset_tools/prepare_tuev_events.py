#!/usr/bin/env python3
"""Reprocess TUEV with the BIOT/LaBraM event protocol (one sample per .rec row).

Why this exists
---------------
`EEGdiff_V2/dataset_tools/prepare_tuev.py` merged the per-channel annotation rows
into spans and emitted ONE window per span, yielding 2,665 windows (2,621 used
in V2 fine-tuning). The
protocol every published TUEV number uses (BIOT's `BuildEvents`, inherited by
LaBraM, CBraMod and EEGDM) emits one sample PER ANNOTATION ROW:

    83,932 train + 29,421 eval = 113,353 samples

i.e. we were discarding ~43 of every 44 samples. That also invalidated the V2
conclusion that spsw is capped by data scarcity: the standard protocol has 567
spsw samples in eval, not 20.

Protocol implemented here
-------------------------
* Every TUEV annotation is exactly 1.0 s, one row per (channel, second):
      channel_index, start_sec, stop_sec, label_code
* One sample per row, window = [start - 2.0, stop + 2.0] = exactly 5.0 s,
  matching BIOT (2 s before, the 1 s event, 2 s after).
* All channels are kept in the window; the row's channel index is the
  "offending channel" and is stored as metadata rather than used to slice.
* Channel indices are 0..21 -> the 22-channel TCP AR montage.

Deviations from BIOT, deliberate and recorded
---------------------------------------------
1. Boundary events. BIOT concatenates the recording three times and indexes into
   the middle copy, so an event 0.4 s into a session gets 2 s of the END of that
   session prepended. About 8% of events start within 2 s of the recording, so
   this is not rare. We clamp the window inside the recording instead, keeping
   the signal contiguous and physiological, and set `clamped=1` on those samples
   so they can be excluded in an ablation.
2. Montage. BIOT feeds a 16-channel TCP bipolar view at 250 Hz. We project onto
   the same 22 monopolar TARGET_CHANNELS at 256 Hz / 1280 samples that every
   other corpus in our pretraining pool uses, so TUEV is poolable. The offending
   channel index still refers to the TCP montage and is stored verbatim.

Output layout matches the existing corpora: {train,test}-<class>/<class>_batch_NNNN.npy
plus a sidecar <class>_batch_NNNN_meta.npz carrying label, offending channel,
session id, event time and the clamped flag.

    python prepare_tuev_events.py --limit-sessions 2   # smoke test
    python prepare_tuev_events.py                      # full run
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly

mne.set_log_level("ERROR")

TARGET_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
    "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "ROC",
]
LABEL_CODES = {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}

# TUEV .rec channel indices index this montage (TUH TCP AR, 22 pairs).
TCP_MONTAGE = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-CZ", "C4-CZ", "T4-C4", "A2-T4", "FP1-F3", "F3-C3",
    "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

DEFAULT_INPUT_ROOT = "/scratch/linah03/EpilepticSeizureProject/Dataset/TUEV_v2.0.1/edf"
DEFAULT_OUTPUT_ROOT = "/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/processed_TUEV_events"

PRE_SEC, POST_SEC = 2.0, 2.0     # BIOT: [start - 2, stop + 2]


def canonical_channel(name: str) -> str:
    c = re.sub(r"[^A-Za-z0-9]", "", name.upper())
    return c.replace("EEG", "").replace("REF", "")


def project_channels(data: np.ndarray, names):
    """Map an EDF's channels onto TARGET_CHANNELS; absent channels stay zero."""
    out = np.zeros((len(TARGET_CHANNELS), data.shape[1]), dtype=np.float32)
    present = {}
    for i, raw_name in enumerate(names):
        c = canonical_channel(raw_name)
        if c in TARGET_CHANNELS and c not in present:
            present[c] = i
    for j, target in enumerate(TARGET_CHANNELS):
        if target in present:
            out[j] = data[present[target]]
    return out, sorted(set(TARGET_CHANNELS) - set(present))


def resample_to_target(data, sfreq, target_sfreq, target_len):
    if abs(sfreq - target_sfreq) > 1e-6:
        g = np.gcd(int(round(sfreq)), int(round(target_sfreq)))
        up, down = int(round(target_sfreq)) // g, int(round(sfreq)) // g
        data = resample_poly(data, up, down, axis=-1).astype(np.float32)
    if data.shape[-1] < target_len:
        data = np.pad(data, ((0, 0), (0, target_len - data.shape[-1])))
    return data[:, :target_len].astype(np.float32)


def parse_rec(rec_path: Path):
    """One tuple per annotation ROW -- no merging, no dedup.

    Returns (channel_index, start_sec, stop_sec, label_name).
    """
    rows = []
    for line in rec_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 4:
            continue
        ch, s0, s1, code = parts
        label = LABEL_CODES.get(int(float(code)))
        if label:
            rows.append((int(float(ch)), float(s0), float(s1), label))
    return rows


class BatchWriter:
    """Accumulates per-class samples and flushes fixed-size .npy batches + meta."""

    def __init__(self, out_root: Path, batch_size: int):
        self.out_root = out_root
        self.batch_size = batch_size
        self.buf = defaultdict(list)     # (split, label) -> [array]
        self.meta = defaultdict(list)    # (split, label) -> [dict]
        self.count = defaultdict(int)

    def add(self, split, label, arr, meta):
        key = (split, label)
        self.buf[key].append(arr)
        self.meta[key].append(meta)
        if len(self.buf[key]) >= self.batch_size:
            self.flush(key)

    def flush(self, key):
        if not self.buf[key]:
            return
        split, label = key
        d = self.out_root / f"{split}-{label}"
        d.mkdir(parents=True, exist_ok=True)
        idx = self.count[key]
        np.save(d / f"{label}_batch_{idx:04d}.npy", np.stack(self.buf[key]))
        m = self.meta[key]
        np.savez(
            d / f"{label}_batch_{idx:04d}_meta.npz",
            offending_channel=np.array([x["offending_channel"] for x in m], dtype=np.int16),
            event_start=np.array([x["event_start"] for x in m], dtype=np.float32),
            clamped=np.array([x["clamped"] for x in m], dtype=np.int8),
            session=np.array([x["session"] for x in m]),
        )
        self.count[key] += 1
        self.buf[key].clear()
        self.meta[key].clear()

    def close(self):
        for key in list(self.buf):
            self.flush(key)


def process_session(edf_path: Path, writer: BatchWriter, args, stats):
    rec_path = edf_path.with_suffix(".rec")
    if not rec_path.exists():
        stats["missing_rec"] += 1
        return
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:                                   # noqa: BLE001
        stats["read_errors"] += 1
        print(f"    WARN {edf_path.name}: {e}")
        return

    sfreq = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float32)
    total_sec = data.shape[1] / sfreq
    projected, missing = project_channels(data, raw.ch_names)
    if missing:
        stats["sessions_missing_channels"] += 1

    window_sec = PRE_SEC + 1.0 + POST_SEC
    win_in = int(round(window_sec * sfreq))
    target_len = int(round(window_sec * args.target_sfreq))
    split = "test" if "eval" in edf_path.parts else "train"
    session = edf_path.stem

    for ch_idx, s0, s1, label in parse_rec(rec_path):
        start_sec = s0 - PRE_SEC
        clamped = 0
        if start_sec < 0:                      # event near the recording start
            start_sec, clamped = 0.0, 1
        if start_sec + window_sec > total_sec:  # ... or near the end
            start_sec, clamped = max(0.0, total_sec - window_sec), 1
        start = int(round(start_sec * sfreq))
        end = start + win_in
        if end > projected.shape[1]:
            stats["too_short"] += 1
            continue
        chunk = resample_to_target(projected[:, start:end], sfreq, args.target_sfreq, target_len)
        writer.add(split, label, chunk,
                   {"offending_channel": ch_idx, "event_start": s0,
                    "clamped": clamped, "session": session})
        stats[f"{split}_{label}"] += 1
        stats["total"] += 1
        stats["clamped"] += clamped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--target-sfreq", type=float, default=256.0)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--limit-sessions", type=int, default=None,
                   help="process at most N sessions per split (smoke test)")
    args = p.parse_args()

    in_root, out_root = Path(args.input_root), Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    writer = BatchWriter(out_root, args.batch_size)
    stats = defaultdict(int)

    for split_dir in ("train", "eval"):
        edfs = sorted((in_root / split_dir).rglob("*.edf"))
        if args.limit_sessions:
            edfs = edfs[: args.limit_sessions]
        print(f"{split_dir}: {len(edfs)} sessions", flush=True)
        for i, edf in enumerate(edfs, 1):
            process_session(edf, writer, args, stats)
            if i % 25 == 0:
                print(f"  {split_dir} {i}/{len(edfs)}  samples so far: {stats['total']:,}", flush=True)
    writer.close()

    report = {
        "protocol": "BIOT/LaBraM: one sample per .rec row, window [start-2s, stop+2s] = 5s",
        "target_channels": TARGET_CHANNELS,
        "tcp_montage_for_offending_channel": TCP_MONTAGE,
        "target_sfreq": args.target_sfreq,
        "window_sec": PRE_SEC + 1.0 + POST_SEC,
        "boundary_handling": "clamped inside recording (BIOT wraps the recording 3x)",
        "counts": dict(sorted(stats.items())),
    }
    (out_root / "tuev_events_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["counts"], indent=2))
    print(f"\nwrote -> {out_root}")


if __name__ == "__main__":
    main()
