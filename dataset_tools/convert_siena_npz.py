#!/usr/bin/env python3
"""Convert the already-processed Siena dataset (per-recording .npz files) into
our standard folder-of-batches format so eval_processed_binary.py /
eval_processed_multiclass.py / finetune_processed_multiclass.py work unchanged.

Siena was processed by someone else (not via our pipeline). Its
siena_processing_report.json confirms channel order, 256 Hz, and 1280-sample
(5s) windows ALREADY match our model's training spec exactly -- no channel or
sample-rate fix needed here (unlike Bonn/EEGMMIDB).

IMPORTANT: Siena's raw signal amplitude is in VOLTS (std ~1e-4), while our
model's THUSZ-V2 normalization stats assume a completely different raw scale
(std ~200+, i.e. those pipelines never converted to volts). Applying THUSZ's
global mean/std to Siena's raw values would silently produce near-zero,
degenerate input. Batches are written RAW (unnormalized) here; use
`--norm-stats-dir .../Siena/processed_siena/normalization` (Siena's OWN
precomputed stats, already in the processed folder) when running eval/finetune
scripts on the output -- do NOT point at THUSZ-V2's normalization dir.

Output: <out_root>/{train,test}-{normal,seizure}/{label}_batch_XXXX.npy
Patient-wise, seeded, no-leakage train/test split (Siena has no official split).
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_RECORDINGS = "/scratch/linah03/EpilepticSeizureProject/Dataset/Siena/processed_siena/recordings"
DEFAULT_OUT = "/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/siena"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS)
    ap.add_argument("--out-root", default=DEFAULT_OUT)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rec_dir = Path(args.recordings_dir)
    files = sorted(rec_dir.glob("*.npz"))
    by_patient = defaultdict(list)
    for f in files:
        patient = f.stem.split("__")[0]
        by_patient[patient].append(f)

    patients = sorted(by_patient)
    rng = random.Random(args.seed)
    rng.shuffle(patients)
    n_test = max(1, round(args.test_frac * len(patients)))
    test_patients = set(patients[:n_test])
    train_patients = set(patients[n_test:])
    print(f"Patients: {len(patients)} total | train {len(train_patients)} | test {len(test_patients)}")
    print(f"  test patients: {sorted(test_patients)}")

    buffers = defaultdict(list)   # (split, label) -> list of windows
    counts = defaultdict(int)
    out_root = Path(args.out_root)

    def flush(key):
        split, label = key
        rows = buffers[key]
        if not rows:
            return
        d = out_root / f"{split}-{label}"
        d.mkdir(parents=True, exist_ok=True)
        idx = counts[key]
        np.save(d / f"{label}_batch_{idx:04d}.npy", np.stack(rows, axis=0).astype(np.float32))
        counts[key] += 1
        buffers[key] = []

    stats = defaultdict(int)
    for patient, recs in by_patient.items():
        split = "test" if patient in test_patients else "train"
        for f in recs:
            d = np.load(f)
            X, Y = d["signals"], d["labels"]
            for x, y in zip(X, Y):
                label = "seizure" if y == 1 else "normal"
                key = (split, label)
                buffers[key].append(x)
                stats[key] += 1
                if len(buffers[key]) >= args.batch_size:
                    flush(key)
    for key in list(buffers):
        flush(key)

    print("\nWindow counts:")
    for key in sorted(stats):
        print(f"  {key[0]}-{key[1]:8s}: {stats[key]:6d}")
    print(f"\n✓ Wrote batches to {out_root}/")
    print(f"  Use --norm-stats-dir /scratch/linah03/EpilepticSeizureProject/Dataset/Siena/processed_siena/normalization")
    print(f"  (Siena's OWN stats -- do NOT reuse THUSZ-V2 normalization, scales differ by ~1e6x)")


if __name__ == "__main__":
    main()
