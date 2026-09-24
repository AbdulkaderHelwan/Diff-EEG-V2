#!/usr/bin/env python3
"""Shared loader for the DiffEEG vs. EEGDM benchmark on OUR datasets.

Both models see the SAME windows, labels and splits. Only the input *view* differs,
because the two architectures were trained on different representations:

    DiffEEG view : 22 monopolar channels, 1280 samples @ 256 Hz, channel-wise z-score
    EEGDM  view : 22 TCP bipolar channels, 1000 samples @ 200 Hz, volts * 1e4

Both transforms are mechanical and lossless in principle: every TCP bipolar pair is a
difference of two monopolar channels, and 256 -> 200 Hz is resample_poly(25/32).
`volts * 1e4` is EEGDM's own `scale: 1e4` (conf/preprocessing/*.yaml), which puts most
values in [-1, 1] as their paper specifies.

Datasets (both binary, both 22 x 1280 @ 256 Hz monopolar on disk):
    tuab5s  -- TUAB abnormality detection, the corpus finetune_tuab_rl.py uses
    tusz    -- TUSZ binary seizure detection, patient-disjoint train/dev/eval
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

D_TUAB = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/TUAB/processed_tuab")
D_TUSZ = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/"
              "segment_5_eeg_all/BinaryClassification_Data")
D_EXT = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed")
NORM_EXT = Path("/home/abdulh/scratch/benchmarking2/results/norm")
TUSZ_NORM = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/"
                 "segment_5_eeg_all/normalization")

DATASETS = {
    # name: (root, normalization dir, class_name -> (dirs per split), label)
    "tuab5s": {
        # stored in MICROVOLTS (raw p95 ~30), so EEGDM's uV/100 needs *1e-2.
        # (TUEV was stored in volts and needed *1e4 -- the unit is per corpus,
        # not per model, and getting it wrong feeds their backbone garbage.)
        "to_uv": 1.0,          # already microvolts
        "eegdm_scale": 1e-2,
        "norm": D_TUAB / "normalization",
        "classes": ["normal", "abnormal"],
        "splits": {
            "train": [(D_TUAB / "train-normal", 0), (D_TUAB / "train-seizure", 1)],
            "test":  [(D_TUAB / "test-normal", 0),  (D_TUAB / "test-seizure", 1)],
        },
    },
    # ---- external validation corpora (same 22 x 1280 @ 256 Hz layout) ----
    "bonn": {
        # Stored in microvolts (live channel |x| up to ~1900), so EEGDM's uV/100
        # needs *1e-2, as for TUAB.
        #
        # CAVEAT: Bonn is intrinsically SINGLE-CHANNEL data zero-padded into the
        # 22-channel layout -- only FP1 carries signal. The TCP bipolar montage
        # therefore collapses to two channels (FP1-F7 and FP1-F3), both of which
        # equal FP1 because F7 and F3 are zero. Monopolar models see 1 live channel,
        # bipolar models see 2 duplicates. Cross-model ordering on Bonn reflects that
        # asymmetry as much as representation quality, and with 400 train / 100 test
        # segments the confidence intervals are wide. Report it as caveated.
        "to_uv": 1.0,          # microvolts
        "eegdm_scale": 1e-2,
        "norm": NORM_EXT / "bonn",
        "classes": ["normal", "seizure"],
        "splits": {
            "train": [(D_EXT / "bonn/train-normal", 0), (D_EXT / "bonn/train-seizure", 1)],
            "test":  [(D_EXT / "bonn/test-normal", 0),  (D_EXT / "bonn/test-seizure", 1)],
        },
    },
    "siena": {
        # Stored in VOLTS (std ~1.6e-4 V ~ 160 uV), unlike TUAB/Bonn which are in
        # microvolts. volts -> uV is *1e6, then EEGDM's uV/100 gives a net *1e4.
        # Getting this wrong feeds their backbone values 1e6 off, which is exactly
        # the failure seen on TUEV.
        "to_uv": 1e6,          # VOLTS -> microvolts
        "eegdm_scale": 1e4,
        "norm": NORM_EXT / "siena",
        "classes": ["non-seizure", "seizure"],
        "splits": {
            "train": [(D_EXT / "siena/train-normal", 0), (D_EXT / "siena/train-seizure", 1)],
            "test":  [(D_EXT / "siena/test-normal", 0),  (D_EXT / "siena/test-seizure", 1)],
        },
    },
    "tusz": {
        "to_uv": 1.0,              # microvolts
        "eegdm_scale": 1e-2,       # microvolts, as above
        "norm": TUSZ_NORM,
        "classes": ["non-seizure", "seizure"],
        "splits": {
            # train and dev are pooled for fine-tuning, exactly as the DiffEEG
            # patient-wise pipeline does; eval is the held-out patient cohort.
            "train": [(D_TUSZ / "train-NS_converted", 0), (D_TUSZ / "train-seizure_converted", 1),
                      (D_TUSZ / "dev-non-seizure_converted", 0), (D_TUSZ / "dev-seizure_converted", 1)],
            "test":  [(D_TUSZ / "eval-non-seizure_converted", 0),
                      (D_TUSZ / "eval-seizure_converted", 1)],
        },
    },
}

TARGET_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
    "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "ROC",
]
CH_IDX = {c: i for i, c in enumerate(TARGET_CHANNELS)}

# EEGDM's ch_order (their conf/finetune/base.yaml): standard 22-way TCP bipolar montage
TCP_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("C4", "CZ"),
    ("T4", "C4"), ("A2", "T4"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

SRC_FS, DST_FS = 256, 200
DST_LEN = 1000

# BIOT's 16-channel montage (README): the same TCP pairs we already derive, in their
# 10-10 naming (T7=T3, T8=T4, P7=T5, P8=T6). These are indices into TCP_PAIRS.
BIOT16_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21]

# LaBraM uses MONOPOLAR referential channels in the same order we store them
# (dataset_maker/make_TUAB.py chOrder_standard). Its 22nd channel is T1, ours is ROC,
# so we feed the first 21, which match exactly. Names are renamed to the 10-10
# convention LaBraM's standard_1020 table uses.
LABRAM_N_CH = 21
LABRAM_RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
LABRAM_CH_NAMES = [LABRAM_RENAME.get(c, c) for c in TARGET_CHANNELS[:LABRAM_N_CH]]
LABRAM_PATCH = 200


def to_bipolar(x):
    out = np.empty(x.shape[:-2] + (len(TCP_PAIRS), x.shape[-1]), dtype=np.float32)
    for i, (a, b) in enumerate(TCP_PAIRS):
        out[..., i, :] = x[..., CH_IDX[a], :] - x[..., CH_IDX[b], :]
    return out


def to_200hz(x):
    y = resample_poly(x, up=DST_FS, down=SRC_FS, axis=-1).astype(np.float32)
    if y.shape[-1] > DST_LEN:
        y = y[..., :DST_LEN]
    elif y.shape[-1] < DST_LEN:
        pad = [(0, 0)] * y.ndim
        pad[-1] = (0, DST_LEN - y.shape[-1])
        y = np.pad(y, pad)
    return y


def build_index(dataset: str, split: str):
    """(file, row, label) triples -- memory-light; the corpora are large."""
    cfg = DATASETS[dataset]
    idx = []
    for d, lab in cfg["splits"][split]:
        files = sorted(f for f in glob.glob(str(d / "*_batch_*.npy")) if "_labels" not in f)
        for f in files:
            n = np.load(f, mmap_mode="r").shape[0]
            idx.extend((f, i, lab) for i in range(n))
    return idx


def materialise(index, dataset: str, view: str, cap=None, seed=0):
    """Load `index` (optionally a class-balanced subsample of it) into memory."""
    cfg = DATASETS[dataset]
    if cap is not None and cap < len(index):
        rng = np.random.default_rng(seed)
        labs = np.array([l for _, _, l in index])
        keep = []
        per = cap // len(cfg["classes"])
        for c in range(len(cfg["classes"])):
            ci = np.where(labs == c)[0]
            keep.extend(rng.choice(ci, size=min(per, len(ci)), replace=False))
        index = [index[i] for i in sorted(keep)]

    by_file = {}
    for pos, (f, r, l) in enumerate(index):
        by_file.setdefault(f, []).append((pos, r, l))

    n = len(index)
    X = np.empty((n, 22, 1280), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    for f, rows in by_file.items():
        arr = np.load(f, mmap_mode="r")
        for pos, r, l in rows:
            s = np.asarray(arr[r], dtype=np.float32)
            if s.ndim == 2 and s.shape[0] != 22 and s.shape[1] == 22:
                s = s.T
            X[pos] = s
            y[pos] = l

    # Corpora are stored in different physical units (TUAB/TUSZ/Bonn in microvolts,
    # Siena in volts). Any view whose published scaling assumes a unit must convert
    # first, or the backbone sees values orders of magnitude off -- LaBraM on Siena
    # without this is ~1e6 too small and reduces to numerical zero.
    to_uv = cfg.get("to_uv", 1.0)
    if view == "eegdm":
        # bipolar (linear, so scaling order is irrelevant), resample, then apply THEIR
        # scale so the pretrained backbone sees the range it was trained on: uV / 100.
        X = to_200hz(to_bipolar(X)) * (to_uv * 1e-2)
    elif view == "labram":
        # 21 monopolar channels @200Hz, scaled by uV/100 (their data_preprocess.py).
        X = to_200hz(X[:, :LABRAM_N_CH, :]) * (to_uv / 100.0)
    elif view == "cbramod":
        # CBraMod: the same 16 TCP bipolar channels at 200 Hz as BIOT, but scaled by
        # uV/100 (their datasets/tuab_dataset.py returns `data/100`), i.e. the same
        # unit assumption as LaBraM. The extractor reshapes to (B, 16, patches, 200).
        X = to_200hz(to_bipolar(X))[:, BIOT16_IDX, :] * (to_uv / 100.0)
    elif view == "biot":
        # BIOT: 16 TCP bipolar channels @200Hz, normalised per channel by the 95th
        # percentile of |x| (their utils.py TUABLoader). That normalisation is scale
        # invariant, so this view needs no unit conversion.
        X = to_200hz(to_bipolar(X))[:, BIOT16_IDX, :]
        X = X / (np.quantile(np.abs(X), 0.95, axis=-1, keepdims=True) + 1e-8)
    else:
        mean = np.load(cfg["norm"] / "mean.npy").reshape(1, -1, 1).astype(np.float32)
        std = np.load(cfg["norm"] / "std.npy").reshape(1, -1, 1).astype(np.float32)
        X = (X - mean) / (std + 1e-6)
    return X.astype(np.float32), y


if __name__ == "__main__":
    for ds in DATASETS:
        print(f"=== {ds} ===")
        for sp in ("train", "test"):
            idx = build_index(ds, sp)
            labs = np.array([l for _, _, l in idx])
            cnt = {DATASETS[ds]["classes"][c]: int((labs == c).sum())
                   for c in range(len(DATASETS[ds]["classes"]))}
            print(f"  {sp:<6} n={len(idx):>9,}  {cnt}")
        X, y = materialise(build_index(ds, "test")[:64], ds, "eegdm")
        Xd, _ = materialise(build_index(ds, "test")[:64], ds, "diffeeg")
        print(f"  eegdm view {X.shape}  p95|x|={np.percentile(np.abs(X),95):.3f}")
        print(f"  diffeeg view {Xd.shape}  p95|x|={np.percentile(np.abs(Xd),95):.3f}")
