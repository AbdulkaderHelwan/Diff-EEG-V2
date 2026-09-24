#!/usr/bin/env python3
"""Convert external EEG datasets to TUAB-style EEGdiff V2 numpy batches."""

import argparse
import csv
import gc
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

try:
    import mne
except ImportError:  # pragma: no cover - optional for non-EDF datasets
    mne = None

if mne is not None:
    mne.set_log_level("WARNING")

# Order matches the ACTUAL training channel order used for THUSZ/TUAB
# (see Dataset/TUAB/processed_tuab/tuab_processing_report.json -> channel_order:
# FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,A1,A2,FZ,CZ,PZ,ROC).
# The backbone's conv filters are channel-POSITION-specific, so this order must
# match exactly -- a same-SET-different-ORDER channel list silently corrupts
# results (no crash, no error) by feeding each electrode into the wrong filter.
TARGET_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
    "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ", "ROC",
]
ALIASES = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6", "TP9": "A1", "TP10": "A2",
    "FPZ": "FP1", "AF3": "FP1", "AF4": "FP2", "EOG1": "ROC", "ECG": "ROC",
    # Sleep-EDF's PSG recordings have only 2 EEG channels total, both BIPOLAR
    # derivations (not monopolar): "Fpz-Cz" and "Pz-Oz". Neither matches any
    # standard 10-20 channel, so without these aliases every channel gets
    # zero-filled -- silently producing all-zero data for the entire corpus
    # (found and fixed after a full run collapsed to predicting one constant
    # class). Best-effort single-electrode approximation of each bipolar pair.
    "FPZCZ": "FZ", "PZOZ": "PZ",
}
BINARY_POSITIVE = {"seizure", "event", "spsw", "gped", "pled", "abnormal"}
BINARY_NEGATIVE = {"normal", "background", "bckg", "wake", "sleep", "control"}


def canonical_channel(name):
    cleaned = re.sub(r"[^A-Za-z0-9]", "", name.upper())
    cleaned = cleaned.replace("EEG", "")
    return ALIASES.get(cleaned, cleaned)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_name(patient_id, test_fraction=0.2):
    digits = re.sub(r"\D", "", str(patient_id))
    bucket = int(digits or abs(hash(patient_id)) % 1000) % 10
    return "test" if bucket < int(test_fraction * 10) else "train"


def fit_length(x, target_len):
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    out = np.zeros((*x.shape[:-1], target_len), dtype=np.float32)
    out[..., :x.shape[-1]] = x
    return out


def resample_to_target(data, sfreq, target_sfreq, target_len):
    if abs(sfreq - target_sfreq) < 1e-6:
        return fit_length(data.astype(np.float32), target_len)
    up = int(target_sfreq)
    down = int(round(sfreq))
    out = resample_poly(data, up, down, axis=-1).astype(np.float32)
    return fit_length(out, target_len)


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


class BatchWriter:
    def __init__(self, out_root, batch_size):
        self.out_root = Path(out_root)
        self.batch_size = batch_size
        self.buffers = defaultdict(list)
        self.counts = defaultdict(int)
        self.files = []

    def add(self, split, label, arr):
        key = (split, label)
        self.buffers[key].append(arr.astype(np.float32, copy=False))
        if len(self.buffers[key]) >= self.batch_size:
            self.flush_key(key)

    def flush_key(self, key):
        split, label = key
        rows = self.buffers[key]
        if not rows:
            return
        out_dir = ensure_dir(self.out_root / f"{split}-{label}")
        idx = self.counts[key]
        path = out_dir / f"{label}_batch_{idx:04d}.npy"
        np.save(path, np.stack(rows, axis=0).astype(np.float32))
        self.files.append(str(path))
        self.counts[key] += 1
        self.buffers[key] = []

    def close(self):
        for key in list(self.buffers):
            self.flush_key(key)


def label_to_binary_folder(label):
    l = str(label).lower()
    if l in BINARY_POSITIVE:
        return "seizure"
    if l in BINARY_NEGATIVE:
        return "normal"
    return l


def folder_label(label, multiclass):
    """Folder name for a window's label.

    Binary mode (default) squashes to seizure/normal (for seizure/abnormal sets).
    Multiclass mode keeps the true class name (sanitized) so non-seizure tasks
    like motor imagery or sleep staging retain their real labels.
    """
    if multiclass:
        return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "unknown"
    return label_to_binary_folder(label)


def resample_full(data, sfreq, target_sfreq):
    """Resample a full (channels, time) array once. Unlike resample_to_target,
    does not pad/crop to a fixed window length -- preserves the whole duration."""
    if abs(sfreq - target_sfreq) < 1e-6:
        return data.astype(np.float32)
    up, down = int(target_sfreq), int(round(sfreq))
    return resample_poly(data, up, down, axis=-1).astype(np.float32)


def window_edf(path, writer, dataset, subject, label_spans=None, default_label="normal", args=None):
    # preload=True + resample ONCE for the whole session (not per-window): a
    # naive per-window resample_poly call (thousands of calls for a full-night
    # recording) is the dominant cost and unnecessary -- resampling is linear
    # and commutes with slicing, so resample once then window the result.
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    names = raw.ch_names
    split = split_name(subject, args.test_fraction)

    data = raw.get_data().astype(np.float32)
    projected, missing = project_channels(data, names)
    resampled = resample_full(projected, sfreq, args.target_sfreq)

    win_in = int(round(args.window_sec * args.target_sfreq))
    stride_in = int(round(args.stride_sec * args.target_sfreq))
    n_times_rs = resampled.shape[1]
    used = 0

    for start in range(0, max(0, n_times_rs - win_in + 1), stride_in):
        end = start + win_in
        center_sec = (start + end) / (2 * args.target_sfreq)
        label = default_label
        if label_spans:
            label = "normal"
            for s0, s1, span_label in label_spans:
                if s0 <= center_sec < s1:
                    label = span_label
                    break
        # .copy() is essential here: a bare view into `resampled` would keep the
        # WHOLE session's resampled array alive in memory for as long as this
        # window sits in BatchWriter's buffer (numpy views hold a ref to their
        # base, and BatchWriter's astype(copy=False) is a no-op on float32
        # input) -- across many files before a flush, this leaks gigabytes and
        # was the actual cause of an OOM kill.
        chunk = resampled[:, start:end].copy()
        writer.add(split, folder_label(label, args.multiclass), chunk)
        used += 1
    del raw, data, projected, resampled
    gc.collect()
    return {"file": str(path), "subject": subject, "sfreq": sfreq, "windows": used, "missing_channels": missing}


def parse_siena_seizures(txt_path):
    spans = []
    current_file = None
    for line in txt_path.read_text(errors="ignore").splitlines():
        m_file = re.search(r"(PN\d+-\d+\.edf)", line, re.I)
        if m_file:
            current_file = m_file.group(1)
        times = re.findall(r"(\d{1,2})[.:](\d{1,2})[.:](\d{1,2})", line)
        if current_file and len(times) >= 2 and "seiz" in line.lower():
            vals = []
            for h, m, s in times[:2]:
                vals.append(int(h) * 3600 + int(m) * 60 + int(s))
            if vals[1] > vals[0]:
                spans.append((current_file, vals[0], vals[1], "seizure"))
    return spans


def prepare_siena(raw_root, writer, args):
    by_file = defaultdict(list)
    for txt in Path(raw_root).rglob("Seizures-list-*.txt"):
        for fname, s0, s1, label in parse_siena_seizures(txt):
            by_file[fname].append((s0, s1, label))
    meta = []
    for edf in sorted(Path(raw_root).rglob("*.edf")):
        subject = edf.parent.name
        meta.append(window_edf(edf, writer, "siena", subject, by_file.get(edf.name), "normal", args))
    return meta


def sleep_stage_from_desc(desc):
    d = desc.lower()
    if "sleep stage w" in d:
        return "wake"
    if "sleep stage r" in d:
        return "rem"
    for n in ["1", "2", "3", "4"]:
        if f"sleep stage {n}" in d:
            return f"n{n}"
    return None


def prepare_sleep_edfx(raw_root, writer, args):
    meta = []
    psgs = sorted(Path(raw_root).rglob("*PSG.edf"))
    for psg in psgs:
        # PSG and Hypnogram differ in a suffix char (e.g. SC4001E0-PSG vs
        # SC4001EC-Hypnogram), so pair on the shared recording prefix, not an
        # exact name replace.
        hyp = next(psg.parent.glob(psg.name[:7] + "*Hypnogram.edf"), None)
        spans = []
        if hyp is not None:
            ann = mne.read_annotations(hyp)
            for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
                stage = sleep_stage_from_desc(desc)
                # Preserve the real stage (wake/n1/n2/n3/n4/rem) so --multiclass
                # gives a proper sleep-staging task; binary mode still collapses
                # via label_to_binary_folder's wake/sleep entries as before.
                if stage:
                    spans.append((float(onset), float(onset + duration), stage))
        subject = psg.name[:6]
        meta.append(window_edf(psg, writer, "sleep_edfx", subject, spans, "sleep", args))
    return meta


def prepare_bonn(raw_root, writer, args):
    meta = []
    zips = list(Path(raw_root).rglob("*.zip"))
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(Path(raw_root) / "_extracted" / zp.stem)
    # Case-insensitive .txt/.TXT (set N uses .TXT); skip macOS resource-fork junk.
    txt_files = sorted(p for p in Path(raw_root).rglob("*")
                       if p.suffix.lower() == ".txt"
                       and not p.name.startswith("._") and "__MACOSX" not in p.parts)
    for txt in txt_files:
        parent = txt.parent.name.upper()
        stem = txt.stem.upper()
        label = "seizure" if stem.startswith("S") or parent.startswith("S") else "normal"
        data = np.loadtxt(txt, dtype=np.float32)[None, :]
        data = resample_to_target(data, 173.61, args.target_sfreq, int(args.window_sec * args.target_sfreq))
        framed = np.zeros((len(TARGET_CHANNELS), data.shape[1]), dtype=np.float32)
        framed[0] = data[0]
        writer.add(split_name(stem, args.test_fraction), label, framed)
        meta.append({"file": str(txt), "subject": stem, "sfreq": 173.61, "windows": 1, "missing_channels": TARGET_CHANNELS[1:]})
    return meta


def prepare_mindbigdata(raw_root, writer, args):
    meta = []
    by_event = defaultdict(dict)
    for path in sorted(Path(raw_root).rglob("*.txt")):
        for line in path.open(errors="ignore"):
            parts = line.strip().split("\t")
            if len(parts) < 7:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
            try:
                event, device, channel, code, size = parts[1], parts[2], parts[3], int(parts[4]), int(parts[5])
                values = np.asarray([float(x) for x in parts[6].split(",") if x], dtype=np.float32)
            except Exception:
                continue
            by_event[(path.name, event, code, device)][canonical_channel(channel)] = values[:size]

    for (fname, event, code, device), chans in by_event.items():
        sfreq = 512 if device == "MW" else 220 if device == "MU" else 128
        max_len = max(len(v) for v in chans.values())
        data = np.zeros((len(chans), max_len), dtype=np.float32)
        names = []
        for i, (ch, vals) in enumerate(chans.items()):
            names.append(ch)
            data[i, :len(vals)] = vals
        projected, missing = project_channels(data, names)
        projected = resample_to_target(projected, sfreq, args.target_sfreq, int(args.window_sec * args.target_sfreq))
        label = "background" if code == -1 else "event"
        writer.add(split_name(event, args.test_fraction), label_to_binary_folder(label), projected)
        meta.append({"file": fname, "subject": event, "device": device, "digit": code, "sfreq": sfreq, "windows": 1, "missing_channels": missing})
    return meta


# BCI Competition IV-2a (BNCI 001-2014) uses 22 EEG channels in its native
# montage. Keep that 22-channel order directly so the exported windows fit a
# model expecting 22-channel 5s inputs without remapping onto the TUAB/TUH
# channel set.
BCI_IV_2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]


def prepare_bci_iv_2a(raw_root, writer, args):
    """4-class motor imagery from BNCI .mat files (left hand/right hand/feet/tongue)."""
    meta = []
    target_len = int(round(args.window_sec * args.target_sfreq))
    for mat in sorted(Path(raw_root).rglob("A0*.mat")):
        subject = mat.stem[:3]                      # A01..A09
        split = "test" if mat.stem.endswith("E") else "train"
        m = sio.loadmat(mat, struct_as_record=False, squeeze_me=True)
        runs = np.atleast_1d(m["data"])
        n_used = 0
        for run in runs:
            X = getattr(run, "X", None)             # (samples, 25)
            trial = np.atleast_1d(getattr(run, "trial", []))
            y = np.atleast_1d(getattr(run, "y", []))
            classes = np.atleast_1d(getattr(run, "classes", []))
            fs = float(getattr(run, "fs", 250.0))
            if X is None or trial.size == 0 or y.size == 0:
                continue
            win_in = int(round(args.window_sec * fs))
            for t_idx, lab in zip(trial, y):
                lab = int(lab)
                if lab < 1 or lab > len(classes):
                    continue
                seg = X[int(t_idx):int(t_idx) + win_in, :22].T  # (22 EEG, win)
                if seg.shape[1] < win_in // 2:
                    continue
                projected = seg.astype(np.float32)
                projected = resample_to_target(projected, fs, args.target_sfreq, target_len)
                writer.add(split, folder_label(str(classes[lab - 1]), args.multiclass), projected)
                n_used += 1
        meta.append({"file": str(mat), "subject": subject, "split": split, "windows": n_used})
    return meta


# EEGMMIDB motor task runs -> (label for annotation T1, label for T2).
# T0 is rest everywhere. Runs 4,8,12 = imagined L/R fist; 6,10,14 = imagined
# both fists / both feet. Real-movement runs are included as their own classes.
EEGMMIDB_RUN_LABELS = {
    3: ("left_fist", "right_fist"), 7: ("left_fist", "right_fist"), 11: ("left_fist", "right_fist"),
    4: ("imag_left_fist", "imag_right_fist"), 8: ("imag_left_fist", "imag_right_fist"), 12: ("imag_left_fist", "imag_right_fist"),
    5: ("both_fists", "both_feet"), 9: ("both_fists", "both_feet"), 13: ("both_fists", "both_feet"),
    6: ("imag_both_fists", "imag_both_feet"), 10: ("imag_both_fists", "imag_both_feet"), 14: ("imag_both_fists", "imag_both_feet"),
}


def prepare_eegmmidb(raw_root, writer, args):
    """Motor imagery/movement from PhysioNet EEGMMIDB EDFs (annotation-driven)."""
    meta = []
    for edf in sorted(Path(raw_root).rglob("*.edf")):
        m = re.search(r"S(\d+)R(\d+)\.edf$", edf.name, re.I)
        if not m:
            continue
        run = int(m.group(2))
        if run not in EEGMMIDB_RUN_LABELS:        # skip baseline runs R01/R02
            continue
        t1_lab, t2_lab = EEGMMIDB_RUN_LABELS[run]
        try:
            raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
            ann = raw.annotations
        except Exception:
            continue
        spans = []
        for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
            d = str(desc).upper().strip()
            lab = "rest" if d == "T0" else t1_lab if d == "T1" else t2_lab if d == "T2" else None
            if lab:
                spans.append((float(onset), float(onset + dur), lab))
        subject = f"S{m.group(1)}"
        meta.append(window_edf(edf, writer, "eegmmidb", subject, spans, "rest", args))
    return meta


def prepare_generic_edf(raw_root, writer, args):
    meta = []
    for edf in sorted(Path(raw_root).rglob("*.edf")):
        label = "seizure" if any(tok in edf.as_posix().lower() for tok in ["seiz", "abnormal", "event"]) else "normal"
        meta.append(window_edf(edf, writer, "generic_edf", edf.parent.name, None, label, args))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["siena", "sleep_edfx", "bonn", "mindbigdata", "bci_iv_2a", "eegmmidb", "tuev", "bern_barcelona", "generic_edf"])
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--stride-sec", type=float, default=5.0)
    # 256 Hz matches the model's actual training input: 5s window = 1280 samples
    # (THUSZ/TUAB were both processed at 256 Hz). Do not change unless you also
    # retrain -- the backbone's conv kernels are tuned to this sample rate.
    p.add_argument("--target-sfreq", type=float, default=256.0)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--multiclass", action="store_true",
                   help="keep true class names as folders (for non-seizure tasks) "
                        "instead of squashing to seizure/normal")
    args = p.parse_args()

    writer = BatchWriter(args.out_root, args.batch_size)
    if args.dataset == "siena":
        records = prepare_siena(args.raw_root, writer, args)
    elif args.dataset == "sleep_edfx":
        records = prepare_sleep_edfx(args.raw_root, writer, args)
    elif args.dataset == "bonn":
        records = prepare_bonn(args.raw_root, writer, args)
    elif args.dataset == "mindbigdata":
        records = prepare_mindbigdata(args.raw_root, writer, args)
    elif args.dataset == "bci_iv_2a":
        records = prepare_bci_iv_2a(args.raw_root, writer, args)
    elif args.dataset == "eegmmidb":
        records = prepare_eegmmidb(args.raw_root, writer, args)
    else:
        records = prepare_generic_edf(args.raw_root, writer, args)
    writer.close()

    target_channels = BCI_IV_2A_CHANNELS if args.dataset == "bci_iv_2a" else TARGET_CHANNELS
    meta = {
        "dataset": args.dataset,
        "target_channels": target_channels,
        "shape": [None, len(target_channels), int(args.window_sec * args.target_sfreq)],
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "target_sfreq": args.target_sfreq,
        "batch_size": args.batch_size,
        "files_written": writer.files,
        "records": records,
    }
    ensure_dir(Path(args.out_root))
    with open(Path(args.out_root) / "metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"Wrote {len(writer.files)} batch files under {args.out_root}")


if __name__ == "__main__":
    main()

