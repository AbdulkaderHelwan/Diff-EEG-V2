# Data-Processing

Scripts that convert each public EEG dataset into the fixed input format the
DiffEEG model expects: **22 channels × 1280 samples @ 256 Hz (5-second windows)**,
with the 22 electrodes in one canonical order.

## The canonical 22-channel frame

Every dataset is projected onto this exact order (derived from the TUAB/THUSZ
training montage):

```
FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 A1 A2 FZ CZ PZ ROC
```

Channel order matters as much as channel names: a right-set/wrong-order montage
silently corrupts results. Electrodes with no match are **zero-filled**; nothing
is interpolated or invented.

## The universal recipe (`prepare_external_eeg.py`)

1. **Canonicalize** each electrode name (`"EEG T7-REF"` → `"T7"`).
2. **Alias-map** lab-specific names onto the frame (`T7→T3`, `P7→T5`, `TP9→A1`,
   the bipolar `Fpz-Cz→FZ` / `Pz-Oz→PZ`, …).
3. **Project** into the 22 slots, zero-filling any missing channel.
4. **Resample** to 256 Hz and slide a 1280-sample window.

## Scripts

| Script | Datasets | What it does |
|---|---|---|
| `prepare_external_eeg.py` | Bonn, EEGMMIDB, Sleep-EDFx | Generic EDF/txt → 22-channel windowed `.npy` batches. Houses `TARGET_CHANNELS`, `ALIASES`, `canonical_channel`, `project_channels`, `window_edf`. |
| `prepare_tuev.py` | TUEV | Same channel mapping + **event-driven windowing** (TUEV labels are 1-second per-channel markers, not continuous, so windows are centered on each event). |
| `merge_tuev_epileptiform.py` | TUEV | Post-hoc relabel: merges `spsw+gped+pled` → `epileptiform` (4-class task) on already-processed data. |
| `convert_siena_npz.py` | Siena | Re-packages Siena's pre-processed `.npz` files (already 22×1280 @256 Hz) into the standard folder layout, patient-wise split. Data stays in **volts** — normalize with Siena's *own* stats (see note in the script), never with THUSZ stats. |
| `download_public_eeg.py` | all external | Download commands / helpers for the public datasets. |

## Per-dataset channel coverage

| Dataset | Native channels | Real channels / 22 | Note |
|---|---:|---:|---|
| TUSZ / TUAB | 29 / 36 | 22 | exact-name match (home montage) |
| Bonn | 1 | 1 | single trace → FP1 |
| EEGMMIDB | 64 | 19 | no A1/A2/ROC in the cap |
| Siena | 22 (pre-processed) | 19 | A1/A2/ROC absent; values in volts |
| TUEV | 27 | 22 | per-session exceptions |
| Sleep-EDFx | 2 (bipolar) | 2 | Fpz-Cz→FZ, Pz-Oz→PZ (approximate) |

## Notes

- The scripts contain absolute cluster paths (`/scratch/...`) as defaults; pass
  the corresponding `--input`/`--output` arguments (or edit the defaults) to run
  elsewhere.
- Dependencies: `numpy`, `scipy`, `mne`, `pyedflib` (for TUAB/THUSZ-style EDFs).
