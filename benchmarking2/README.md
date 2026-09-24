# Benchmarking against EEG foundation models

Controlled comparison of DiffEEG with five published EEG foundation models: EEGDM, BIOT,
LaBraM, CBraMod and EEGMamba. Every model is run by us on the same windows, labels and
splits, each receiving the input view its authors specify, so only the representation
differs between rows.

Two protocols:

- **Frozen linear probe** (`code/extract_features.py`, `code/probe.py`): backbone frozen,
  logistic regression on the pooled representation, regularisation chosen on validation,
  95% CIs from 1,000 bootstrap resamples of the test set.
- **Task-trained baselines** (`code/supervised_baselines.py`): EEGNet, EEG-Conformer and a
  spatio-temporal transformer trained from scratch on TUH Seizure, beside fine-tuned DiffEEG.

Corpora: TUH Seizure, TUAB, Siena Scalp EEG and Bonn.

## Setup

The upstream model code is not copied into this repository. This fetches each project
at the exact commit used here and installs our EEGMamba additions:

```bash
./setup_third_party.sh
```

It then lists the pretrained weights to download for EEGMamba, EEGDM and CBraMod.

## Layout

| Path | Contents |
|---|---|
| `code/` | feature extraction, probing, fine-tuning, baselines, SLURM scripts |
| `results/` | metrics JSON for every probe and baseline, splits, normalisation stats |
| `eegmamba_patch/` | our loader, smoke test and Triton shim for EEGMamba |
| `logs/` | SLURM logs |

## Notes

- Scripts use absolute paths on the Narval cluster (`/home/abdulh/scratch/benchmarking2`)
  and two environments: EEGDM needs PyTorch Lightning and Hydra, so it runs in its own
  venv, separate from the DiffEEG one.
- Siena is stored in volts, not microvolts. Models that assume microvolt input need the
  scale applied before extraction or they silently receive near-zero signal.
