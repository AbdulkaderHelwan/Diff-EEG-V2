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

## Model weights

### Our trained models

Binary seizure detection on TUH Seizure, all four trained and evaluated by us on the
same 50,000-segment test subset (6.7% seizure prevalence). Download each file and place
it at the path shown, which is where the scripts expect it.

| Model | Params | Test AUROC | PR-AUC | Macro F1 | Sens. | Spec. |
|---|---|---|---|---|---|---|
| DiffEEG, fine-tuned with RL | 9.67M | 0.859 | 0.492 | 0.638 | 0.750 | 0.835 | `results/diffeeg_ft_tusz_rl-on/best.pth` |
| EEGNet, from scratch | 2,738 | 0.880 | 0.481 | 0.550 | 0.073 | 0.998 | `results/supervised_eegnet_tusz/best.pth` | 
| ST-Transformer, from scratch | 3.43M | 0.832 | 0.404 | 0.594 | 0.742 | 0.784 | `results/supervised_sttransformer_tusz/best.pth` | 
| EEG-Conformer, from scratch | 0.97M | 0.780 | 0.265 | 0.576 | 0.651 | 0.784 | `results/supervised_conformer_tusz/best.pth` | 

Metrics are from each model's `results/<model>/result.json`. EEGNet has the highest AUROC
but a sensitivity of 0.073: at the default threshold it catches about 7% of seizures, so
its ranking is good while its operating point is not.

Each run directory also holds a `ckpt.pth` with the optimiser state, needed only to
resume training; it is not published.

### Third-party pretrained models

These are published by their authors and are not redistributed here.

| Model | Source | Place at |
|---|---|---|
| EEGDM | [Hugging Face](https://huggingface.co/jhpuah/eegdm) | https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQDpZ5TgM20zQJnc11Udk4KVAaeCiuvz53Fiw9GH9QlNXPI?e=iO3WYa 
| CBraMod | [Hugging Face](https://huggingface.co/weighting666/CBraMod) | https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQCg-4gQ76buR44ImmLNCt0vAU2T8D4SXlgnLhAvLD5TYNQ?e=0v3ANS |
| EEGMamba | [Hugging Face](https://huggingface.co/weighting666/EEGMamba) | https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQDVOWHmQqMTQ4i9T0LHk9nOAUmcgE6uRmUo8xR9KQb-um8?e=4V3HHZ |
| BIOT| included in their repositories | https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQCLnZJ9YDz3TaQ8jCcDMnACAehGNHCh0Pao3BNQdJg72XM?e=DHYjA1
| LaBraM | included in their repositories | https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQDVOWHmQqMTQ4i9T0LHk9nOAUmcgE6uRmUo8xR9KQb-um8?e=4V3HHZ |
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
