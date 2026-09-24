# Diff-EEG V2

A 40.7M-parameter denoising-diffusion model pre-trained on unlabelled EEG and used as a
frozen or partially fine-tuned feature extractor for seizure detection, abnormal-EEG
detection and external EEG tasks.

> **Status.** V2 is superseded by V3 and is published for reproducibility. See
> [Known issues](#known-issues) before reusing any number from this repository.

Related paper (V1): *DiffEEG: A Self-Supervised Denoising Diffusion Model for Learning
EEG Generic Representations*, [arXiv:2607.11578](https://arxiv.org/abs/2607.11578).

## Model

| | |
|---|---|
| Architecture | 1-D U-Net with residual blocks, FiLM timestep conditioning and multi-head self-attention (`DeepEnhancedEEGDiffusionModel`, `--model-size large`) |
| Parameters | 40,679,574 |
| Configuration | `in_channels=22, model_channels=64, channel_multipliers=[1,2,4,8], num_res_blocks=3, time_emb_dim=768, attention_heads=8` |
| Input | 22 channels × 1280 samples (5 s at 256 Hz) |
| Pre-training data | 1,445,392 unlabelled segments: TUH Seizure (non-seizure), CHB-MIT (non-seizure), TUAB (normal) |
| Embedding | encoder probed at timesteps {50, 250, 500, 750, 950}, each level average-pooled and concatenated → 4,800-d |

## Model weights ([link](https://lauedu74602-my.sharepoint.com/:f:/g/personal/abedelkader_helwan_lau_edu_lb/IgA6exIChUv-T4c7WWJdo-b4AT3dD_1_SkWII2RXbeya_yY?e=pnUv3l))

The checkpoints are 187–466 MB each, over GitHub's 100 MB file limit, so they are hosted
on OneDrive. Download each file and place it at the path shown, which is where the
scripts expect it.

| Checkpoint | Size | Place at | Download |
|---|---|---|---|
| Pre-trained backbone (epoch 26, best validation loss) | 466 MB | `training_diffusion_v2/best_EEGDIFF_V2.pth` | (https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQAxsDKK45aFQKq4d7cWjAFGASe4yBa_WSTkasZ27cArqvE?e=nxesK2)|
| TUH Seizure detection, last two levels unfrozen, with RL | 187 MB | `Binary_finetune_unfrozen/run_last_two_levels_20260630_083121/best_classifier.pth` | https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQCSJOf0xEWwQreoXq5KUjP2AftooFbwIypJQm3WP_3JZ2I?e=llldL2 |
| TUAB abnormal detection, `unfreeze_3` | 187 MB | `TUAB_Res/run_20260702_034225/unfreeze_3/best.pth` | [ADD_ONEDRIVE_LINK ](https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQCrQg57h_tlRK1KQphKH9l_ARdB9C9_UUxJN_H0oGnt2nA?e=TA5exa)|

## Results

All numbers are on held-out test sets and come from the result files in this repository.

| Task | Checkpoint | Test metrics | Source |
|---|---|---|---|
| TUH Seizure, seizure vs non-seizure (6.7% prevalence) | unfrozen + RL | ROC-AUC 0.841, PR-AUC 0.484, seizure F1 0.473 at threshold 0.5 | `PROJECT_LOG.md` §5.1 |
| TUAB, normal vs abnormal | `unfreeze_3` | ROC-AUC 0.862, PR-AUC 0.854, balanced accuracy 0.760 | `TUAB_Res/run_20260702_034225/summary.json` |
| Siena, seizure vs non-seizure (0.7% prevalence) | full fine-tune + RL | macro F1 0.638, weighted F1 0.991 | `Benchmarking/siena_finetuned_all_RL_20260707_100303/summary.json` |
| TUEV, 4 classes | full fine-tune + RL | macro F1 0.499, weighted F1 0.632 | `Benchmarking/tuev_4class_finetuned_all_RL_20260708_072026/summary.json` |

On TUAB, `unfreeze_3` was the configuration selected on validation F2. `unfreeze_2` scores
higher on the test set (ROC-AUC 0.878), but choosing it on that basis would be selecting
on test data, so the validation-selected model is the one reported and released.

On Siena, accuracy (0.992) is not reported as a result: at 0.7% prevalence a classifier
that never predicts a seizure scores 0.993.

## Known issues

1. **Pre-training diverged.** The loss became NaN at epoch 32. The released backbone is
   the epoch-26 checkpoint with the best validation loss, and both saved checkpoints were
   verified to contain only finite weights.
2. **TUEV was windowed incorrectly.** `dataset_tools/prepare_tuev.py` merges adjacent
   annotation rows into spans and extracts one window per span, which gives 2,621 windows.
   The protocol used by BIOT, LaBraM, CBraMod and EEGDM extracts one window per `.rec`
   row, which gives 113,353 (83,932 train, 29,421 eval). The TUEV results above are
   therefore **not comparable with published TUEV numbers**. For the same reason, the
   statement in `PROJECT_LOG.md` that spike-and-slow-wave F1 is limited by data scarcity
   (20 test examples) is a preprocessing artefact: the standard protocol has 567.
3. **The RL decision layer shifts the threshold; it does not improve discrimination.** In
   controlled ablations on the V1 model, a single learned scalar reproduces its effect on
   ROC-AUC to four decimal places, and on patient-wise subtype classification it lowered
   macro F1. Treat "+RL" gains as operating-point effects.
4. **Paths are cluster-specific.** Scripts reference `/scratch/...` locations on the
   Digital Research Alliance of Canada Narval cluster and need editing to run elsewhere.

## Data

No EEG data is included. The corpora are available from their providers and several
require a data-use agreement:

- TUH Seizure (TUSZ), TUAB and TUEV: [Temple University Hospital EEG corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)
- CHB-MIT: [PhysioNet](https://physionet.org/content/chbmit/)
- Siena Scalp EEG: [PhysioNet](https://physionet.org/content/siena-scalp-eeg/)
- Bonn, Sleep-EDF, EEGMMIDB, Mumtaz 2016: see `Data-Processing/download_public_eeg.py`

`paper/analysis/interp.npz` holds derived analysis outputs (t-SNE coordinates, an averaged
saliency map and 4,800-d embeddings of TUEV, Siena and Bonn segments), not raw EEG.

## Repository layout

| Path | Contents |
|---|---|
| `Diff_EEG_train_v2.py`, `train_diffusion_v2.sh` | pre-training |
| `finetuning/` | fine-tuning and evaluation scripts |
| `dataset_tools/`, `Data-Processing/` | per-corpus preprocessing |
| `Binary_finetune_*`, `TUHSZ_Res/`, `TUAB_Res/` | TUH Seizure and TUAB runs: configs, histories, scores |
| `Benchmarking/` | external-corpus runs and the prevalence-matched literature comparison |
| `benchmarking2/` | controlled comparison against EEGDM, BIOT, LaBraM, CBraMod and EEGMamba (see its README) |
| `sync_benchmarking2.sh` | refreshes `benchmarking2/` from the working copy on the cluster |
| `paper/` | V2 paper draft and analysis figures |
| `logs/` | SLURM output and error logs |
| `PROJECT_LOG.md` | full experiment log |

## Pre-training

```bash
sbatch train_diffusion_v2.sh
```

This runs `Diff_EEG_train_v2.py --model-size large --epochs 1000 --batch-size 256 --lr 5e-5`
on one A100. The data paths in the script point to preprocessed batches on the cluster.
