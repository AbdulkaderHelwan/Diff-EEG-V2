#!/bin/bash
# Process the full sleep_edfx corpus (197 PSG+Hypnogram sessions) into our
# standard 22x1280 batch format, 6-stage sleep labels (wake/n1/n2/n3/n4/rem).
# CPU-only, no GPU needed. After the per-window-resample perf fix, ~14s/file
# (~46 min total), down from ~8min/file (~26h) before the fix.
#SBATCH --job-name=EEGdiff_V2_prep_sleepedfx
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/PrepSleepEDFX_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/PrepSleepEDFX_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true

set -euo pipefail

VENV_PYTHON="/home/abdulh/venvs/pt-vae-alliance/bin/python"

"$VENV_PYTHON" -u /home/abdulh/scratch/EEGdiff_V2/dataset_tools/prepare_external_eeg.py \
  --dataset sleep_edfx --multiclass \
  --raw-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_raw/sleep_edfx \
  --out-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/sleep_edfx
