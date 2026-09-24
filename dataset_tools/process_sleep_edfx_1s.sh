#!/bin/bash
# Reprocess Sleep-EDFx with 1-second windows (window-size ablation: 1 / 5 / 20 s).
# 1 s @ 256 Hz = 256 samples. Only the window length changes vs the 5 s run.
#SBATCH --job-name=PrepSleep1s
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/PrepSleep1s_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/PrepSleep1s_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=84G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
set -euo pipefail

"/home/abdulh/venvs/pt-vae-alliance/bin/python" -u \
  /home/abdulh/scratch/EEGdiff_V2/dataset_tools/prepare_external_eeg.py \
  --dataset sleep_edfx --multiclass \
  --raw-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_raw/sleep_edfx \
  --out-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/sleep_edfx_1s \
  --window-sec 1 --stride-sec 1
