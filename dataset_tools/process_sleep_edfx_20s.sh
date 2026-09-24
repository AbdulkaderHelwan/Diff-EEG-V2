#!/bin/bash
# Reprocess Sleep-EDFx with 20-second windows instead of 5 s. Sleep is scored in
# 30 s epochs, and stages (N1/N2/N3, spindles, K-complexes, slow waves) are
# defined over long timescales -- a 5 s window is too short to separate them
# (frozen kNN was at chance). 20 s gives the model most of an epoch's context.
# Backbone accepts variable length (20 s @ 256 Hz = 5120 samples, /16 = 320).
# Only the window length changes vs the original run; separate out-root.
#SBATCH --job-name=PrepSleep20s
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/PrepSleep20s_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/PrepSleep20s_%j.err
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
  --out-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/sleep_edfx_20s \
  --window-sec 20 --stride-sec 20
