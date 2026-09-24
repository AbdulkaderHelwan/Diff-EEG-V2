#!/bin/bash
# Process Mumtaz2016 with the NeurIPT/CBraMod preprocessing pipeline, so the
# result is a fair comparison to their published subject-independent numbers.
# Differences from our original run (kept as-is):
#   - EC + EO sessions ONLY (drop the P300 TASK session)   [they do this]
#   - 50 Hz notch filter                                   [they do this]
#   - 0.1-30 Hz band-pass                                  [they do this]
#   - global (common) average re-reference                 [they do this]
# SAME subject-wise split as the original run (test-frac 0.2), so the ONLY thing
# that changes vs the kept run is the preprocessing -- this isolates its effect.
# We keep 256 Hz (backbone's native rate) rather than their 64 Hz.
# Separate out-root so the original mumtaz2016 run is preserved.
#SBATCH --job-name=PrepMumtazM
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/PrepMumtazM_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/PrepMumtazM_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
set -euo pipefail

"/home/abdulh/venvs/pt-vae-alliance/bin/python" -u \
  /home/abdulh/scratch/EEGdiff_V2/dataset_tools/prepare_mumtaz2016.py \
  --input-root "/scratch/linah03/EpilepticSeizureProject/Dataset/external_raw/mumtaz2016" \
  --out-root "/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/mumtaz2016_matched" \
  --target-sfreq 256 \
  --window-sec 5 \
  --stride-sec 5 \
  --conditions EC EO \
  --notch-freq 50 \
  --l-freq 0.1 \
  --h-freq 30 \
  --reref-average \
  --test-frac 0.2
