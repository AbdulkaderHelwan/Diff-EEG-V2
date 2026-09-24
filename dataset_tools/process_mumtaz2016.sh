#!/bin/bash
# Process Mumtaz2016 (MDD vs healthy) EDF -> 22x1280 @256Hz windows.
# Source already at 256 Hz / 19 scalp electrodes (10-20), so only channel
# projection + windowing is needed. Subject-wise, group-stratified split.
#SBATCH --job-name=PrepMumtaz
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/PrepMumtaz_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/PrepMumtaz_%j.err
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
  --out-root "/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/mumtaz2016" \
  --target-sfreq 256 \
  --window-sec 5 \
  --stride-sec 5 \
  --conditions EC EO TASK \
  --test-frac 0.2
