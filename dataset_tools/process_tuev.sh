#!/bin/bash
# Process the full TUEV corpus (359 train + 159 eval sessions) into our
# standard 22x1280 batch format, patient-wise (TUEV's own train/eval dirs are
# already patient-disjoint, reused directly -- same approach as prepare_tuab.py).
# CPU-only, no GPU needed for this preprocessing step.
#SBATCH --job-name=EEGdiff_V2_prep_tuev
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/PrepTUEV_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/PrepTUEV_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true

set -euo pipefail

VENV_PYTHON="/home/abdulh/venvs/pt-vae-alliance/bin/python"

"$VENV_PYTHON" -u /home/abdulh/scratch/EEGdiff_V2/dataset_tools/prepare_tuev.py
