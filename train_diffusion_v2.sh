#!/bin/bash
# Diff-EEG V2 pre-training
#SBATCH --job-name=EEGdiff_V2
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/eegdiff_v2_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/eegdiff_v2_%j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=300G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true

set -euo pipefail

VENV_PYTHON="/home/abdulh/venvs/pt-vae-alliance/bin/python"
WORKDIR="/home/abdulh/scratch"
SCRIPT="/home/abdulh/scratch/EEGdiff_V2/Diff_EEG_train_v2.py"
LOGDIR="/home/abdulh/scratch/EEGdiff_V2/logs"

mkdir -p "$LOGDIR"
cd "$WORKDIR"

"$VENV_PYTHON" -u "$SCRIPT" \
  --model-size large \
  --epochs 1000 \
  --batch-size 256 \
  --val-batch-size 64 \
  --num-workers 4 \
  --lr 5e-5 \
  --data-parallel \
  --cache-in-memory \
  --save-every 1

