#!/bin/bash
# Diff-EEG V2 finetuning on TUAB (normal vs abnormal), patient-wise,
# with progressive-unfreezing experiments.
#SBATCH --job-name=EEGdiff_TUAB
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/TUAB_PW_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/TUAB_PW_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=84G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true

set -euo pipefail

VENV_PYTHON="/home/abdulh/venvs/pt-vae-alliance/bin/python"
WORKDIR="/home/abdulh/scratch/EEGdiff_V2"
SCRIPT="/home/abdulh/scratch/EEGdiff_V2/finetuning/finetune_tuab_PW.py"
LOGDIR="/home/abdulh/scratch/EEGdiff_V2/logs"

mkdir -p "$LOGDIR"
cd "$WORKDIR"

# Partial unfreezing only: unfreeze the 2 and 3 deepest encoder levels.
# Full-encoder unfreeze (unfreeze_all) overfit fast (early-stopped at epoch 7),
# so we stick to a small number of layers. Both experiments run in one job and
# the script reports the better of the two. More epochs + patience than the
# overfit run, since partial unfreezing keeps improving for longer.
"$VENV_PYTHON" -u "$SCRIPT" \
  --epochs 40 \
  --batch-size 256 \
  --val-batch-size 256 \
  --num-workers 8 \
  --lr 5e-4 \
  --backbone-lr-mult 0.1 \
  --patience 10 \
  --experiments unfreeze_2,unfreeze_3
