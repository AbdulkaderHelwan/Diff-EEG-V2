#!/bin/bash
# Diff-EEG V2 finetuning Binary
#SBATCH --job-name=EEGdiff_V2
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/BinaryClassificationPW_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/BinaryClassificationPW_%j.err
#SBATCH --time=100:00:00
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
WORKDIR="/home/abdulh/scratch"
SCRIPT="/home/abdulh/scratch/EEGdiff_V2/finetuning/finetune_binary_PW.py"
LOGDIR="/home/abdulh/scratch/EEGdiff_V2/logs"

mkdir -p "$LOGDIR"
cd "$WORKDIR"

# Sequential block-shuffle loader + bf16 AMP are built into the script.
# (unused legacy flags like --data-parallel/--cache-in-memory are ignored)
"$VENV_PYTHON" -u "$SCRIPT" \
  --epochs 100 \
  --batch-size 256 \
  --val-batch-size 256 \
  --num-workers 8 \
  --shuffle-buffer 4000 \
  --lr 5e-5

