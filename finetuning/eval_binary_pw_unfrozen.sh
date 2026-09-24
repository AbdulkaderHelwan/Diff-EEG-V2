#!/bin/bash
# Evaluate the UNFROZEN (+RL, last_two_levels) binary checkpoint on val + eval,
# saving results alongside the model in Binary_finetune_unfrozen/.
#SBATCH --job-name=EEGdiff_V2_eval_unf
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/EvalBinaryPW_unfrozen_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/EvalBinaryPW_unfrozen_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true

set -euo pipefail

VENV_PYTHON="/home/abdulh/venvs/pt-vae-alliance/bin/python"
SCRIPT="/home/abdulh/scratch/EEGdiff_V2/finetuning/eval_binary_PW.py"
CKPT="/home/abdulh/scratch/EEGdiff_V2/Binary_finetune_unfrozen/run_last_two_levels_20260630_083121/best_classifier.pth"

cd /home/abdulh/scratch/EEGdiff_V2

"$VENV_PYTHON" -u "$SCRIPT" \
  --ckpt "$CKPT" \
  --res-root "/home/abdulh/scratch/EEGdiff_V2/Binary_finetune_unfrozen" \
  --batch-size 256 \
  --num-workers 8
