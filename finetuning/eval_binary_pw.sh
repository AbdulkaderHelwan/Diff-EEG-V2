#!/bin/bash
# Diff-EEG V2 — standalone EVAL of a saved binary seizure classifier checkpoint.
# Reports ROC/PR/F1 on the held-out eval split + F1-optimal threshold (chosen on val).
#SBATCH --job-name=EEGdiff_V2_eval
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/EvalBinaryPW_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/EvalBinaryPW_%j.err
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

cd /home/abdulh/scratch

# --ckpt defaults to the currently-training run's best_classifier.pth; override if needed.
"$VENV_PYTHON" -u "$SCRIPT" \
  --batch-size 256 \
  --num-workers 8
