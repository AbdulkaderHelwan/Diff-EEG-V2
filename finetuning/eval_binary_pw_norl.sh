#!/bin/bash
# Evaluate the best NO-RL binary checkpoint on val + eval (same methodology as
# eval_binary_pw.sh, but for a no-RL checkpoint). Saves into Binary_finetune_norl/.
#SBATCH --job-name=EEGdiff_V2_eval_norl
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/EvalBinaryPW_norl_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/EvalBinaryPW_norl_%j.err
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
SCRIPT="/home/abdulh/scratch/EEGdiff_V2/finetuning/eval_binary_PW_norl.py"
CKPT="/home/abdulh/scratch/EEGdiff_V2/Binary_finetune_norl/run_norl_none_20260702_071546/best_classifier.pth"

cd /home/abdulh/scratch/EEGdiff_V2

"$VENV_PYTHON" -u "$SCRIPT" \
  --ckpt "$CKPT" \
  --res-root "/home/abdulh/scratch/EEGdiff_V2/Binary_finetune_norl" \
  --batch-size 256 \
  --num-workers 8
