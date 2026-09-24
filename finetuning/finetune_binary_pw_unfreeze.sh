#!/bin/bash
# Diff-EEG V2 finetuning Binary — UNFREEZE variant (real backbone fine-tuning).
# Counterpart to finetune_binary_pw.sh, which keeps the backbone fully frozen.
#SBATCH --job-name=EEGdiff_V2_unf
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/BinaryPW_unfreeze_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/BinaryPW_unfreeze_%j.err
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
WORKDIR="/home/abdulh/scratch/EEGdiff_V2"
SCRIPT="/home/abdulh/scratch/EEGdiff_V2/finetuning/finetune_binary_PW_unfreeze.py"
LOGDIR="/home/abdulh/scratch/EEGdiff_V2/logs"

mkdir -p "$LOGDIR"
cd "$WORKDIR"

# Backbone fine-tuning controls:
#   --unfreeze {none|last_level|last_two_levels|encoder_all|all}
#   --backbone-lr   LR for unfrozen backbone (default: head LR / 10)
#   --grad-checkpoint  trade compute for memory (recommended for encoder_all/all)
# Batch size is halved vs the frozen run (256 -> 128) because backprop now flows
# through the backbone x5 probe timesteps; raise it back if memory allows.
"$VENV_PYTHON" -u "$SCRIPT" \
  --epochs 100 \
  --batch-size 128 \
  --val-batch-size 256 \
  --num-workers 8 \
  --shuffle-buffer 4000 \
  --lr 5e-5 \
  --backbone-lr 5e-6 \
  --unfreeze last_two_levels \
  --grad-checkpoint
