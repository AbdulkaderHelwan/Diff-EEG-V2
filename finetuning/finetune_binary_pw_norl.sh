#!/bin/bash
# Diff-EEG V2 binary finetuning — NO-RL ablation.
# Same setup as finetune_binary_pw_unfreeze.sh but the ReinforcedDecisionLayer is
# removed: the MLP head is trained directly with weighted CE on the embedding.
# Change --unfreeze to fill the with/without-RL x frozen/finetuned matrix:
#   --unfreeze none            -> frozen backbone  (core "embedding is good" number)
#   --unfreeze last_two_levels -> partial finetune (matches the +RL run)
#SBATCH --job-name=EEGdiff_V2_norl
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/BinaryPW_norl_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/BinaryPW_norl_%j.err
#SBATCH --time=75:00:00
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
SCRIPT="/home/abdulh/scratch/EEGdiff_V2/finetuning/finetune_binary_PW_norl.py"
LOGDIR="/home/abdulh/scratch/EEGdiff_V2/logs"

mkdir -p "$LOGDIR"
cd "$WORKDIR"

# Frozen backbone by default (the headline embedding-quality number, no RL).
# For the finetuned no-RL cell, set --unfreeze last_two_levels (+ --grad-checkpoint,
# and drop batch-size to 128 as in the +RL run).
"$VENV_PYTHON" -u "$SCRIPT" \
  --epochs 100 \
  --batch-size 256 \
  --val-batch-size 256 \
  --num-workers 8 \
  --shuffle-buffer 4000 \
  --lr 5e-5 \
  --unfreeze none
