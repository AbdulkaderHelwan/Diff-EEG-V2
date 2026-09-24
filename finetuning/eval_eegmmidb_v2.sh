#!/bin/bash
# EEGMMIDB follow-up experiments after the 9-class result was near/below chance.
# Combines: (1) per-window normalization instead of THUSZ global stats,
# (2) simplified 2-class task (executed left_fist vs right_fist, runs 3/7/11 --
#     the classic motor-imagery benchmark, no reprocessing needed), (3) a
# diagnostic pooled-random split (mixes subjects between train/test) to check
# whether the official subject-wise split's difficulty is a cross-subject
# generalization issue, and (4) gentler finetuning (less unfreezing, lower
# backbone LR) since last_two_levels/0.1 destabilized training. Results ->
# EEGdiff_V2/Benchmarking/.
#SBATCH --job-name=EEGdiff_V2_eegmmidb_v2
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/EEGMMIDB_v2_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/EEGMMIDB_v2_%j.err
#SBATCH --time=04:00:00
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
DIR="/home/abdulh/scratch/EEGdiff_V2/finetuning"
OUT="/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
ROOT="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/eegmmidb"

echo "===== [1] 2-class, OFFICIAL subject-wise split, per-window norm, NO finetuning ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name eegmmidb_2class_subjectwise \
  --classes left_fist right_fist \
  --per-window-norm \
  --out-root "$OUT"

echo
echo "===== [2] 2-class, OFFICIAL subject-wise split, per-window norm, GENTLE finetuning ====="
"$VENV_PYTHON" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name eegmmidb_2class_subjectwise \
  --classes left_fist right_fist \
  --per-window-norm \
  --unfreeze last_level \
  --backbone-lr-mult 0.01 \
  --epochs 40 \
  --patience 8 \
  --batch-size 128 \
  --lr 5e-4 \
  --out-root "$OUT"

echo
echo "===== [3] 2-class, POOLED RANDOM split (diagnostic), per-window norm, NO finetuning ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name eegmmidb_2class_pooled_diagnostic \
  --classes left_fist right_fist \
  --per-window-norm \
  --pooled-random-split \
  --out-root "$OUT"
