#!/bin/bash
# EEGMMIDB 2-class: FULL finetuning (unfreeze=all -- entire encoder + stem, the
# maximum meaningful unfreeze since bottleneck/up_blocks are never touched by
# the classifier's forward pass). Same gentle LR settings that stabilized the
# last_level run (backbone-lr-mult=0.01), for a direct, isolated comparison of
# "how much to unfreeze" on this task. Results -> EEGdiff_V2/Benchmarking/.
#SBATCH --job-name=EEGdiff_V2_eegmmidb_fullft
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/EEGMMIDB_fullft_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/EEGMMIDB_fullft_%j.err
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

echo "===== 2-class, OFFICIAL subject-wise split, per-window norm, FULL finetuning (unfreeze=all) ====="
"$VENV_PYTHON" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name eegmmidb_2class_subjectwise \
  --classes left_fist right_fist \
  --per-window-norm \
  --unfreeze all \
  --backbone-lr-mult 0.01 \
  --epochs 40 \
  --patience 8 \
  --batch-size 128 \
  --lr 5e-4 \
  --out-root "$OUT"
