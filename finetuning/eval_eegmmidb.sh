#!/bin/bash
# EEGMMIDB motor-imagery/movement (9-class): two experiments, same pattern as
# Bonn -- (1) NO finetuning: frozen backbone + freshly trained head, (2)
# FINETUNING: partial backbone unfreeze + trained head. Results ->
# EEGdiff_V2/Benchmarking/eegmmidb_{trainedhead,finetuned_last_two_levels}_<ts>/.
#SBATCH --job-name=EEGdiff_V2_eegmmidb
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/EEGMMIDB_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/EEGMMIDB_%j.err
#SBATCH --time=06:00:00
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

echo "===== EEGMMIDB: NO finetuning (frozen backbone + trained head) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name eegmmidb \
  --out-root "$OUT"

echo
echo "===== EEGMMIDB: FINETUNING (unfreeze last_two_levels + trained head) ====="
"$VENV_PYTHON" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name eegmmidb \
  --unfreeze last_two_levels \
  --epochs 40 \
  --patience 8 \
  --batch-size 128 \
  --lr 5e-4 \
  --backbone-lr-mult 0.1 \
  --out-root "$OUT"
