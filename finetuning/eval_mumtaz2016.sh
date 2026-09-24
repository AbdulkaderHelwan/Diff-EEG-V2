#!/bin/bash
# Mumtaz2016 MDD-detection (binary: healthy vs mdd), subject-wise split.
# Positive class = 'mdd' (classes ordered "healthy mdd" -> index 1), so the
# eval reports ROC-AUC / PR-AUC for MDD detection alongside acc / macro-F1.
#
# Ladder (same as our other external datasets):
#   1. HEAD ONLY   -- frozen backbone + trained MLP head (reports ROC/PR-AUC)
#   2. UNFROZEN    -- full backbone finetune (gentle backbone LR)
#   3. UNFROZEN+RL -- full finetune + macro-F1 RL decision head
# Uses Mumtaz's OWN normalization stats.
#SBATCH --job-name=EEGdiff_mumtaz
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/Mumtaz_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/Mumtaz_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true
set -euo pipefail

PY="/home/abdulh/venvs/pt-vae-alliance/bin/python"
DIR="/home/abdulh/scratch/EEGdiff_V2/finetuning"
OUT="/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
ROOT="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/mumtaz2016"
NORM="$ROOT/normalization"
CLASSES="healthy mdd"

echo "===== 1/3  Mumtaz2016: HEAD ONLY (frozen backbone + trained head) ====="
"$PY" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtaz_headonly \
  --norm-stats-dir "$NORM" \
  --classes $CLASSES \
  --epochs 100 \
  --patience 15 \
  --out-root "$OUT"

echo
echo "===== 2/3  Mumtaz2016: FULL finetune (unfreeze=all, no RL) ====="
"$PY" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtaz \
  --norm-stats-dir "$NORM" \
  --classes $CLASSES \
  --unfreeze all \
  --backbone-lr-mult 0.01 \
  --epochs 40 \
  --patience 10 \
  --batch-size 64 \
  --lr 5e-4 \
  --out-root "$OUT"

echo
echo "===== 3/3  Mumtaz2016: FULL finetune + RL ====="
"$PY" -u "$DIR/finetune_processed_rl.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtaz \
  --norm-stats-dir "$NORM" \
  --classes $CLASSES \
  --unfreeze all \
  --backbone-lr-mult 0.01 \
  --rl-weight 0.1 \
  --weight-power 0.5 \
  --epochs 40 \
  --patience 10 \
  --batch-size 64 \
  --lr 5e-4 \
  --out-root "$OUT"
