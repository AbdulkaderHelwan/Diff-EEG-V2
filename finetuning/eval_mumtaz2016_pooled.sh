#!/bin/bash
# Mumtaz2016 MDD-detection -- NON-subject-independent (POOLED RANDOM) split, to
# compare against foundation-model papers that evaluate Mumtaz2016 this way.
#
# Instead of our honest subject-wise split, --pooled-random-split pools all
# windows and draws a fresh random 80/20 train/test. The same subject's windows
# then appear on BOTH sides (subject leakage), which is the protocol behind the
# high published Mumtaz2016 numbers (~0.95-0.98). Same processed data + norm
# stats as the subject-wise run (external_processed/mumtaz2016); only the split
# differs, so the two runs are directly comparable.
#
# Ladder: head-only / full finetune / full finetune + RL. Positive class = mdd.
#SBATCH --job-name=EEGdiff_mumtazP
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/MumtazPooled_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/MumtazPooled_%j.err
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

echo "===== 1/3  Mumtaz2016 POOLED: HEAD ONLY ====="
"$PY" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtaz_pooled_headonly \
  --norm-stats-dir "$NORM" \
  --classes $CLASSES \
  --pooled-random-split \
  --epochs 100 \
  --patience 15 \
  --out-root "$OUT"

echo
echo "===== 2/3  Mumtaz2016 POOLED: FULL finetune (unfreeze=all, no RL) ====="
"$PY" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtaz_pooled \
  --norm-stats-dir "$NORM" \
  --classes $CLASSES \
  --pooled-random-split \
  --unfreeze all \
  --backbone-lr-mult 0.01 \
  --epochs 40 \
  --patience 10 \
  --batch-size 64 \
  --lr 5e-4 \
  --out-root "$OUT"

echo
echo "===== 3/3  Mumtaz2016 POOLED: FULL finetune + RL ====="
"$PY" -u "$DIR/finetune_processed_rl.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtaz_pooled \
  --norm-stats-dir "$NORM" \
  --classes $CLASSES \
  --pooled-random-split \
  --unfreeze all \
  --backbone-lr-mult 0.01 \
  --rl-weight 0.1 \
  --weight-power 0.5 \
  --epochs 40 \
  --patience 10 \
  --batch-size 64 \
  --lr 5e-4 \
  --out-root "$OUT"
