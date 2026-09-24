#!/bin/bash
# Mumtaz2016 MDD-detection -- MATCHED-preprocessing run (EC+EO only, 50Hz notch,
# 0.1-30Hz band-pass, global-average reference), same subject-wise split as the
# original run. Directly comparable to NeurIPT/CBraMod's subject-independent
# Mumtaz2016 numbers (their Bal.Acc ~0.92-0.98). Positive class = mdd.
# Root: .../external_processed/mumtaz2016_matched
#SBATCH --job-name=EEGdiff_mumtazM
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/MumtazM_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/MumtazM_%j.err
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
ROOT="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/mumtaz2016_matched"
NORM="$ROOT/normalization"
CLASSES="healthy mdd"

echo "===== 1/3  Mumtaz2016 MATCHED: HEAD ONLY ====="
"$PY" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtazM_headonly \
  --norm-stats-dir "$NORM" \
  --classes $CLASSES \
  --epochs 100 \
  --patience 15 \
  --out-root "$OUT"

echo
echo "===== 2/3  Mumtaz2016 MATCHED: FULL finetune (unfreeze=all, no RL) ====="
"$PY" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtazM \
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
echo "===== 3/3  Mumtaz2016 MATCHED: FULL finetune + RL ====="
"$PY" -u "$DIR/finetune_processed_rl.py" \
  --processed-root "$ROOT" \
  --dataset-name mumtazM \
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
