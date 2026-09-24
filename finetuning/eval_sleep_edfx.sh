#!/bin/bash
# sleep_edfx (6-stage staging: wake/n1/n2/n3/n4/rem; 'normal' dropped -- it's a
# fallback for unscored/gap epochs, not a real stage). Massive dataset
# (2.77M windows, ~330GB) -- subsampled at the FILE level via --max-per-class
# so we never load more than needed. Uses --per-window-norm (same reason as
# EEGMMIDB: this pipeline's raw scale isn't THUSZ-V2-compatible, and
# prepare_external_eeg.py doesn't compute its own dataset-level stats).
# Same 3-experiment pattern as Siena/TUEV: no-finetune, full-finetune (no RL),
# full-finetune+RL. Results -> EEGdiff_V2/Benchmarking/.
#SBATCH --job-name=EEGdiff_V2_sleepedfx
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/SleepEDFXeval_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/SleepEDFXeval_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true

set -euo pipefail

VENV_PYTHON="/home/abdulh/venvs/pt-vae-alliance/bin/python"
DIR="/home/abdulh/scratch/EEGdiff_V2/finetuning"
OUT="/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
ROOT="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/sleep_edfx"
CLASSES="wake n1 n2 n3 n4 rem"
CAP=12000

echo "===== sleep_edfx: NO finetuning (frozen backbone + trained head) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name sleep_edfx \
  --classes $CLASSES \
  --max-per-class $CAP \
  --per-window-norm \
  --epochs 100 \
  --patience 15 \
  --out-root "$OUT"

echo
echo "===== sleep_edfx: FULL finetuning (unfreeze=all, no RL) ====="
"$VENV_PYTHON" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name sleep_edfx \
  --classes $CLASSES \
  --max-per-class $CAP \
  --per-window-norm \
  --unfreeze all \
  --backbone-lr-mult 0.01 \
  --epochs 40 \
  --patience 10 \
  --batch-size 128 \
  --lr 5e-4 \
  --out-root "$OUT"

echo
echo "===== sleep_edfx: FULL finetuning + RL ====="
"$VENV_PYTHON" -u "$DIR/finetune_processed_rl.py" \
  --processed-root "$ROOT" \
  --dataset-name sleep_edfx \
  --classes $CLASSES \
  --max-per-class $CAP \
  --per-window-norm \
  --unfreeze all \
  --backbone-lr-mult 0.01 \
  --rl-weight 0.1 \
  --epochs 40 \
  --patience 10 \
  --batch-size 128 \
  --lr 5e-4 \
  --out-root "$OUT"
