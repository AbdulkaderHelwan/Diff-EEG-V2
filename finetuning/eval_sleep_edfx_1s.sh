#!/bin/bash
# Sleep-EDFx 6-class staging with 1-second windows (ablation vs 5 s / 20 s).
# 1 s = 256 samples -> small self-attention, so normal batches (no OOM). Same
# default (THUSZ) norm, same subject split, same --max-per-class cap as 5 s run.
#SBATCH --job-name=EEGdiff_sleep1s
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/Sleep1s_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/Sleep1s_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=84G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true
set -euo pipefail

PY="/home/abdulh/venvs/pt-vae-alliance/bin/python"
DIR="/home/abdulh/scratch/EEGdiff_V2/finetuning"
OUT="/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
ROOT="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/sleep_edfx_1s"
CLASSES="wake n1 n2 n3 n4 rem"
CAP=12000

echo "===== 1/3  Sleep-EDFx 1s: HEAD ONLY ====="
"$PY" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" --dataset-name sleep1s_headonly \
  --classes $CLASSES --max-per-class $CAP \
  --epochs 100 --patience 15 --batch-size 256 --out-root "$OUT"

echo
echo "===== 2/3  Sleep-EDFx 1s: FULL finetune (unfreeze=all, no RL) ====="
"$PY" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$ROOT" --dataset-name sleep1s \
  --classes $CLASSES --max-per-class $CAP \
  --unfreeze all --backbone-lr-mult 0.01 \
  --epochs 40 --patience 10 --batch-size 64 --lr 5e-4 --out-root "$OUT"

echo
echo "===== 3/3  Sleep-EDFx 1s: FULL finetune + RL ====="
"$PY" -u "$DIR/finetune_processed_rl.py" \
  --processed-root "$ROOT" --dataset-name sleep1s \
  --classes $CLASSES --max-per-class $CAP \
  --unfreeze all --backbone-lr-mult 0.01 \
  --rl-weight 0.1 --weight-power 0.5 \
  --epochs 40 --patience 10 --batch-size 64 --lr 5e-4 --out-root "$OUT"
