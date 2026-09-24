#!/bin/bash
# Siena (seizure vs normal, patient-wise split, correctly processed at 256Hz/
# 1280-samples/our channel order -- only fix needed was using Siena's OWN
# normalization stats, since its raw signal is in volts (~1e6x smaller scale
# than THUSZ-V2's convention). Two experiments, same pattern as Bonn:
#   1. zero-shot: reuse the pretrained THUSZ +RL classifier as-is
#   2. trained head: frozen backbone + freshly trained head on Siena's own split
# Results -> EEGdiff_V2/Benchmarking/.
#SBATCH --job-name=EEGdiff_V2_siena
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/Siena_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/Siena_%j.err
#SBATCH --time=01:00:00
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
ROOT="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/siena"
SIENA_NORM="/scratch/linah03/EpilepticSeizureProject/Dataset/Siena/processed_siena/normalization"

echo "===== Siena: zero-shot (THUSZ +RL classifier, no training) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_binary.py" \
  --processed-root "$ROOT" \
  --ckpt /home/abdulh/scratch/EEGdiff_V2/Binary_finetune_unfrozen/run_last_two_levels_20260630_083121/best_classifier.pth \
  --norm-stats-dir "$SIENA_NORM" \
  --out-root "$OUT/siena_binary"

echo
echo "===== Siena: trained head (frozen backbone + fresh head) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name siena \
  --norm-stats-dir "$SIENA_NORM" \
  --out-root "$OUT"
