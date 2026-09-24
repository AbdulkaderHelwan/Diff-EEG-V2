#!/bin/bash
# TUEV, 4-class merged task (artf/bckg/eyem/epileptiform -- spsw+gped+pled
# merged, since they proved genuinely inseparable at our embedding's temporal
# resolution in the 6-class experiments; see PROJECT_LOG 5.6/5.7). Two runs:
# frozen baseline (comparison point) and the best-known recipe from the 6-class
# work (full finetune + RL + sqrt-softened class weighting, 100 epochs).
# Results -> EEGdiff_V2/Benchmarking/.
#SBATCH --job-name=EEGdiff_V2_tuev4c
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/TUEV4class_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/TUEV4class_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true

set -euo pipefail

VENV_PYTHON="/home/abdulh/venvs/pt-vae-alliance/bin/python"
DIR="/home/abdulh/scratch/EEGdiff_V2/finetuning"
OUT="/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
ROOT="/scratch/linah03/EpilepticSeizureProject/Dataset/TUEV_v2.0.1/processed_TUEV_4class"
NORM="$ROOT/normalization"

echo "===== TUEV 4-class: frozen baseline (no finetune) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$ROOT" \
  --dataset-name tuev_4class \
  --norm-stats-dir "$NORM" \
  --epochs 100 \
  --patience 20 \
  --out-root "$OUT"

echo
echo "===== TUEV 4-class: full finetune + RL (softened weighting) ====="
"$VENV_PYTHON" -u "$DIR/finetune_processed_rl.py" \
  --processed-root "$ROOT" \
  --dataset-name tuev_4class \
  --norm-stats-dir "$NORM" \
  --unfreeze all \
  --backbone-lr-mult 0.01 \
  --rl-weight 0.1 \
  --weight-power 0.5 \
  --epochs 100 \
  --patience 20 \
  --batch-size 64 \
  --lr 5e-4 \
  --out-root "$OUT"
