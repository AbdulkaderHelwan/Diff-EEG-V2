#!/bin/bash
# Auxiliary-SSL fine-tuning sweep on TUEV 4-class. Fine-tune the backbone for
# classification while keeping its diffusion-denoising objective as an auxiliary
# regularizer (total = CE + ssl_coef * MSE(pred_noise, noise)). ssl_coef=0 is the
# plain full-finetune baseline; 0.1 / 0.3 add increasing SSL regularization.
# Tests whether staying on the pretraining manifold helps this OOD-ish task.
#SBATCH --job-name=EEGdiff_auxssl
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/AuxSSL_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/AuxSSL_%j.err
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true
set -euo pipefail

PY="/home/abdulh/venvs/pt-vae-alliance/bin/python"
DIR="/home/abdulh/scratch/EEGdiff_V2/finetuning"
OUT="/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
ROOT="/scratch/linah03/EpilepticSeizureProject/Dataset/TUEV_v2.0.1/processed_TUEV_4class"
NORM="$ROOT/normalization"
CLASSES="artf bckg epileptiform eyem"

for COEF in 0.0 0.1 0.3; do
  echo; echo "############## aux-SSL fine-tune: ssl_coef=$COEF ##############"
  "$PY" -u "$DIR/finetune_processed_aux_ssl.py" \
    --processed-root "$ROOT" \
    --dataset-name tuev4 \
    --norm-stats-dir "$NORM" \
    --classes $CLASSES \
    --unfreeze all \
    --ssl-coef "$COEF" \
    --backbone-lr-mult 0.01 \
    --epochs 40 \
    --patience 10 \
    --batch-size 32 \
    --lr 5e-4 \
    --out-root "$OUT"
done

echo; echo "===== AUX-SSL SWEEP (0.0 / 0.1 / 0.3) DONE ====="
