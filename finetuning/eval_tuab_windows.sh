#!/bin/bash
# TUAB abnormality (binary normal/abnormal) window-size comparison: 5 s vs 10 s,
# same pipeline. Head-only (frozen, reports ROC-AUC/PR-AUC) + full finetune, per
# window size. 10 s = 2560 samples -> reduce --embed-batch-size / finetune batch
# (self-attention is quadratic in sequence length). Each root uses its OWN norm.
#SBATCH --job-name=EEGdiff_tuabwin
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/TUABwin_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/TUABwin_%j.err
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=84G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="/home/abdulh/venvs/pt-vae-alliance/bin/python"
DIR="/home/abdulh/scratch/EEGdiff_V2/finetuning"
OUT="/home/abdulh/scratch/EEGdiff_V2/Benchmarking"
BASE="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed"
CLASSES="normal abnormal"
CAP=20000

echo "########## TUAB 5s: HEAD ONLY ##########"
"$PY" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$BASE/tuab_5s" --dataset-name tuab5s_headonly \
  --norm-stats-dir "$BASE/tuab_5s/normalization" --classes $CLASSES \
  --max-per-class $CAP --embed-batch-size 128 \
  --epochs 100 --patience 15 --batch-size 256 --out-root "$OUT"

echo "########## TUAB 5s: FULL finetune ##########"
"$PY" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$BASE/tuab_5s" --dataset-name tuab5s \
  --norm-stats-dir "$BASE/tuab_5s/normalization" --classes $CLASSES \
  --max-per-class $CAP --unfreeze all --backbone-lr-mult 0.01 \
  --epochs 30 --patience 8 --batch-size 64 --lr 5e-4 --out-root "$OUT"

echo "########## TUAB 10s: HEAD ONLY ##########"
"$PY" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root "$BASE/tuab_10s" --dataset-name tuab10s_headonly \
  --norm-stats-dir "$BASE/tuab_10s/normalization" --classes $CLASSES \
  --max-per-class $CAP --embed-batch-size 32 \
  --epochs 100 --patience 15 --batch-size 256 --out-root "$OUT"

echo "########## TUAB 10s: FULL finetune ##########"
"$PY" -u "$DIR/finetune_processed_multiclass.py" \
  --processed-root "$BASE/tuab_10s" --dataset-name tuab10s \
  --norm-stats-dir "$BASE/tuab_10s/normalization" --classes $CLASSES \
  --max-per-class $CAP --unfreeze all --backbone-lr-mult 0.01 \
  --epochs 30 --patience 8 --batch-size 16 --lr 5e-4 --out-root "$OUT"

echo "===== TUAB 5s + 10s EVAL DONE ====="
