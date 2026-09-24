#!/bin/bash
# Evaluate the model on external datasets: Bonn (binary, pretrained +RL head)
# and BCI IV-2a (multiclass, frozen embedding + fresh head). Results ->
# EEGdiff_V2/Benchmarking/.
#SBATCH --job-name=EEGdiff_V2_ext_eval
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/ExtEval_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/ExtEval_%j.err
#SBATCH --time=00:30:00
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

echo "===== Bonn (binary) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_binary.py" \
  --processed-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/bonn \
  --ckpt /home/abdulh/scratch/EEGdiff_V2/Binary_finetune_unfrozen/run_last_two_levels_20260630_083121/best_classifier.pth \
  --out-root "$OUT/bonn_binary"

echo
echo "===== BCI IV-2a (multiclass) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/bci_iv_2a \
  --dataset-name bci_iv_2a \
  --out-root "$OUT"
