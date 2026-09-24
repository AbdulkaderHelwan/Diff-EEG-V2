#!/bin/bash
# "Trained classifier" experiment: frozen EEGdiff V2 backbone + a FRESHLY
# TRAINED head on each external dataset's own train/test split (Bonn binary,
# BCI IV-2a 4-class motor imagery). Contrast with eval_external_datasets.sh,
# which zero-shot reuses the THUSZ-trained seizure classifier with NO
# adaptation -- keep both, they answer different questions:
#   zero-shot     -> does the THUSZ seizure classifier transfer as-is?
#   trained-head  -> are the frozen embeddings themselves useful, given a
#                    task-specific head trained on that dataset's own labels?
# Results -> EEGdiff_V2/Benchmarking/<dataset>_trainedhead_<ts>/.
#SBATCH --job-name=EEGdiff_V2_ext_trainedhead
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/ExtTrainedHead_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/ExtTrainedHead_%j.err
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

echo "===== Bonn (trained head, binary) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/bonn \
  --dataset-name bonn \
  --out-root "$OUT"

echo
echo "===== BCI IV-2a (trained head, 4-class) ====="
"$VENV_PYTHON" -u "$DIR/eval_processed_multiclass.py" \
  --processed-root /scratch/linah03/EpilepticSeizureProject/Dataset/external_processed/bci_iv_2a \
  --dataset-name bci_iv_2a \
  --out-root "$OUT"
