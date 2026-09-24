#!/bin/bash
# Zero-shot / parameter-free kNN probe (5-fold CV balanced accuracy) of the frozen
# DiffEEG embedding. Same knn_sep protocol as the t-SNE panels (Fig. 5), so the
# numbers are consistent paper-wide. Bonn/Siena/TUEV are already in interp.npz
# (0.775 / 0.615 / 0.328); this run covers the remaining three datasets.
# Memory-bounded via --max-per-class; TUAB/TUSZ use the patient-wise pipeline.
#SBATCH --job-name=EEGdiff_zsknn
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/ZSkNN_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/ZSkNN_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true
set -euo pipefail

PY="/home/abdulh/venvs/pt-vae-alliance/bin/python"
DIR="/home/abdulh/scratch/EEGdiff_V2/finetuning"
EP="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed"
DEFNORM="/home/abdulh/scratch/EEGdiff_V2/training_diffusion_v2/normalization"

run() { echo; echo "########## $1 ##########"; "$PY" -u "$DIR/zeroshot_knn.py" "${@:2}"; }

run eegmmidb   --processed-root "$EP/eegmmidb"   --dataset-name eegmmidb   --norm-stats-dir "$DEFNORM" --classes left_fist right_fist
run sleep_edfx --processed-root "$EP/sleep_edfx" --dataset-name sleep_edfx --norm-stats-dir "$DEFNORM" --classes wake n1 n2 n3 n4 rem
run mumtaz     --processed-root "$EP/mumtaz2016_matched" --dataset-name mumtaz --norm-stats-dir "$EP/mumtaz2016_matched/normalization" --classes healthy mdd

echo; echo "===== ZERO-SHOT kNN (eegmmidb, sleep, mumtaz) DONE ====="
