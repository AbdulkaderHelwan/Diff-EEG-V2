#!/bin/bash
# Process TUAB (abnormality) at BOTH 5 s and 10 s windows with the SAME pipeline,
# so the 5 s-vs-10 s comparison is controlled (only window length differs; the
# paper's 0.878 used a collaborator's separate pipeline, so we make our own 5 s
# baseline too). Official patient-disjoint train/eval split; 250->256 Hz.
#SBATCH --job-name=PrepTUABwin
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/PrepTUABwin_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/PrepTUABwin_%j.err
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=84G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
set -euo pipefail

PY="/home/abdulh/venvs/pt-vae-alliance/bin/python"
SC="/home/abdulh/scratch/EEGdiff_V2/dataset_tools/prepare_tuab.py"
BASE="/scratch/linah03/EpilepticSeizureProject/Dataset/external_processed"

echo "########## TUAB 5s ##########"
"$PY" -u "$SC" --out-root "$BASE/tuab_5s"  --window-sec 5  --stride-sec 5

echo "########## TUAB 10s ##########"
"$PY" -u "$SC" --out-root "$BASE/tuab_10s" --window-sec 10 --stride-sec 10

echo "===== TUAB 5s + 10s PROCESSING DONE ====="
