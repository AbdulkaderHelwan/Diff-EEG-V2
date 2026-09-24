#!/bin/bash
# DiffEEG frozen-feature extraction only. Short walltime so it fits before the
# MAINT20260902 reservation starts at 09:00.
#SBATCH --job-name=BM2dfeat
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/dfeat_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/dfeat_%j.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96000M
module purge || true; module load python/3.11 cuda/12.2 || true
set -euo pipefail
cd /home/abdulh/scratch/benchmarking2/code
/home/abdulh/venvs/pt-vae-alliance/bin/python -u extract_features.py --model diffeeg --dataset "${1:-tuab5s}"
echo "Done: $(date)"
