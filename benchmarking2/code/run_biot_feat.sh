#!/bin/bash
# BIOT frozen-feature extraction. Short walltime to fit around the maintenance window.
#SBATCH --job-name=BM2biot
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/biot_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/biot_%j.err
#SBATCH --time=01:40:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96000M
module purge || true; module load python/3.11 cuda/12.2 || true
set -euo pipefail
cd /home/abdulh/scratch/benchmarking2/code
/home/abdulh/venvs/eegdm/bin/python -u extract_features.py --model biot --dataset "${1:-tuab5s}"
echo "Done: $(date)"
