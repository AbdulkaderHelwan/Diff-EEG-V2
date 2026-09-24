#!/bin/bash
#SBATCH --job-name=BM2labram
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/labram_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/labram_%j.err
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96000M
module purge || true; module load python/3.11 cuda/12.2 || true
set -euo pipefail
cd /home/abdulh/scratch/benchmarking2/code
/home/abdulh/venvs/eegdm/bin/python -u extract_features.py --model labram --dataset "${1:-tuab5s}"
echo "Done: $(date)"
