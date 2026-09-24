#!/bin/bash
#   sbatch run_sup.sh <eegnet|conformer|sttransformer> [dataset]
#SBATCH --job-name=BM2sup
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/sup_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/sup_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96000M
module purge || true
module load python/3.11 cuda/12.2 || true
set -euo pipefail
mkdir -p /home/abdulh/scratch/benchmarking2/logs
cd /home/abdulh/scratch/benchmarking2/code
/home/abdulh/venvs/eegdm/bin/python -u supervised_baselines.py --model "${1:?model}" --dataset "${2:-tusz}"
echo "Done: $(date)"
