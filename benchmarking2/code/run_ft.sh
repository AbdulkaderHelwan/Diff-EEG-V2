#!/bin/bash
#   sbatch run_ft.sh on    |    sbatch run_ft.sh off
#SBATCH --job-name=BM2ft
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/ft_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96000M
module purge || true
module load python/3.11 cuda/12.2 || true
set -euo pipefail
mkdir -p /home/abdulh/scratch/benchmarking2/logs
cd /home/abdulh/scratch/benchmarking2/code
/home/abdulh/venvs/pt-vae-alliance/bin/python -u finetune_diffeeg.py --rl "${1:-on}" --dataset "${2:-tuab5s}"
echo "Done: $(date)"
