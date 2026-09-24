#!/bin/bash
#   sbatch run_mlp.sh              (all four models)
#   sbatch run_mlp.sh diffeeg,eegdm
#SBATCH --job-name=BM2mlp
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/mlp_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/mlp_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96000M
module purge || true
module load python/3.11 cuda/12.2 || true
set -euo pipefail
mkdir -p /home/abdulh/scratch/benchmarking2/logs
cd /home/abdulh/scratch/benchmarking2/code
/home/abdulh/venvs/eegdm/bin/python -u mlp_head.py --dataset "${1:-tuab5s}" ${2:+--models "$2"}
echo "Done: $(date)"
