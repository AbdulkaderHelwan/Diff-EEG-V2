#!/bin/bash
#   sbatch run_probe.sh tuab5s
#SBATCH --job-name=BM2probe
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/probe_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/probe_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000M
module purge || true
module load python/3.11 || true
set -euo pipefail
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/abdulh/scratch/benchmarking2/code
/home/abdulh/venvs/eegdm/bin/python -u probe.py --dataset "${1:-tuab5s}" ${2:+--models "$2"}
echo "Done: $(date)"
