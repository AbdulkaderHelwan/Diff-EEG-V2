#!/bin/bash
# Frozen-feature extraction for BOTH models on one of our datasets.
#   sbatch run_bench.sh tuab5s
#   sbatch run_bench.sh tusz
#SBATCH --job-name=BM2feat
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/feat_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/feat_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000M
#SBATCH --nodes=1
module purge || true
module load python/3.11 cuda/12.2 || true
set -euo pipefail
# S4 needs pykeops for its Cauchy kernel; without it the backbone runs ~4x slower
# and OOMs at modest batch sizes. NOTE: lib64 only -- adding lib64/stubs shadows the
# real driver and breaks NVML.
PYKEOPS_CACHE_FOLDER=/home/abdulh/scratch/benchmarking2/.keops
mkdir -p "$PYKEOPS_CACHE_FOLDER"
DS=${1:-tuab5s}
cd /home/abdulh/scratch/benchmarking2/code
# The CUDA env below is REQUIRED by pykeops but BREAKS the DiffEEG venv's torch
# (mismatched cudacore libcusparse), so it is scoped to this subshell only.
if [ -f "/home/abdulh/scratch/benchmarking2/results/features/eegdm_${DS}_test.npz" ]; then
  echo "===== EEGDM / $DS : features present, skipping ====="
else
echo "===== EEGDM / $DS ====="
(
  export CUDA_PATH="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"
  export CUDA_HOME="$CUDA_PATH"
  export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
  export PYKEOPS_CACHE_FOLDER
  /home/abdulh/venvs/eegdm/bin/python -u extract_features.py --model eegdm --dataset "$DS"
)
fi

echo; echo "===== DiffEEG / $DS ====="
/home/abdulh/venvs/pt-vae-alliance/bin/python -u extract_features.py --model diffeeg --dataset "$DS"
echo; echo "Done: $(date)"
