#!/bin/bash
#   sbatch run_feat.sh <model> <dataset>
#SBATCH --job-name=BM2feat
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/feat_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/feat_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000M
module purge || true
module load python/3.11 cuda/12.2 || true
set -euo pipefail
M=${1:?model}; DS=${2:?dataset}
mkdir -p /home/abdulh/scratch/benchmarking2/logs
cd /home/abdulh/scratch/benchmarking2/code
if [ -f "/home/abdulh/scratch/benchmarking2/results/features/${M}_${DS}_test.npz" ]; then
  echo "===== $M / $DS : features present, skipping ====="; exit 0
fi
echo "===== $M / $DS ====="
if [ "$M" = "eegdm" ]; then
  # S4 needs pykeops. lib64 ONLY -- lib64/stubs shadows the real driver and breaks NVML.
  # Scoped to a subshell: this CUDA env breaks the DiffEEG venv's torch.
  (
    export CUDA_PATH="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"
    export CUDA_HOME="$CUDA_PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
    export PYKEOPS_CACHE_FOLDER=/home/abdulh/scratch/benchmarking2/.keops
    mkdir -p "$PYKEOPS_CACHE_FOLDER"
    /home/abdulh/venvs/eegdm/bin/python -u extract_features.py --model eegdm --dataset "$DS"
  )
elif [[ "$M" == diffeeg* ]]; then
  /home/abdulh/venvs/pt-vae-alliance/bin/python -u extract_features.py --model "$M" --dataset "$DS"
else
  /home/abdulh/venvs/eegdm/bin/python -u extract_features.py --model "$M" --dataset "$DS"
fi
echo "Done: $(date)"
