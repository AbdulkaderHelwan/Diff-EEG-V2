#!/bin/bash
#SBATCH --job-name=BM2speed
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/speed_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/speed_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64000M
module purge || true; module load python/3.11 cuda/12.2 || true
set -euo pipefail
export CUDA_PATH="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"
export CUDA_HOME="$CUDA_PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"   # NOT stubs -- they shadow the real driver
export PYKEOPS_CACHE_FOLDER=/home/abdulh/scratch/benchmarking2/.keops
mkdir -p "$PYKEOPS_CACHE_FOLDER"
echo "CUDA_PATH=$CUDA_PATH"; ls "$CUDA_PATH/lib64/libnvrtc.so"* 2>/dev/null | head -2; ls "$CUDA_PATH"/lib64/stubs/libcuda.so 2>/dev/null
cd /home/abdulh/scratch/benchmarking2/code
/home/abdulh/venvs/eegdm/bin/python -u bench_speed.py
