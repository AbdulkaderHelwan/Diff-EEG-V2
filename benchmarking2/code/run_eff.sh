#!/bin/bash
#   sbatch run_eff.sh diffeeg   |   sbatch run_eff.sh eegdm
#SBATCH --job-name=BM2eff
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/eff_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/eff_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64000M
module purge || true
module load python/3.11 cuda/12.2 || true
set -euo pipefail
M=${1:?model}
cd /home/abdulh/scratch/benchmarking2/code
if [ "$M" = "eegdm" ]; then
  # pykeops CUDA env, scoped: it breaks the DiffEEG venv's torch if it leaks
  (
    export CUDA_PATH="${CUDA_HOME:-${EBROOTCUDA:-/usr/local/cuda}}"
    export CUDA_HOME="$CUDA_PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
    export PYKEOPS_CACHE_FOLDER=/home/abdulh/scratch/benchmarking2/.keops
    /home/abdulh/venvs/eegdm/bin/python -u efficiency.py --model eegdm
  )
elif [ "$M" = "diffeeg" ]; then
  /home/abdulh/venvs/pt-vae-alliance/bin/python -u efficiency.py --model diffeeg
else
  /home/abdulh/venvs/eegdm/bin/python -u efficiency.py --model "$M"
fi
echo "Done: $(date)"
