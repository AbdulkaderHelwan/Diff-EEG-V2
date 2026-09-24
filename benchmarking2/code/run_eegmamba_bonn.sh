#!/bin/bash
#SBATCH --account=def-linah03
#SBATCH --job-name=MambaBonn
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/eegmamba_bonn_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/eegmamba_bonn_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48000M
module purge || true; module load python/3.11 cuda/12.2 || true
set -euo pipefail
cd /home/abdulh/scratch/benchmarking2/code
echo "=== extract ==="
/home/abdulh/venvs/pt-vae-alliance/bin/python -u extract_features.py --model eegmamba --dataset bonn
echo "=== probe (eegmamba only) ==="
/home/abdulh/venvs/pt-vae-alliance/bin/python -u probe.py --dataset bonn --models eegmamba,diffeeg
echo "Done: $(date)"
