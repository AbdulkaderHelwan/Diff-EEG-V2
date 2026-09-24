#!/bin/bash
#SBATCH --account=def-linah03
#SBATCH --job-name=MambaSiena
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/eegmamba_siena_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/eegmamba_siena_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64000M
module purge || true; module load python/3.11 cuda/12.2 || true
set -euo pipefail
cd /home/abdulh/scratch/benchmarking2/code
echo "=== extract (Siena, ~101k segments) ==="
/home/abdulh/venvs/pt-vae-alliance/bin/python -u extract_features.py --model eegmamba --dataset siena
echo "=== probe: eegmamba vs diffeeg (cached features, no other model re-run) ==="
/home/abdulh/venvs/pt-vae-alliance/bin/python -u probe.py --dataset siena --models eegmamba,diffeeg
echo "Done: $(date)"
