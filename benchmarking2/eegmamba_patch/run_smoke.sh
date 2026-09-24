#!/bin/bash
#SBATCH --account=def-linah03
#SBATCH --job-name=MambaSmoke
#SBATCH --output=/home/abdulh/scratch/benchmarking2/logs/mamba_smoke_%j.out
#SBATCH --error=/home/abdulh/scratch/benchmarking2/logs/mamba_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32000M
module purge || true; module load python/3.11 cuda/12.2 || true
set -euo pipefail
/home/abdulh/venvs/pt-vae-alliance/bin/python -u /home/abdulh/scratch/benchmarking2/EEGMamba/smoke_test.py
