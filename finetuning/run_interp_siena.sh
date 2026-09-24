#!/bin/bash
#SBATCH --job-name=EEGdiff_interp_siena
#SBATCH --output=/home/abdulh/scratch/EEGdiff_V2/logs/InterpSiena_%j.out
#SBATCH --error=/home/abdulh/scratch/EEGdiff_V2/logs/InterpSiena_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge || true
module load python/3.11 || true
module load cuda || true
set -euo pipefail

"/home/abdulh/venvs/pt-vae-alliance/bin/python" -u \
  /home/abdulh/scratch/EEGdiff_V2/finetuning/interpretability_siena.py
