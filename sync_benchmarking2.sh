#!/bin/bash
# Copy the publishable part of /home/abdulh/scratch/benchmarking2 into this repo.
#
# The live folder stays the working copy (V3 experiments read from it), so this is a
# one-way copy that can be re-run whenever it changes. It takes only our own files:
#
#   code/, logs/                         our scripts and SLURM logs
#   results/**.json, results/norm/       metrics, splits and normalisation stats
#   EEGMamba additions -> eegmamba_patch/  loader, smoke test, triton shim
#
# and leaves out everything that is not ours or not small:
#
#   BIOT/ CBraMod/ EEGDM_ref/ EEGMamba/ LaBraM/   upstream repos, pinned instead in
#                                                  benchmarking2/setup_third_party.sh
#   results/features/*.npz                        18.8 GB of extracted features
#   *.pth                                         trained weights (OneDrive, not git)
#   *.bak*, __pycache__                           editor backups and bytecode
#
# README.md and setup_third_party.sh live only in this repo and are never overwritten.
set -euo pipefail
SRC=/home/abdulh/scratch/benchmarking2
DST="$(cd "$(dirname "$0")" && pwd)/benchmarking2"
mkdir -p "$DST"

rsync -a --prune-empty-dirs \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.bak*' \
  --exclude='*.pth' --exclude='*.npz' --exclude='*.ckpt' \
  --include='/code/***' \
  --include='/logs/***' \
  --include='/results/' --include='/results/**/' \
  --include='/results/**.json' --include='/results/norm/***' \
  --exclude='*' \
  "$SRC/" "$DST/"

mkdir -p "$DST/eegmamba_patch/_deps"
cp "$SRC/EEGMamba/eegmamba_loader.py" "$SRC/EEGMamba/smoke_test.py" \
   "$SRC/EEGMamba/run_smoke.sh" "$DST/eegmamba_patch/"
cp "$SRC/EEGMamba/_deps/_triton_alias.py" "$DST/eegmamba_patch/_deps/"

echo "synced -> $DST"
find "$DST" -type f | wc -l | xargs echo "files:"
du -sh "$DST" | cut -f1 | xargs echo "size:"
