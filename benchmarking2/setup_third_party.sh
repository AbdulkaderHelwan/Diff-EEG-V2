#!/bin/bash
# Fetch the third-party models this benchmark runs, pinned to the commits used in our
# experiments, then install our EEGMamba additions into the EEGMamba checkout.
#
# The upstream code is not copied into this repository: it is other people's work under
# their own licences, and pinning a commit reproduces it exactly without redistributing it.
set -euo pipefail
cd "$(dirname "$0")"

clone() {
  local name=$1 url=$2 commit=$3
  [ -d "$name/.git" ] || git clone --quiet "$url" "$name"
  git -C "$name" fetch --quiet origin "$commit" 2>/dev/null || true
  git -C "$name" checkout --quiet "$commit"
  echo "  $name @ $(git -C "$name" rev-parse --short HEAD)"
}

echo "cloning pinned upstream repositories:"
clone BIOT      https://github.com/ycq091044/BIOT.git         d138e32634e52ae9fa6ec98ac9c4087b14ca869a
clone EEGDM_ref https://github.com/jhpuah/EEGDM.git           003b5a2a016b2f02788ea5693d8b7cd00b6bd969
clone LaBraM    https://github.com/935963004/LaBraM.git       c431221e6cfd23dbfa9950e0180682fb322b0548
clone EEGMamba  https://github.com/wjq-learning/EEGMamba.git  dbc83fa072744201e8897aeb9f65007b952ad323
# The CBraMod copy used in our experiments was downloaded without git history, so its
# exact commit is unknown. This pins upstream HEAD as of 2026-09-24.
clone CBraMod   https://github.com/wjq-learning/CBraMod.git   b9e961003214326972c567eff390e75b0287e32a

echo "installing our EEGMamba additions:"
mkdir -p EEGMamba/_deps
cp eegmamba_patch/eegmamba_loader.py eegmamba_patch/smoke_test.py eegmamba_patch/run_smoke.sh EEGMamba/
cp eegmamba_patch/_deps/_triton_alias.py EEGMamba/_deps/
echo "  eegmamba_loader.py, smoke_test.py, run_smoke.sh, _deps/_triton_alias.py"

cat <<'EOF'

Pretrained weights still to download (not redistributed here):
  EEGMamba  https://huggingface.co/weighting666/EEGMamba -> EEGMamba/pretrained_weights/pretrained_EEGMamba.pth
  EEGDM     https://huggingface.co/jhpuah/eegdm         -> EEGDM_ref/checkpoint/pretrain/backbone.ckpt
  CBraMod   https://huggingface.co/weighting666/CBraMod -> CBraMod/pretrained_weights/pretrained_weights.pth
  BIOT and LaBraM ship their weights in their repositories.
EOF
