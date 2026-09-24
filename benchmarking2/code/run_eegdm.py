#!/usr/bin/env python3
"""Deterministic launcher for EEGDM's main.py.

Why this exists: PyTorch >= 2.6 defaults torch.load(weights_only=True), but EEGDM's
published backbone.ckpt is a Lightning checkpoint pickling omegaconf config objects.
The allowlist route bottoms out at builtins, so weights_only=False is required. Their
code predates the default change, and the checkpoint is the authors' own release from
https://huggingface.co/jhpuah/eegdm.

Lightning calls pl_load(..., weights_only=weights_only) EXPLICITLY, so a setdefault
does not help -- the value has to be forced. A sitecustomize.py proved unreliable
under the SLURM environment, so the patch is applied here, in-process, before
main.py is imported.

Usage mirrors their CLI:
    python run_eegdm.py cache=tuev4
    python run_eegdm.py finetune=tuev4 finetune.rng_seeding.seed=42
"""
import runpy
import sys
from pathlib import Path

import torch

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load
print(f"[launcher] torch {torch.__version__}: forced weights_only=False for EEGDM checkpoints",
      flush=True)

REPO = Path(__file__).resolve().parent.parent / "EEGDM_ref"
sys.path.insert(0, str(REPO))

# hydra reads sys.argv; present it exactly as `python main.py <overrides>`
sys.argv = [str(REPO / "main.py")] + sys.argv[1:]
runpy.run_path(str(REPO / "main.py"), run_name="__main__")
