#!/usr/bin/env python3
"""Load the released EEGMamba backbone, frozen, ready for feature extraction.

Kept in one place so the smoke test and the probe extractor build the model
identically -- a divergence between them would mean the thing we validated is not
the thing we benchmark.
"""
import sys

_ROOT = "/home/abdulh/scratch/benchmarking2/EEGMamba"
for _p in (f"{_ROOT}/_deps", _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _triton_alias  # noqa: F401,E402  -- triton alias + mamba_ssm import shims
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

CKPT = f"{_ROOT}/pretrained_weights/pretrained_EEGMamba.pth"
ARCH = dict(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
            seq_len=30, n_layer=12, nhead=8)


def load_eegmamba(device="cuda", verbose=True):
    """Frozen EEGMamba backbone emitting (B, ch, patches, 200) representations."""
    from models.eegmamba import EEGMamba

    model = EEGMamba(**ARCH)
    sd = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"EEGMamba checkpoint mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys. Refusing to extract features from a "
            "partially initialised backbone."
        )

    # Their downstream models drop the projection head and read the encoder output.
    model.proj_out = nn.Identity()

    # The fused Mamba2 kernel calls causal_conv1d_cuda unconditionally, and that
    # extension will not load against this torch build. The unfused path computes the
    # same thing -- Triton chunk-scan plus the module's own nn.Conv1d -- and is what
    # mamba-ssm falls back to when causal_conv1d is absent. Force it everywhere.
    n_switched = 0
    for m in model.modules():
        if hasattr(m, "use_mem_eff_path") and m.use_mem_eff_path:
            m.use_mem_eff_path = False
            n_switched += 1

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"EEGMamba loaded: {n:,} params | unfused path on {n_switched} Mamba2 blocks",
              flush=True)
    return model


@torch.no_grad()
def embed(model, x):
    """(B, ch, patches, 200) -> (B, 200), average-pooled over channels and patches.

    This is their own `avgpooling_patch_reps` readout (models/model_for_tuab.py),
    and it matches CBraMod's 200-d descriptor, so the two are directly comparable.
    """
    return model(x).mean(dim=(1, 2))
