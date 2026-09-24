#!/usr/bin/env python3
"""Validate the released EEGMamba before spending a probe run on it."""
import sys
sys.path.insert(0, "/home/abdulh/scratch/benchmarking2/EEGMamba")
import torch
from eegmamba_loader import load_eegmamba, embed

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| torch", torch.__version__, flush=True)
m = load_eegmamba(dev)

for n_patch, label in [(5, "Bonn / 5 s"), (10, "TUAB / 10 s")]:
    x = torch.randn(2, 16, n_patch, 200, device=dev)
    out = m(x); z = embed(m, x)
    print(f"{label:14s} in {tuple(x.shape)} -> out {tuple(out.shape)} -> embed {tuple(z.shape)}")

x = torch.randn(4, 16, 5, 200, device=dev)
a, b = embed(m, x), embed(m, x)
print("deterministic:", torch.equal(a, b), "| max|a-b| =", (a - b).abs().max().item())
print("embed finite:", bool(torch.isfinite(a).all()), "| std across batch =", a.std(0).mean().item())
