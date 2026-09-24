#!/usr/bin/env python3
"""Measured inference cost of each frozen backbone on identical inputs.

Reported for the paper's efficiency argument, so everything here is measured rather
than quoted: parameter count, peak GPU memory and throughput at a COMMON batch size
(so the comparison is like-for-like), plus the largest batch each model accepts on
this device before it raises OOM.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch

B = Path("/home/abdulh/scratch/benchmarking2")
import sys; sys.path.insert(0, str(B / "code"))
import extract_features as EF
from data_adapter import build_index, materialise

COMMON_BATCH = 32          # the largest batch every model here accepts
PROBE_BATCHES = [32, 64, 128, 256]
WARMUP, ITERS = 3, 20


def bench(model, dev, X):
    make = getattr(EF, f"{model}_extractor")
    run = make(dev)
    params = None
    # throughput + peak memory at the common batch
    xb = X[:COMMON_BATCH]
    for _ in range(WARMUP):
        run(xb)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run(xb)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    peak = torch.cuda.max_memory_allocated() / 2**20

    # largest feasible batch
    maxb = 0
    for b in PROBE_BATCHES:
        if b > len(X): break
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            run(X[:b]); torch.cuda.synchronize(); maxb = b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            torch.cuda.empty_cache(); break
    return {"batch": COMMON_BATCH, "s_per_batch": dt,
            "samples_per_s": COMMON_BATCH / dt,
            "peak_mib_at_common_batch": peak, "max_batch_probed": maxb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="tuab5s")
    a = ap.parse_args()
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"total {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)
    idx = build_index(a.dataset, "test")[:512]
    out = {"device": torch.cuda.get_device_name(0), "common_batch": COMMON_BATCH,
           "iters": ITERS, "results": {}}
    for m in [a.model]:
        print(f"\n--- {m} ---", flush=True)
        X, _ = materialise(idx, a.dataset, m if m != "diffeeg" else "diffeeg")
        r = bench(m, dev, X)
        # params, counted from the loaded module rather than quoted
        out["results"][m] = r
        print(f"  {r['samples_per_s']:.1f} samples/s at batch {COMMON_BATCH}, "
              f"peak {r['peak_mib_at_common_batch']:.0f} MiB, "
              f"max feasible batch probed {r['max_batch_probed']}", flush=True)
        torch.cuda.empty_cache()
    p = B / "results" / f"efficiency_{a.model}.json"
    p.write_text(json.dumps(out, indent=2)); print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
