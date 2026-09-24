#!/usr/bin/env python3
"""Download the Mumtaz2016 MDD/healthy EEG dataset from figshare (article
4244171, 'MDD Patients and Healthy Controls EEG Data (New)').

193 EDF files, ~0.9 GB total. Names look like '<H|MDD> S<subj> <EC|EO|TASK>.edf'.
Files are fetched via the per-file figshare download_url, with a size check and
resume-skip so re-runs don't re-download completed files.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

API = "https://api.figshare.com/v2/articles/4244171"


def fetch_manifest():
    with urllib.request.urlopen(API, timeout=60) as r:
        return json.load(r)["files"]


def download(url, dest, expected_size, retries=4):
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = dest.stat().st_size
            if expected_size and got != expected_size:
                raise IOError(f"size mismatch {got} != {expected_size}")
            return got
        except Exception as e:
            print(f"    attempt {attempt} failed: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"failed to download {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root",
                    default="/scratch/linah03/EpilepticSeizureProject/Dataset/external_raw/mumtaz2016")
    args = ap.parse_args()
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    files = fetch_manifest()
    print(f"manifest: {len(files)} files")
    done = skipped = 0
    total_bytes = 0
    for i, f in enumerate(sorted(files, key=lambda x: x["name"]), 1):
        dest = out / f["name"]
        size = f.get("size")
        if dest.exists() and (not size or dest.stat().st_size == size):
            skipped += 1
            continue
        got = download(f["download_url"], dest, size)
        total_bytes += got
        done += 1
        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {f['name']}  ({got/1e6:.1f} MB)", flush=True)
    print(f"\nDONE: downloaded {done}, skipped {skipped}, "
          f"{total_bytes/1e9:.2f} GB new -> {out}")
    # quick integrity: count by group/condition
    edfs = sorted(out.glob("*.edf"))
    print(f"total EDF on disk: {len(edfs)}")


if __name__ == "__main__":
    main()
