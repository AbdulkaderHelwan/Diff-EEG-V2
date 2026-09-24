#!/usr/bin/env python3
"""Download or print commands for external EEG datasets.

Large public EEG datasets move slowly, but some require credentials or manual
confirmation. This script keeps the source commands in one auditable place.
"""

import argparse
import json
import subprocess
from pathlib import Path


SOURCES = {
    "sleep_edfx": {
        "url": "https://physionet.org/files/sleep-edfx/1.0.0/",
        "command": ["wget", "-r", "-N", "-c", "-np", "https://physionet.org/files/sleep-edfx/1.0.0/"],
        "notes": "Open PhysioNet dataset, about 8.1 GB uncompressed.",
    },
    "siena": {
        "url": "https://physionet.org/files/siena-scalp-eeg/1.0.0/",
        "command": ["wget", "-r", "-N", "-c", "-np", "https://physionet.org/files/siena-scalp-eeg/1.0.0/"],
        "notes": "Open PhysioNet dataset, about 20.3 GB uncompressed.",
    },
    "tuev": {
        "url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
        "command": [
            "rsync", "-auvxL", "-e", "ssh -i ~/.ssh/id_ed25519",
            "nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_events/v2.0.1", ".",
        ],
        "notes": "Requires approved TUH access, registered ssh key, and NEDC rsync credentials.",
    },
    "bonn": {
        "url": "https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/",
        "command": None,
        "notes": "Download SET A-E zip files from the official Bonn page; links are embedded by the site.",
    },
    "bern_barcelona": {
        "url": "https://www.upf.edu/web/ntsa/downloads",
        "command": None,
        "notes": "Use the official UPF downloads page entry: The Bern-Barcelona EEG database.",
    },
    "mindbigdata": {
        "url": "https://mindbigdata.com/opendb/index.html",
        "command": ["wget", "-r", "-N", "-c", "-np", "https://mindbigdata.com/opendb/"],
        "notes": "Open text-format BCI digit EEG dataset; download page documents fields and channels.",
    },
    "eegmmidb": {
        "url": "https://physionet.org/files/eegmmidb/1.0.0/",
        "command": ["wget", "-r", "-N", "-c", "-np", "https://physionet.org/files/eegmmidb/1.0.0/"],
        "notes": "PhysioNet EEG Motor Movement/Imagery, 109 subjects x 14 runs, 64ch/160Hz, ~2.7GB. Multiclass motor imagery.",
    },
    "bci_iv_2a": {
        "url": "https://lampx.tugraz.at/~bci/database/001-2014/",
        "command": None,
        "notes": ("BNCI 001-2014 (BCI Competition IV-2a). 4-class motor imagery, 22ch/250Hz. "
                  "Download A01T..A09T and A01E..A09E .mat from "
                  "https://lampx.tugraz.at/~bci/database/001-2014/AxxT.mat (loop 01-09, T and E)."),
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="/scratch/linah03/EpilepticSeizureProject/Dataset/external_raw")
    p.add_argument("--datasets", nargs="+", default=list(SOURCES))
    p.add_argument("--print", action="store_true", dest="print_only")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name in args.datasets:
        if name not in SOURCES:
            raise SystemExit(f"Unknown dataset '{name}'. Choices: {', '.join(sorted(SOURCES))}")
        src = SOURCES[name]
        out_dir = raw_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest[name] = {**src, "raw_dir": str(out_dir)}

        cmd = src["command"]
        if args.print_only or cmd is None or name == "tuev":
            print(f"\n[{name}] {src['notes']}")
            print(f"  source: {src['url']}")
            if cmd:
                print(f"  command from {out_dir}: {' '.join(cmd)}")
            continue

        print(f"\n[{name}] downloading into {out_dir}")
        subprocess.run(cmd, cwd=out_dir, check=True)

    with open(raw_root / "download_manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)


if __name__ == "__main__":
    main()

