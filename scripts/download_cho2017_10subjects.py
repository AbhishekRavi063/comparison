#!/usr/bin/env python3
"""
Download and prepare 10 subjects from Cho2017 (GigaDB) for the comparison pipeline.

Writes .npz files to data/cho2017/processed/subject_1.npz ... subject_10.npz.
Uses the same format as the main pipeline (X, y, sfreq, ch_names).

Usage:
    cd /path/to/comparison
    python scripts/download_cho2017_10subjects.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
mne_data = PROJECT_ROOT / ".mne_home" / "MNE-data"
mne_data.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MNE_DATA", str(mne_data))

OUT_ROOT = PROJECT_ROOT / "data" / "cho2017" / "processed"
N_SUBJECTS = 10


def main() -> None:
    from src.data.prepare_cho2017 import _prepare_subject

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    subjects = list(range(1, N_SUBJECTS + 1))

    for sid in subjects:
        npz_path = OUT_ROOT / f"subject_{sid}.npz"
        if npz_path.exists():
            print(f"Cho2017 subject {sid} already exists — skipping.", flush=True)
            continue
        print(f"Downloading Cho2017 subject {sid}...", flush=True)
        try:
            info = _prepare_subject(subject=sid, out_root=OUT_ROOT)
            print(
                f"  ✓ subj={sid} | trials={info['n_trials']} | ch={info['n_channels']} "
                f"| times={info['n_times']} | sfreq={info['sfreq']} Hz",
                flush=True,
            )
        except Exception as e:
            print(f"  ✗ subject {sid} failed: {e}", flush=True)
            raise

    print(f"\nDone. Processed data in {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
