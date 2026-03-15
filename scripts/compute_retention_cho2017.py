#!/usr/bin/env python3
"""Compute alpha/beta retention ratios for Cho2017.

For 10 subjects, compares bandpass baseline vs GEDAI (our fix and professor fix)
using the `_retention_ratios` metric from `src.denoising.pipelines`.

Usage:
    cd /Users/abhishekr/Documents/EEG/comparison
    python3 scripts/compute_retention_cho2017.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.dataset import NpzMotorImageryDataset
from src.denoising.pipelines import (
    bandpass_filter,
    apply_gedai,
    _retention_ratios,
    ALPHA_BAND,
    BETA_BAND,
)

DATA_ROOT = PROJECT_ROOT / "data" / "cho2017" / "processed"
SUBJECTS = list(range(1, 11))
L_FREQ = 8.0
H_FREQ = 30.0

GEDAI_OFFICIAL = PROJECT_ROOT / "gedai_official"
GEDAI_PROFESSOR = PROJECT_ROOT / "gedai_professor_fix"


def compute_subject_retention(x_bp: np.ndarray, x_gd: np.ndarray, sfreq: float, ch_names: List[str]) -> float:
    """Return effective retention ratio for one subject.

    Ratios >= 1 mean GEDAI increased band power; < 1 means some loss.
    """
    return _retention_ratios(
        x_clean=x_gd,
        x_band_ref=x_bp,
        sfreq=sfreq,
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        ch_names=ch_names,
    )


def run_engine(engine_path: Path, label: str) -> Dict[int, float]:
    print(f"\n=== Engine: {label} ===")
    os.environ["GEDAI_LIBRARY_PATH"] = str(engine_path.resolve())
    os.environ.setdefault("PYGEDAI_FORCE_CPU", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    dataset = NpzMotorImageryDataset(DATA_ROOT, subjects=SUBJECTS)
    retention: Dict[int, float] = {}

    for sid in SUBJECTS:
        subj = dataset._load_subject_file(sid)
        X = subj.X.astype("float32", copy=False)
        sfreq = float(subj.sfreq)
        ch_names = list(subj.ch_names)

        # Bandpass baseline (reference brain signal)
        X_bp = bandpass_filter(X, sfreq, L_FREQ, H_FREQ)

        # GEDAI-cleaned data for this engine
        X_gd = apply_gedai(X, sfreq, ch_names, L_FREQ, H_FREQ)

        r = compute_subject_retention(X_bp, X_gd, sfreq, ch_names)
        retention[sid] = r
        print(f"  Subject {sid:2d}: retention = {r:.3f}")

    vals = np.array(list(retention.values()))
    print(f"  -> Mean retention ({label}): {vals.mean():.3f} ± {vals.std():.3f}")
    return retention


def main() -> None:
    if not DATA_ROOT.exists():
        raise SystemExit(f"Cho2017 processed data not found at {DATA_ROOT} — run download_cho2017_10subjects.py first.")

    print("Computing retention ratios on Cho2017 (10 subjects, 8–30 Hz band)...")

    # Our bypass fix
    run_engine(GEDAI_OFFICIAL, "Our Bypass Fix")

    # Professor threshold fix
    run_engine(GEDAI_PROFESSOR, "Professor Threshold Fix")


if __name__ == "__main__":
    main()
