#!/usr/bin/env python3
"""
Check whether denoising pipelines preserve alpha (8–12 Hz) and beta (13–30 Hz) power.
Compares band power of each pipeline vs bandpass (baseline). Ratio < 1 suggests
possible over-removal of brain signal; ratio ~1 suggests preservation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.denoising.pipelines import bandpass_filter, apply_icalabel, apply_gedai
from src.io.dataset import NpzMotorImageryDataset


def band_power(sig: np.ndarray, sfreq: float, lo: float, hi: float) -> float:
    """Total power in [lo, hi] Hz using Welch."""
    nperseg = min(256, len(sig) // 4, 1024)
    f, p = welch(sig, fs=sfreq, nperseg=nperseg)
    mask = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[mask], f[mask]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check alpha/beta preservation across pipelines.")
    parser.add_argument("--config", default="config/config_real_physionet_5subjects_quick.yml")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--channel", default="C3", help="Channel for PSD (e.g. C3, C4)")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dataset = NpzMotorImageryDataset(
        data_root=cfg.data_root,
        subjects=cfg.subjects,
        float_dtype=cfg.memory.float_dtype,
    )
    subj_data = None
    for sid, data in dataset.iter_subjects():
        if sid == args.subject:
            subj_data = data
            break
    if subj_data is None:
        raise SystemExit(f"Subject {args.subject} not found.")

    X = subj_data.X
    sfreq = subj_data.sfreq
    ch_names = list(subj_data.ch_names)
    ch_upper = [c.upper() for c in ch_names]
    try:
        ch_idx = ch_upper.index(args.channel.upper())
    except ValueError:
        ch_idx = 0
        args.channel = ch_names[0]

    l_freq, h_freq = cfg.bandpass.l_freq, cfg.bandpass.h_freq
    X_bp = bandpass_filter(X, sfreq, l_freq, h_freq)
    # Use trial-averaged for summary; also compute per-trial ratios to catch over-removal.
    sig_bp = X_bp[:, ch_idx, :].mean(axis=0)

    signals = {"bandpass": sig_bp}
    X_ica = X_gd = None
    if cfg.denoising.use_icalabel:
        X_ica = apply_icalabel(X, sfreq, ch_names, l_freq, h_freq)
        signals["icalabel"] = X_ica[:, ch_idx, :].mean(axis=0)
    if cfg.denoising.use_gedai:
        X_gd = apply_gedai(X, sfreq, ch_names, l_freq, h_freq)
        signals["gedai"] = X_gd[:, ch_idx, :].mean(axis=0)

    alpha_band = (8.0, 12.0)
    beta_band = (13.0, 30.0)
    ref_alpha = band_power(sig_bp, sfreq, *alpha_band)
    ref_beta = band_power(sig_bp, sfreq, *beta_band)

    # Per-trial ratios (detect over-removal on any single trial).
    n_trials = X_bp.shape[0]
    min_alpha_ratio = min_beta_ratio = 1.0
    for name, data in [("icalabel", X_ica), ("gedai", X_gd)]:
        if data is None:
            continue
        for t in range(n_trials):
            ref_a_t = band_power(X_bp[t, ch_idx, :], sfreq, *alpha_band)
            ref_b_t = band_power(X_bp[t, ch_idx, :], sfreq, *beta_band)
            a_t = band_power(data[t, ch_idx, :], sfreq, *alpha_band)
            b_t = band_power(data[t, ch_idx, :], sfreq, *beta_band)
            if ref_a_t > 0:
                min_alpha_ratio = min(min_alpha_ratio, a_t / ref_a_t)
            if ref_b_t > 0:
                min_beta_ratio = min(min_beta_ratio, b_t / ref_b_t)

    print(f"\nBrain signal preservation — Subject {args.subject}, channel {args.channel}")
    print("(Band power ratio vs bandpass; 1.0 = preserved, <0.75 = possible over-removal)")
    print("-" * 60)
    print(f"{'Pipeline':<12} {'Alpha (8-12 Hz)':>18} {'Beta (13-30 Hz)':>18}")
    print(f"{'bandpass':<12} {1.0:>18.4f} {1.0:>18.4f}  (reference)")
    for name, sig in signals.items():
        if name == "bandpass":
            continue
        a = band_power(sig, sfreq, *alpha_band)
        b = band_power(sig, sfreq, *beta_band)
        r_alpha = a / ref_alpha if ref_alpha > 0 else float("nan")
        r_beta = b / ref_beta if ref_beta > 0 else float("nan")
        flag = ""
        if r_alpha < 0.75 or r_beta < 0.75:
            flag = "  ⚠ possible over-removal"
        print(f"{name:<12} {r_alpha:>18.4f} {r_beta:>18.4f}{flag}")
    print("-" * 60)
    print("Per-trial minimum ratio (worst trial):")
    print(f"  Alpha: {min_alpha_ratio:.4f}  Beta: {min_beta_ratio:.4f}")
    if min_alpha_ratio < 0.75 or min_beta_ratio < 0.75:
        print("  ⚠ At least one trial shows possible over-removal (ratio < 0.75).")
    else:
        print("  OK: all trials preserve brain bands (ratio >= 0.75).")
    print("-" * 60)
    print("Interpretation: ratio near 1.0 = brain bands preserved; <<0.75 may indicate")
    print("excessive removal (retention guards in pipelines should prevent this).\n")


if __name__ == "__main__":
    main()
