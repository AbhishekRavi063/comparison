"""
export_weibo_edf.py
-------------------
Exports Weibo2014 EEG data in EDF format for professor review.
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
from pathlib import Path

# Force GEDAI to CPU
os.environ["PYGEDAI_FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
from src.io.dataset import NpzMotorImageryDataset
from src.denoising.pipelines import bandpass_filter, apply_gedai

DATA_ROOT   = Path("data/weibo2014/processed")
OUT_DIR     = Path("exports/gedai_review_weibo")
SUBJECTS    = [1] # Just one subject for a quick representative file
L_FREQ      = 8.0
H_FREQ      = 30.0

# ── Setup ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
dataset = NpzMotorImageryDataset(
    data_root=DATA_ROOT,
    subjects=SUBJECTS,
    float_dtype="float32",
)

for sid in SUBJECTS:
    data = dataset._load_subject_file(sid)
    X = data.X          # (n_trials, n_channels, n_times)
    sfreq = float(data.sfreq)
    ch_names = list(data.ch_names)
    n_trials, n_ch, n_times = X.shape

    # ── Bandpass (= EEG_in) ──────────────────────────────────────────────────
    X_bp = bandpass_filter(X, sfreq, L_FREQ, H_FREQ).astype(np.float32)

    # ── GEDAI (= EEG_out) ────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_clean = apply_gedai(X, sfreq, ch_names, L_FREQ, H_FREQ).astype(np.float32)

    raw_before_v = X_bp.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times) * 1e-6
    raw_after_v  = X_clean.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times) * 1e-6

    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")

    raw_mne_before = mne.io.RawArray(raw_before_v, info, verbose=False)
    out_before = OUT_DIR / f"weibo_subj_{sid:03d}_before_gedai.edf"
    mne.export.export_raw(str(out_before), raw_mne_before, fmt="edf", overwrite=True, verbose=False)

    raw_mne_after = mne.io.RawArray(raw_after_v, info, verbose=False)
    out_after = OUT_DIR / f"weibo_subj_{sid:03d}_after_gedai.edf"
    mne.export.export_raw(str(out_after), raw_mne_after, fmt="edf", overwrite=True, verbose=False)

    raw_noise_v = raw_before_v - raw_after_v
    raw_mne_noise = mne.io.RawArray(raw_noise_v, info, verbose=False)
    out_noise = OUT_DIR / f"weibo_subj_{sid:03d}_removed_noise.edf"
    mne.export.export_raw(str(out_noise), raw_mne_noise, fmt="edf", overwrite=True, verbose=False)

print(f"Exported to {OUT_DIR.resolve()}")
