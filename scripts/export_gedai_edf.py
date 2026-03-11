"""
export_gedai_edf.py
-------------------
Exports 3 PhysioNet subjects' EEG data in EDF format:
  - subject_X_before_gedai.edf  (bandpassed input to GEDAI)
  - subject_X_after_gedai.edf   (GEDAI denoised output)

Usage:
    cd /Users/abhishekr/Documents/EEG/comparison
    python scripts/export_gedai_edf.py

Output: exports/gedai_review/
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

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("data/physionet_eegbci")
OUT_DIR     = Path("exports/gedai_review")
SUBJECTS    = [1, 2, 3]       # 3 subjects for professor review
L_FREQ      = 8.0              # bandpass low
H_FREQ      = 30.0             # bandpass high
MAX_TRIALS  = None             # None = all trials

# ── Setup ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
dataset = NpzMotorImageryDataset(
    data_root=DATA_ROOT,
    subjects=SUBJECTS,
    float_dtype="float32",
)

print(f"Exporting {len(SUBJECTS)} subjects to {OUT_DIR}/")
print(f"Bandpass: {L_FREQ}–{H_FREQ} Hz\n")

for sid in SUBJECTS:
    print(f"Subject {sid}:")
    data = dataset._load_subject_file(sid)
    X = data.X          # (n_trials, n_channels, n_times)
    sfreq = float(data.sfreq)
    ch_names = list(data.ch_names)
    n_trials, n_ch, n_times = X.shape
    print(f"  {n_trials} trials, {n_ch} channels, {n_times} samples @ {sfreq} Hz")

    # ── Bandpass (= EEG_in) ──────────────────────────────────────────────────
    # This is what the classifier sees WITHOUT any denoising.
    X_bp = bandpass_filter(X, sfreq, L_FREQ, H_FREQ).astype(np.float32)

    # ── GEDAI (= EEG_out) ────────────────────────────────────────────────────
    # IMPORTANT: apply_gedai must receive BROADBAND raw X (not pre-bandpassed).
    # It internally runs wavelet denoising on the full spectrum, then bandpasses.
    # This matches exactly what happens during the classification benchmarks.
    print(f"  Running GEDAI on broadband signal...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_clean = apply_gedai(X, sfreq, ch_names, L_FREQ, H_FREQ).astype(np.float32)
    print(f"  GEDAI done.")


    # ── Concatenate trials into one continuous signal ─────────────────────────
    # Shape: (n_ch, n_trials * n_times)  — standard EDF/MNE layout
    raw_before = X_bp.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times)
    raw_after  = X_clean.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times)

    # EDF units: MNE expects Volts, PhysioNet data is in microvolts
    # Scale to Volts for EDF standard
    raw_before_v = raw_before * 1e-6
    raw_after_v  = raw_after  * 1e-6

    info = mne.create_info(
        ch_names=list(ch_names),
        sfreq=sfreq,
        ch_types="eeg",
    )
    info.set_montage("standard_1020", on_missing="ignore")

    # ── Export before ─────────────────────────────────────────────────────────
    raw_mne_before = mne.io.RawArray(raw_before_v, info, verbose=False)
    out_before = OUT_DIR / f"subject_{sid:03d}_before_gedai.edf"
    mne.export.export_raw(str(out_before), raw_mne_before, fmt="edf", overwrite=True, verbose=False)
    print(f"  Saved: {out_before.name}")

    # ── Export after ──────────────────────────────────────────────────────────
    raw_mne_after = mne.io.RawArray(raw_after_v, info, verbose=False)
    out_after = OUT_DIR / f"subject_{sid:03d}_after_gedai.edf"
    mne.export.export_raw(str(out_after), raw_mne_after, fmt="edf", overwrite=True, verbose=False)
    print(f"  Saved: {out_after.name}")

    # ── Also export the removed noise (before - after) ────────────────────────
    raw_noise_v = raw_before_v - raw_after_v
    raw_mne_noise = mne.io.RawArray(raw_noise_v, info, verbose=False)
    out_noise = OUT_DIR / f"subject_{sid:03d}_removed_noise.edf"
    mne.export.export_raw(str(out_noise), raw_mne_noise, fmt="edf", overwrite=True, verbose=False)
    print(f"  Saved: {out_noise.name}  (= before - after, i.e. what GEDAI removed)")
    print()

print("=" * 60)
print(f"Done! Files exported to:  {OUT_DIR.resolve()}")
print()
print("Send these 9 files to your professor (3 subjects × 3 files):")
for sid in SUBJECTS:
    print(f"  subject_{sid:03d}_before_gedai.edf   — raw bandpassed signal")
    print(f"  subject_{sid:03d}_after_gedai.edf    — GEDAI denoised signal")
    print(f"  subject_{sid:03d}_removed_noise.edf  — what GEDAI removed (before-after)")
print()
print("In EEGlab: File > Import Data > Using EEG Functions > From EDF/BDF Files")
print("In MNE: mne.io.read_raw_edf('subject_001_before_gedai.edf')")
