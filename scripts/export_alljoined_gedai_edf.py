"""
export_alljoined_gedai_edf.py
-------------------
Exports Alljoined-1.6M subjects' EEG data in EDF format for professor review:
  - subject_X_before_gedai.edf  (bandpassed input)
  - subject_X_after_gedai.edf   (GEDAI denoised output)
  - subject_X_removed_noise.edf (the noise GEDAI stripped away)

Usage:
    python scripts/export_alljoined_gedai_edf.py
"""
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
from src.denoising.pipelines import bandpass_filter, apply_gedai

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("data/alljoined/processed")
OUT_DIR     = Path("alljoined_gedai")
SUBJECTS    = [1, 2, 3]       
L_FREQ      = 1.0              # standard matching smoke test
H_FREQ      = 40.0             

# ── Setup ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Exporting Alljoined subjects to {OUT_DIR}/")
print(f"Bandpass: {L_FREQ}-{H_FREQ} Hz\n")

for sid in SUBJECTS:
    fname = DATA_ROOT / f"subject_{sid}.npz"
    if not fname.exists():
        print(f"  [Skip] Subject {sid} not found in {DATA_ROOT}")
        continue
        
    print(f"Subject {sid}:")
    with np.load(fname, allow_pickle=True) as data:
        X = data['X']          # (n_trials, n_ch, n_times) in Volts
        sfreq = float(data['sfreq'])
        ch_names = list(data['ch_names'])
        
    n_trials, n_ch, n_times = X.shape
    print(f"  {n_trials} trials, {n_ch} channels @ {sfreq} Hz")

    # ── 1) Baseline (EEG_in) ─────────────────────────────────────────────────
    X_bp = bandpass_filter(X, sfreq, L_FREQ, H_FREQ).astype(np.float32)

    # ── 2) GEDAI (EEG_out) ───────────────────────────────────────────────────
    print(f"  Running GEDAI...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_clean = apply_gedai(X, sfreq, ch_names, L_FREQ, H_FREQ).astype(np.float32)
    print(f"  GEDAI done.")

    # ── 3) Concatenate trials for review ─────────────────────────────────────
    raw_before = X_bp.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times)
    raw_after  = X_clean.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times)
    raw_noise  = raw_before - raw_after

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")

    # ── Exporting ────────────────────────────────────────────────────────────
    for data_arr, suffix in [(raw_before, "before_gedai"), 
                             (raw_after, "after_gedai"), 
                             (raw_noise, "removed_noise")]:
        raw_mne = mne.io.RawArray(data_arr, info, verbose=False)
        out_path = OUT_DIR / f"subject_{sid}_{suffix}.edf"
        mne.export.export_raw(str(out_path), raw_mne, fmt="edf", overwrite=True, verbose=False)
        print(f"  ✓ Saved: {out_path.name}")
    print()

print(f"\nDone! Please send the files in '{OUT_DIR}' to the professor.")
