"""
gedai_smoke_test.py
-------------------
An isolated, clean smoke test based on the OFFICIAL GEDAI tutorial
(gedai_official/tutorials/00_gedai.py) adapted to use our Weibo2014 data.

This lets us verify the CORRECT usage pattern vs our pipeline's setup,
and check the signal preservation ratios under proper conditions.

Key differences from our current apply_gedai():
  1. Average reference is applied BEFORE fit_raw (as shown in official tutorial line 35)
  2. Single-step GEDAI (not two-step broadband+spectral)
  3. noise_multiplier=3.0 (official tutorial default)
"""
from __future__ import annotations
import os, sys
import warnings
import numpy as np
from pathlib import Path

os.environ["PYGEDAI_FORCE_CPU"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
from gedai import Gedai
from scipy.signal import welch

DATA_PATH = Path("data/weibo2014/processed/subject_1.npz")
L_FREQ, H_FREQ = 8.0, 30.0

def bandpow(signal, sfreq, fmin, fmax):
    """Welch-based bandpower for a 1D signal."""
    freqs, psd = welch(signal, fs=sfreq, nperseg=min(256, len(signal)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapz(psd[mask], freqs[mask]))

print("=" * 60)
print("GEDAI Official Smoke Test (Weibo2014 Subject 1)")
print("=" * 60)

data = np.load(DATA_PATH, allow_pickle=True)
X = data["X"].astype(np.float32)
sfreq = float(data["sfreq"])
ch_names = list(data["ch_names"])
n_trials, n_ch, n_times = X.shape
print(f"Loaded: {n_trials} trials, {n_ch} channels, {n_times} samples @ {sfreq}Hz")

# ── Drop unmontaged channels (same as our pipeline fix) ──────────────────────
info_full = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
info_full.set_montage(mne.channels.make_standard_montage("standard_1020"),
                      on_missing="ignore", verbose=False)
no_pos = [ch["ch_name"] for ch in info_full["chs"] if np.isnan(ch["loc"][:3]).any()]
kept_idx = [i for i in range(n_ch) if ch_names[i] not in no_pos]
ch_kept = [ch_names[i] for i in kept_idx]
X_kept = X[:, kept_idx, :]  # (n_trials, n_kept, n_times)
print(f"Channels kept after montage filter: {len(kept_idx)} (dropped: {no_pos})")

# ── Build RawArray from concatenated trials ───────────────────────────────────
n_kept = len(kept_idx)
X_concat = X_kept.transpose(1, 0, 2).reshape(n_kept, n_trials * n_times)
info = mne.pick_info(info_full, kept_idx)
raw = mne.io.RawArray(X_concat, info, verbose=False)

# ── STEP 1 (OFFICIAL): Average reference BEFORE GEDAI ────────────────────────
print("\n[Official pattern] Applying average reference BEFORE fitting GEDAI...")
raw.set_eeg_reference("average", projection=False, verbose=False)

# ── STEP 2: Fit and transform
print("[Official pattern] Fitting GEDAI (noise_multiplier=6.0)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gedai = Gedai(wavelet_type="haar", wavelet_level=0)
    gedai.fit_raw(raw, noise_multiplier=6.0, n_jobs=1, verbose=False)
    raw_clean = gedai.transform_raw(raw, n_jobs=1, verbose=False)
print("GEDAI transform done.")

# ── Band-power preservation on C3 ────────────────────────────────────────────
c3_idx = ch_kept.index("C3") if "C3" in ch_kept else 0
raw_arr = raw.get_data()
clean_arr = raw_clean.get_data()

# Per-trial band power
alpha_ratios, beta_ratios = [], []
for t in range(n_trials):
    s = t * n_times
    e = s + n_times
    before_a = bandpow(raw_arr[c3_idx, s:e], sfreq, 8, 12)
    before_b = bandpow(raw_arr[c3_idx, s:e], sfreq, 13, 30)
    after_a  = bandpow(clean_arr[c3_idx, s:e], sfreq, 8, 12)
    after_b  = bandpow(clean_arr[c3_idx, s:e], sfreq, 13, 30)
    if before_a > 0: alpha_ratios.append(after_a / before_a)
    if before_b > 0: beta_ratios.append(after_b / before_b)

print(f"\nBrain signal preservation — channel {ch_kept[c3_idx]}")
print(f"(Ratio vs pre-GEDAI; 1.0 = fully preserved, <0.75 = over-removal)")
print("-" * 60)
print(f"  Mean Alpha (8-12 Hz) retention:  {np.mean(alpha_ratios):.4f}")
print(f"  Mean Beta  (13-30 Hz) retention: {np.mean(beta_ratios):.4f}")
print(f"  Min Alpha over all trials:        {np.min(alpha_ratios):.4f}")
print(f"  Min Beta  over all trials:        {np.min(beta_ratios):.4f}")

if np.mean(alpha_ratios) > 0.75 and np.mean(beta_ratios) > 0.75:
    print("\n✓ Brain signal well-preserved (ratios > 0.75)")
else:
    print("\n⚠ Possible over-removal detected (ratios < 0.75)")
print("=" * 60)
