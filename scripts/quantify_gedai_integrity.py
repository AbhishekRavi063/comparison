import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["PYGEDAI_FORCE_CPU"] = "1"
sys.path.insert(0, "/Users/abhishekr/Documents/EEG/comparison/gedai_official")

import mne
from gedai.gedai.gedai import Gedai
from scipy.signal import welch

def measure_integrity(subject_id):
    data_path = Path(f"/Users/abhishekr/Documents/EEG/comparison/data/weibo2014/processed/subject_{subject_id}.npz")
    data = np.load(data_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    sfreq = float(data["sfreq"])
    ch_names = list(data["ch_names"])
    n_trials, n_ch, n_times = X.shape
    
    info_full = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info_full.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore", verbose=False)
    no_pos_chs = [ch["ch_name"] for ch in info_full["chs"] if np.isnan(ch["loc"][:3]).any()]
    kept_idx = [i for i in range(n_ch) if ch_names[i] not in no_pos_chs]
    X_kept = X[:, kept_idx, :]
    ch_kept = [ch_names[i] for i in kept_idx]
    
    X_concat = X_kept.transpose(1, 0, 2).reshape(len(kept_idx), n_trials * n_times)
    info = mne.pick_info(info_full, kept_idx)
    raw = mne.io.RawArray(X_concat, info, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    
    # Run Fixed GEDAI
    gedai = Gedai(wavelet_type="haar", wavelet_level=4)
    gedai.fit_raw(raw, noise_multiplier=3.0, n_jobs=1, verbose=False)
    raw_clean = gedai.transform_raw(raw, n_jobs=1, verbose=False)
    
    raw_data = raw.get_data()
    clean_data = raw_clean.get_data()
    
    # Fp1 is usually best for seeing eye blinks
    fp1_idx = ch_kept.index("Fp1")
    
    def get_band_power(data, fmin, fmax):
        f, psd = welch(data, fs=sfreq, nperseg=min(256, data.shape[-1]), axis=-1)
        mask = (f >= fmin) & (f <= fmax)
        return np.trapezoid(psd[..., mask], f[mask], axis=-1)

    # 1. BRAIN SIGNAL RETENTION (8-30 Hz)
    raw_brain = get_band_power(raw_data, 8, 30)
    clean_brain = get_band_power(clean_data, 8, 30)
    retention = np.mean(clean_brain / raw_brain)
    
    # 2. NOISE REMOVAL (1-7 Hz, Focused on Fp1)
    raw_noise = get_band_power(raw_data[fp1_idx], 1, 7)
    clean_noise = get_band_power(clean_data[fp1_idx], 1, 7)
    removal_pct = (1.0 - (clean_noise / raw_noise)) * 100
    
    print(f"Subject {subject_id}:")
    print(f"  Brain Signal Retention (8-30 Hz): {retention*100:.2f}%")
    print(f"  Eye Blink Noise Removal (1-7 Hz on Fp1): {removal_pct:.2f}%")
    print("-" * 30)

measure_integrity(1)
measure_integrity(2)
