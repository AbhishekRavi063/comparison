import os, sys
import numpy as np
import mne
from pathlib import Path
from scipy.signal import welch

# Force CPU
os.environ["PYGEDAI_FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.io.dataset import NpzMotorImageryDataset
from src.denoising.pipelines import apply_gedai

DATA_ROOT = "data/weibo2014/processed"
SUBJECT = 1

def measure_retention(engine_path, label):
    print(f"\n--- Checking Signal Integrity: {label} ---")
    os.environ["GEDAI_LIBRARY_PATH"] = engine_path
    
    dataset = NpzMotorImageryDataset(Path(DATA_ROOT), subjects=[SUBJECT])
    subject_data = dataset._load_subject_file(SUBJECT)
    X, y = subject_data.X, subject_data.y
    sfreq = subject_data.sfreq
    ch_names = list(subject_data.ch_names)
    
    # Run GEDAI
    X_clean = apply_gedai(X, sfreq, ch_names, 1.0, 100.0) # wide bandpass to see all, well below Nyquist
    
    def get_band_power(data, fmin, fmax):
        f, psd = welch(data, fs=sfreq, nperseg=min(256, data.shape[-1]), axis=-1)
        mask = (f >= fmin) & (f <= fmax)
        return np.trapezoid(psd[..., mask], f[mask], axis=-1)
        
    # Alpha (8-12)
    p_orig_alpha = get_band_power(X, 8, 12)
    p_clean_alpha = get_band_power(X_clean, 8, 12)
    ret_alpha = np.mean(p_clean_alpha / p_orig_alpha) * 100
    
    # Beta (13-30)
    p_orig_beta = get_band_power(X, 13, 30)
    p_clean_beta = get_band_power(X_clean, 13, 30)
    ret_beta = np.mean(p_clean_beta / p_orig_beta) * 100
    
    # Noise (1-7)
    p_orig_noise = get_band_power(X, 1, 7)
    p_clean_noise = get_band_power(X_clean, 1, 7)
    rem_noise = (1.0 - np.mean(p_clean_noise / p_orig_noise)) * 100
    
    print(f"  [{label}] Alpha Retention (8-12 Hz): {ret_alpha:.2f}%")
    print(f"  [{label}] Beta Retention (13-30 Hz): {ret_beta:.2f}%")
    print(f"  [{label}] Noise Removal (1-7 Hz): {rem_noise:.2f}%")
    
    return X_clean

if __name__ == "__main__":
    X_bypass = measure_retention("/Users/abhishekr/Documents/EEG/comparison/gedai_official", "Our Bypass")
    X_prof = measure_retention("/Users/abhishekr/Documents/EEG/comparison/gedai_professor_fix", "Professor Threshold")
    
    # Are they literally the same array?
    diff = np.max(np.abs(X_bypass - X_prof))
    print(f"\nMax difference between Bypass and Professor output: {diff:.6e}")
    
    if diff < 1e-6:
        print("Conclusion: The outputs are IDENTICAL.")
    else:
        print("Conclusion: The outputs are DIFFERENT.")
