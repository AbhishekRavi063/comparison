import mne
import numpy as np
from scipy.signal import welch

def check_edf_retention(before_path, after_path):
    raw_before = mne.io.read_raw_edf(before_path, preload=True, verbose=False)
    raw_after = mne.io.read_raw_edf(after_path, preload=True, verbose=False)
    
    data_before = raw_before.get_data()
    data_after = raw_after.get_data()
    sfreq = raw_before.info["sfreq"]
    
    def get_band_power(data, fmin, fmax):
        f, psd = welch(data, fs=sfreq, nperseg=min(256, data.shape[-1]), axis=-1)
        mask = (f >= fmin) & (f <= fmax)
        return np.trapezoid(psd[..., mask], f[mask], axis=-1)
    
    # Brain band (8-30 Hz)
    power_before = get_band_power(data_before, 8, 30)
    power_after = get_band_power(data_after, 8, 30)
    
    retention = np.mean(power_after / power_before) * 100
    print(f"Verified Retention (8-30 Hz) in EDFs: {retention:.2f}%")

if __name__ == "__main__":
    check_edf_retention(
        "/Users/abhishekr/Documents/EEG/comparison/exports/gedai_review/subject_002_before_gedai.edf",
        "/Users/abhishekr/Documents/EEG/comparison/exports/gedai_review/subject_002_after_gedai.edf"
    )
