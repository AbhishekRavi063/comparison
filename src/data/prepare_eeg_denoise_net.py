from __future__ import annotations

"""
Prepare EEGdenoiseNet (eeg_denoise_dataset).

Downloads contaminated EOG segments from Hugging Face,
mixes them as pseudo-subjects, and saves each as:

    <out_root>/subject_<ID>.npz

with keys: X (float32), y, sfreq (256.0), ch_names.
"""

import os
import tarfile
import requests
import numpy as np
import scipy.io
from pathlib import Path

# MathWorks Source with .mat files
DATA_URL = "https://ssd.mathworks.com/supportfiles/SPT/data/EEGEOGDenoisingData.zip"

def _prepare_subject(subject: int, out_root: Path) -> dict:
    """Download, extract, and convert EEGdenoiseNet segments into a 'subject'."""
    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir = out_root.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "EEGEOGDenoisingData.zip"
    
    # 1. Download if not exists
    if not zip_path.exists():
        print(f"  Downloading EEGdenoiseNet EOG data from MathWorks...")
        response = requests.get(DATA_URL, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # 2. Extract
    import zipfile
    extract_target = raw_dir / "EEGEOGDenoisingData"
    if not extract_target.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_target)
            
    # 3. Load segments
    mat_path = extract_target / "EOG_all_epochs.mat"
    eeg_mat_path = extract_target / "EEG_all_epochs.mat"
    
    if not mat_path.exists() or not eeg_mat_path.exists():
        # Check subfolders
        mat_path = list(extract_target.glob("**/EOG_all_epochs.mat"))[0]
        eeg_mat_path = list(extract_target.glob("**/EEG_all_epochs.mat"))[0]
    
    eog_data = scipy.io.loadmat(mat_path)['EOG_all_epochs']
    eeg_data = scipy.io.loadmat(eeg_mat_path)['EEG_all_epochs']
    
    # EEGdenoiseNet has segments of shape (n_epochs, n_samples) = (3400, 512) typically
    # We simulate a "subject" by taking a subset of these and creating a multi-channel-like array
    # Since it's single-channel denoising, we'll just stack them as 'trials'
    
    # Subject 1 gets segments 0-400, Subject 2 gets 400-800, etc.
    start = (subject - 1) * 400
    end = subject * 400
    
    # Mix them to create contaminated data (SNR=0 for max noise)
    # Contaminated = Clean EEG + EOG
    X_clean = eeg_data[start:end, :]
    X_noise = eog_data[start:end, :]
    
    # Normalise and mix
    X_contaminated = X_clean + X_noise
    
    # Reshape to (n_trials, n_channels, n_times) -> (400, 1, 512)
    X = X_contaminated[:, np.newaxis, :].astype(np.float32)
    y = np.zeros(X.shape[0], dtype=int) # Fake labels
    
    sfreq = 256.0
    ch_names = ["EEG01"]
    
    out_path = out_root / f"subject_{subject}.npz"
    np.savez(out_path, X=X, y=y, sfreq=sfreq, ch_names=np.array(ch_names, dtype=object))
    
    return {
        "subject": subject,
        "n_trials": X.shape[0],
        "n_channels": X.shape[1],
        "n_times": X.shape[2],
        "sfreq": sfreq,
    }
