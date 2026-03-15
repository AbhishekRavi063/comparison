"""
export_final_professor_package.py
---------------------------------
Exports Weibo2014 Subject 1 & 2 data in EDF format using Professor's Fix.
"""
import os, sys
import numpy as np
import mne
from pathlib import Path

# Force CPU
os.environ["PYGEDAI_FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.io.dataset import NpzMotorImageryDataset
from src.denoising.pipelines import bandpass_filter, apply_gedai

DATA_ROOT = "data/weibo2014/processed"
OUT_DIR = Path("exports/final_professor_review")
SUBJECTS = [1, 2]
L_FREQ = 8.0
H_FREQ = 30.0

os.environ["GEDAI_LIBRARY_PATH"] = "/Users/abhishekr/Documents/EEG/comparison/gedai_professor_fix"

OUT_DIR.mkdir(parents=True, exist_ok=True)

for sid in SUBJECTS:
    print(f"Processing Subject {sid}...")
    dataset = NpzMotorImageryDataset(Path(DATA_ROOT), subjects=[sid])
    subject_data = dataset._load_subject_file(sid)
    X, y = subject_data.X, subject_data.y
    sfreq = subject_data.sfreq
    ch_names = list(subject_data.ch_names)
    
    # 1. Bandpass Only (Before GEDAI)
    X_bp = bandpass_filter(X, sfreq, L_FREQ, H_FREQ).astype(np.float32)
    
    # 2. After GEDAI (Professor Fix)
    X_clean = apply_gedai(X, sfreq, ch_names, L_FREQ, H_FREQ).astype(np.float32)
    
    # 3. Concatenate and Export
    n_trials, n_ch, n_times = X.shape
    raw_before = X_bp.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times) * 1e-6
    raw_after = X_clean.transpose(1, 0, 2).reshape(n_ch, n_trials * n_times) * 1e-6
    raw_noise = (raw_before - raw_after)
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    
    mne.export.export_raw(str(OUT_DIR / f"sub-{sid}_before_gedai.edf"), mne.io.RawArray(raw_before, info), fmt="edf", overwrite=True)
    mne.export.export_raw(str(OUT_DIR / f"sub-{sid}_after_gedai_professor_fix.edf"), mne.io.RawArray(raw_after, info), fmt="edf", overwrite=True)
    mne.export.export_raw(str(OUT_DIR / f"sub-{sid}_removed_noise.edf"), mne.io.RawArray(raw_noise, info), fmt="edf", overwrite=True)

print(f"Done! Files in {OUT_DIR}")
