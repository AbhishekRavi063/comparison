import os
import sys
import numpy as np
from pathlib import Path
from typing import List

# Force CPU for both engines
os.environ["PYGEDAI_FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import mne
from src.io.dataset import NpzMotorImageryDataset
from src.denoising.pipelines import apply_gedai, bandpass_filter
from src.backbones.tangent_space import _covariance_matrices, _riemannian_mean, _tangent_space_projection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = "data/weibo2014/processed"
SUBJECTS = [1, 2]
L_FREQ = 8.0
H_FREQ = 30.0

def evaluate_accuracy(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        cov_train = _covariance_matrices(X_train)
        cov_test = _covariance_matrices(X_test)
        C_ref = _riemannian_mean(cov_train)
        
        X_train_feat = _tangent_space_projection(cov_train, C_ref)
        X_test_feat = _tangent_space_projection(cov_test, C_ref)
        
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(X_train_feat, y_train)
        fold_accuracies.append(clf.score(X_test_feat, y_test))
    
    return np.mean(fold_accuracies), np.std(fold_accuracies)

def run_benchmark(engine_path: str, label: str, subjects: List[int]):
    print(f"\n--- Testing GEDAI Engine: {label} ---")
    os.environ["GEDAI_LIBRARY_PATH"] = engine_path
    
    engine_results = {}
    for subj in subjects:
        # Load data
        dataset = NpzMotorImageryDataset(Path(DATA_ROOT), subjects=[subj])
        subject_data = dataset._load_subject_file(subj)
        X, y = subject_data.X, subject_data.y
        sfreq = subject_data.sfreq
        ch_names = list(subject_data.ch_names)
        
        # Apply GEDAI
        print(f"  [{label}] Processing Subject {subj}...")
        X_clean = apply_gedai(X, sfreq, ch_names, L_FREQ, H_FREQ)
        
        # Classification
        mean_acc, std_acc = evaluate_accuracy(X_clean, y)
        print(f"    S{subj} Mean Accuracy: {mean_acc:.4f}")
        engine_results[subj] = mean_acc
    return engine_results

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"Error: {DATA_ROOT} not found. Run prepare_weibo2014 first.")
        sys.exit(1)
        
    final_table = {}
    
    # 1. Baseline
    print("\n--- Baseline (Bandpass Only) ---")
    final_table["Baseline"] = {}
    for subj in SUBJECTS:
        dataset = NpzMotorImageryDataset(Path(DATA_ROOT), subjects=[subj])
        subject_data = dataset._load_subject_file(subj)
        X_bp = bandpass_filter(subject_data.X, subject_data.sfreq, L_FREQ, H_FREQ)
        acc, _ = evaluate_accuracy(X_bp, subject_data.y)
        print(f"  S{subj} Baseline: {acc:.4f}")
        final_table["Baseline"][subj] = acc

    # 2. Our Fix (Bypass)
    final_table["Our Bypass Fix"] = run_benchmark("/Users/abhishekr/Documents/EEG/comparison/gedai_official", "Our Bypass", SUBJECTS)

    # 3. Professor's Fix (Threshold)
    final_table["Professor Threshold Fix"] = run_benchmark("/Users/abhishekr/Documents/EEG/comparison/gedai_professor_fix", "Professor Threshold", SUBJECTS)

    print("\n" + "="*50)
    print(f"{'Engine':25} | {'S1':8} | {'S2':8}")
    print("-" * 50)
    for engine, res in final_table.items():
        print(f"{engine:25} | {res[1]:.4f}   | {res[2]:.4f}")
    print("="*50)
