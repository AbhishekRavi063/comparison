#!/usr/bin/env python3
"""
Compare GEDAI engines on Cho2017: Our Bypass Fix (gedai_official) vs Professor Threshold Fix (gedai_professor_fix).

Requires 10 Cho2017 subjects in data/cho2017/processed (run scripts/download_cho2017_10subjects.py first).

Usage:
    cd /path/to/comparison
    python scripts/compare_gedai_fixes_cho2017.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List

# Force CPU for both engines
os.environ["PYGEDAI_FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mne
from src.io.dataset import NpzMotorImageryDataset
from src.denoising.pipelines import apply_gedai, bandpass_filter
from src.backbones.tangent_space import (
    _covariance_matrices,
    _riemannian_mean,
    _tangent_space_projection,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = PROJECT_ROOT / "data" / "cho2017" / "processed"
SUBJECTS = list(range(1, 11))
L_FREQ = 8.0
H_FREQ = 30.0
GEDAI_OFFICIAL = PROJECT_ROOT / "gedai_official"
GEDAI_PROFESSOR = PROJECT_ROOT / "gedai_professor_fix"


def evaluate_accuracy(X: np.ndarray, y: np.ndarray):
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


def run_benchmark(engine_path: Path, label: str, subjects: List[int]) -> dict:
    print(f"\n--- Testing GEDAI Engine: {label} ---")
    os.environ["GEDAI_LIBRARY_PATH"] = str(engine_path.resolve())
    engine_results = {}
    for subj in subjects:
        dataset = NpzMotorImageryDataset(DATA_ROOT, subjects=[subj])
        subject_data = dataset._load_subject_file(subj)
        X, y = subject_data.X, subject_data.y
        sfreq = subject_data.sfreq
        ch_names = list(subject_data.ch_names)
        print(f"  [{label}] Processing Subject {subj}...")
        X_clean = apply_gedai(X, sfreq, ch_names, L_FREQ, H_FREQ)
        mean_acc, std_acc = evaluate_accuracy(X_clean, y)
        print(f"    S{subj} Mean Accuracy: {mean_acc:.4f}")
        engine_results[subj] = mean_acc
    return engine_results


def main() -> None:
    if not DATA_ROOT.exists():
        print(f"Error: {DATA_ROOT} not found. Run: python scripts/download_cho2017_10subjects.py")
        sys.exit(1)
    if not (GEDAI_OFFICIAL / "gedai").exists():
        print(f"Error: {GEDAI_OFFICIAL} not found (gedai_official clone).")
        sys.exit(1)
    if not (GEDAI_PROFESSOR / "gedai").exists():
        print(f"Error: {GEDAI_PROFESSOR} not found (gedai_professor_fix clone).")
        sys.exit(1)

    final_table = {}

    # 1. Baseline
    print("\n--- Baseline (Bandpass Only) ---")
    final_table["Baseline"] = {}
    for subj in SUBJECTS:
        dataset = NpzMotorImageryDataset(DATA_ROOT, subjects=[subj])
        subject_data = dataset._load_subject_file(subj)
        X_bp = bandpass_filter(subject_data.X, subject_data.sfreq, L_FREQ, H_FREQ)
        acc, _ = evaluate_accuracy(X_bp, subject_data.y)
        print(f"  S{subj} Baseline: {acc:.4f}")
        final_table["Baseline"][subj] = acc

    # 2. Our Fix (Bypass)
    final_table["Our Bypass Fix"] = run_benchmark(GEDAI_OFFICIAL, "Our Bypass", SUBJECTS)

    # 3. Professor's Fix (Threshold)
    final_table["Professor Threshold Fix"] = run_benchmark(
        GEDAI_PROFESSOR, "Professor Threshold", SUBJECTS
    )

    # Summary table: S1..S10 + mean
    n_subj = len(SUBJECTS)
    col_w = 8
    header_cols = " | ".join([f"S{s:2d}".rjust(col_w) for s in SUBJECTS])
    sep_len = 25 + (col_w + 3) * n_subj
    print("\n" + "=" * sep_len)
    print(f"{'Engine':25} | {header_cols} | {'Mean':>8}")
    print("-" * sep_len)
    for engine, res in final_table.items():
        vals = [res[s] for s in SUBJECTS]
        mean_val = np.mean(vals)
        row_vals = " | ".join([f"{v:.4f}".rjust(col_w) for v in vals])
        print(f"{engine:25} | {row_vals} | {mean_val:.4f}")
    print("=" * sep_len)


if __name__ == "__main__":
    main()
