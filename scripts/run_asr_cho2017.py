#!/usr/bin/env python3
"""
Run ASR on Cho2017 (10 subjects) and report accuracy vs baseline.

Same evaluation as compare_gedai_fixes_cho2017.py (tangent-space + 5-fold CV)
so you can compare ASR with your existing GEDAI results.

Requires: data/cho2017/processed (run scripts/download_cho2017_10subjects.py first).
Requires: asrpy installed in the environment (pip install asrpy).

Usage:
    cd /path/to/comparison
    python scripts/run_asr_cho2017.py
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.dataset import NpzMotorImageryDataset
from src.denoising.pipelines import bandpass_filter, apply_asr
from src.backbones.tangent_space import (
    _covariance_matrices,
    _riemannian_mean,
    _tangent_space_projection,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = PROJECT_ROOT / "data" / "cho2017" / "processed"
DEFAULT_SUBJECTS = list(range(1, 11))
L_FREQ = 8.0
H_FREQ = 30.0


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR on Cho2017 and compare with baseline.")
    parser.add_argument("--smoke", action="store_true", help="Run only subject 1 (smoke test).")
    args = parser.parse_args()
    subjects = [1] if args.smoke else DEFAULT_SUBJECTS
    if args.smoke:
        print("Smoke test: subject 1 only.")

    if not DATA_ROOT.exists():
        print(f"Error: {DATA_ROOT} not found. Run: python scripts/download_cho2017_10subjects.py")
        sys.exit(1)

    final_table = {}

    # 1. Baseline
    print("\n--- Baseline (Bandpass Only) ---")
    final_table["Baseline"] = {}
    for subj in subjects:
        dataset = NpzMotorImageryDataset(DATA_ROOT, subjects=[subj])
        subject_data = dataset._load_subject_file(subj)
        X_bp = bandpass_filter(subject_data.X, subject_data.sfreq, L_FREQ, H_FREQ)
        acc, _ = evaluate_accuracy(X_bp, subject_data.y)
        print(f"  S{subj} Baseline: {acc:.4f}")
        final_table["Baseline"][subj] = acc

    # 2. ASR
    print("\n--- ASR (Artifact Subspace Reconstruction) ---")
    final_table["ASR"] = {}
    for subj in subjects:
        dataset = NpzMotorImageryDataset(DATA_ROOT, subjects=[subj])
        subject_data = dataset._load_subject_file(subj)
        X, y = subject_data.X, subject_data.y
        sfreq = subject_data.sfreq
        ch_names = list(subject_data.ch_names)
        print(f"  [ASR] Processing Subject {subj}...")
        X_clean = apply_asr(X, sfreq, ch_names, L_FREQ, H_FREQ)
        mean_acc, _ = evaluate_accuracy(X_clean, y)
        print(f"    S{subj} Mean Accuracy: {mean_acc:.4f}")
        final_table["ASR"][subj] = mean_acc

    # Summary table
    n_subj = len(subjects)
    col_w = 8
    header_cols = " | ".join([f"S{s:2d}".rjust(col_w) for s in subjects])
    sep_len = 25 + (col_w + 3) * n_subj
    print("\n" + "=" * sep_len)
    print(f"{'Engine':25} | {header_cols} | {'Mean':>8}")
    print("-" * sep_len)
    for engine, res in final_table.items():
        vals = [res[s] for s in subjects]
        mean_val = np.mean(vals)
        row_vals = " | ".join([f"{v:.4f}".rjust(col_w) for v in vals])
        print(f"{engine:25} | {row_vals} | {mean_val:.4f}")
    print("=" * sep_len)
    print("\nCompare this ASR row with your GEDAI results from compare_gedai_fixes_cho2017.py")


if __name__ == "__main__":
    main()
