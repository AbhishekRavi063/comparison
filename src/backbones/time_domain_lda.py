from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class TimeLdaResult:
    fold_accuracies: List[float]
    pooled_test_correct: int = 0
    pooled_test_total: int = 0


def _downsample_trials(
    X: np.ndarray,
    sfreq: float,
    target_sfreq: Optional[float] = None,
) -> np.ndarray:
    """Lightweight decimation for ERP-style decoding."""
    X = np.asarray(X, dtype=np.float32)
    if target_sfreq is None or target_sfreq <= 0 or sfreq <= target_sfreq:
        return X
    step = max(1, int(round(float(sfreq) / float(target_sfreq))))
    return X[..., ::step].astype(np.float32, copy=False)


def build_time_lda_features_for_splits(
    X_proc: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    sfreq: float,
    target_sfreq: Optional[float] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Precompute flattened time-domain features once per split."""
    X_ds = _downsample_trials(X_proc, sfreq=sfreq, target_sfreq=target_sfreq)
    n_trials = X_ds.shape[0]
    X_feat = X_ds.reshape(n_trials, -1).astype(np.float32, copy=False)
    fold_features: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in cv_splits:
        fold_features.append((X_feat[train_idx], X_feat[test_idx]))
    return fold_features


def run_time_lda_cv_precomputed_features(
    y: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    fold_features: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> TimeLdaResult:
    """Run shrinkage LDA on precomputed ERP-style flattened features."""
    y = np.asarray(y)
    fold_accuracies: List[float] = []
    pooled_correct = 0
    pooled_total = 0
    for (train_idx, test_idx), (X_train_feat, X_test_feat) in zip(cv_splits, fold_features):
        y_train = y[train_idx]
        y_test = y[test_idx]
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_train_feat, y_train)
        pred = clf.predict(X_test_feat)
        pooled_correct += int(np.sum(pred == y_test))
        pooled_total += int(len(y_test))
        fold_accuracies.append(float(np.mean(pred == y_test)))
    return TimeLdaResult(
        fold_accuracies=fold_accuracies,
        pooled_test_correct=pooled_correct,
        pooled_test_total=pooled_total,
    )


def fit_time_lda_model_preprocessed(
    X_proc: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    denoising: str,
    target_sfreq: Optional[float] = None,
) -> dict:
    """Train time-domain shrinkage LDA on already-preprocessed data."""
    X_ds = _downsample_trials(X_proc, sfreq=sfreq, target_sfreq=target_sfreq)
    X_feat = X_ds.reshape(X_ds.shape[0], -1).astype(np.float32, copy=False)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(X_feat, y)
    return {
        "backbone": "time_lda",
        "clf": clf,
        "denoising": denoising,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "feature_sfreq": (float(target_sfreq) if target_sfreq and target_sfreq > 0 else float(sfreq)),
        "l_freq": l_freq,
        "h_freq": h_freq,
    }
