from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from ..denoising.pipelines import bandpass_filter, apply_icalabel, apply_gedai


@dataclass
class CSPResult:
    fold_accuracies: List[float]


def _compute_csp_filters(
    X_train: np.ndarray, y_train: np.ndarray, n_components: int = 6
) -> np.ndarray:
    """Compute simple CSP spatial filters for binary classification."""
    classes = np.unique(y_train)
    if len(classes) != 2:
        # Fallback for degenerate folds (e.g. during permutation testing)
        n_ch = X_train.shape[1]
        return np.eye(n_ch, dtype=np.float32)[:, :n_components].T

    X1 = X_train[y_train == classes[0]]
    X2 = X_train[y_train == classes[1]]

    def cov(trials: np.ndarray) -> np.ndarray:
        # trials: (n_trials, n_channels, n_times)
        covs = []
        for tr in trials:
            covs.append(np.cov(tr))
        return np.mean(covs, axis=0)

    C1 = cov(X1)
    C2 = cov(X2)
    C = C1 + C2
    n_ch = C.shape[0]
    # Regularize so C is strictly positive definite (avoids LinAlgError with shuffled labels or few trials)
    reg = max(1e-6 * np.trace(C) / n_ch, 1e-10)
    C = C + reg * np.eye(n_ch)

    # Generalized eigenvalue problem C1 v = λ (C1 + C2) v
    eigvals, eigvecs = eigh(C1, C)
    # Sort by eigenvalue magnitude
    ix = np.argsort(eigvals)[::-1]
    W = eigvecs[:, ix]
    # Take leading and trailing components
    half = n_components // 2
    filters = np.hstack([W[:, :half], W[:, -half:]])
    return filters.T  # (n_components, n_channels)


def _project_csp(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # X: (n_trials, n_channels, n_times)
    # W: (n_components, n_channels)
    X_proj = np.einsum("kc,tcw->tkw", W, X)
    # Log-variance features; keep float32 for memory
    var = np.var(X_proj, axis=-1, dtype=np.float32)
    var /= var.sum(axis=1, keepdims=True)
    return np.log(np.clip(var, 1e-10, None)).astype(np.float32, copy=False)


def run_csp_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    cv_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
    denoising: str,
) -> CSPResult:
    """Run CSP → LDA backbone (float32 throughout for memory)."""
    if denoising not in {"baseline", "icalabel", "gedai"}:
        raise ValueError(f"Unknown denoising strategy: {denoising}")

    X = np.asarray(X, dtype=np.float32)
    skf = StratifiedKFold(
        n_splits=cv_splits, shuffle=cv_shuffle, random_state=cv_random_state
    )

    fold_accuracies: List[float] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Bandpass always applied
        X_train_bp = bandpass_filter(X_train, sfreq, l_freq, h_freq)
        X_test_bp = bandpass_filter(X_test, sfreq, l_freq, h_freq)

        if denoising == "baseline":
            X_train_proc, X_test_proc = X_train_bp, X_test_bp
        elif denoising == "icalabel":
            X_all = np.concatenate([X_train_bp, X_test_bp], axis=0)
            X_all_clean = apply_icalabel(X_all, sfreq, ch_names, l_freq, h_freq)
            X_train_proc = X_all_clean[: len(X_train_bp)]
            X_test_proc = X_all_clean[len(X_train_bp) :]
        else:  # gedai
            X_all = np.concatenate([X_train_bp, X_test_bp], axis=0)
            X_all_clean = apply_gedai(X_all, sfreq, ch_names)
            X_train_proc = X_all_clean[: len(X_train_bp)]
            X_test_proc = X_all_clean[len(X_train_bp) :]

        W = _compute_csp_filters(X_train_proc, y_train)
        X_train_feat = _project_csp(X_train_proc, W)
        X_test_feat = _project_csp(X_test_proc, W)

        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_feat, y_train)
        acc = clf.score(X_test_feat, y_test)
        fold_accuracies.append(float(acc))

    return CSPResult(fold_accuracies=fold_accuracies)


def run_csp_cv_preprocessed(
    X_proc: np.ndarray,
    y: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> CSPResult:
    """Run CSP->LDA CV on already-preprocessed data for fixed splits."""
    X_proc = np.asarray(X_proc, dtype=np.float32)
    y = np.asarray(y)
    fold_accuracies: List[float] = []

    for train_idx, test_idx in cv_splits:
        X_train = X_proc[train_idx]
        X_test = X_proc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        W = _compute_csp_filters(X_train, y_train)
        X_train_feat = _project_csp(X_train, W)
        X_test_feat = _project_csp(X_test, W)

        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_feat, y_train)
        acc = clf.score(X_test_feat, y_test)
        fold_accuracies.append(float(acc))

    return CSPResult(fold_accuracies=fold_accuracies)


def fit_csp_model(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    denoising: str,
    n_components: int = 6,
) -> dict:
    """Train CSP + LDA on full data; return a serializable model dict for saving.

    Returns dict with keys: backbone='csp', W (ndarray), clf (LDA), denoising, ch_names, sfreq, l_freq, h_freq.
    """
    X = np.asarray(X, dtype=np.float32)
    X_bp = bandpass_filter(X, sfreq, l_freq, h_freq)
    if denoising == "baseline":
        X_proc = X_bp
    elif denoising == "icalabel":
        X_proc = apply_icalabel(X_bp, sfreq, ch_names, l_freq, h_freq)
    else:
        X_proc = apply_gedai(X_bp, sfreq, ch_names)
    W = _compute_csp_filters(X_proc, y, n_components=n_components)
    X_feat = _project_csp(X_proc, W)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_feat, y)
    return {
        "backbone": "csp",
        "W": W.astype(np.float32),
        "clf": clf,
        "denoising": denoising,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "l_freq": l_freq,
        "h_freq": h_freq,
    }


def fit_csp_model_preprocessed(
    X_proc: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    denoising: str,
    n_components: int = 6,
) -> dict:
    """Train CSP + LDA on already-preprocessed data."""
    X_proc = np.asarray(X_proc, dtype=np.float32)
    W = _compute_csp_filters(X_proc, y, n_components=n_components)
    X_feat = _project_csp(X_proc, W)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_feat, y)
    return {
        "backbone": "csp",
        "W": W.astype(np.float32),
        "clf": clf,
        "denoising": denoising,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "l_freq": l_freq,
        "h_freq": h_freq,
    }
