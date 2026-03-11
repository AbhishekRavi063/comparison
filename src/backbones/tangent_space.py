from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import logm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ..denoising.pipelines import bandpass_filter, apply_icalabel, apply_gedai


@dataclass
class TangentSpaceResult:
    fold_accuracies: List[float]


def _covariance_matrices(X: np.ndarray) -> np.ndarray:
    """Compute covariance matrices for each trial (float32 for memory)."""
    n_trials, n_channels, _ = X.shape
    covs = np.empty((n_trials, n_channels, n_channels), dtype=np.float32)
    for i in range(n_trials):
        covs[i] = np.cov(X[i].astype(np.float32)).astype(np.float32)
    return covs


def _riemannian_mean(covs: np.ndarray) -> np.ndarray:
    """Regularized mean covariance (float32); diagonal loading for positive definiteness."""
    C = np.mean(covs, axis=0).astype(np.float32)
    C = 0.5 * (C + C.T)
    eps = 1e-6 * np.trace(C) / C.shape[0]
    C = (C + eps * np.eye(C.shape[0], dtype=np.float32)).astype(np.float32)
    return C


def _tangent_space_projection(covs: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    """Project covariance matrices to the tangent space at C_ref."""
    from scipy.linalg import fractional_matrix_power
    # Symmetric inverse square root
    inv_sqrt_C = fractional_matrix_power(C_ref, -0.5).real.astype(np.float32)
    # T = C_ref^{-1/2} @ covs @ C_ref^{-1/2}
    T = np.einsum("...ij,jk->...ik", covs, inv_sqrt_C)
    T = np.einsum("ij,...jk->...ik", inv_sqrt_C, T)
    
    logs = np.empty_like(T, dtype=np.float32)
    for i in range(len(T)):
        # Compute logm in float64 precision to avoid warnings/inaccuracies
        l = logm(T[i].astype(np.float64)).real
        logs[i] = l.astype(np.float32)
        
    # Flatten upper triangle (including diagonal)
    n_ch = logs.shape[1]
    idxs = np.triu_indices(n_ch)
    feat = logs[:, idxs[0], idxs[1]].copy()
    
    # Scale off-diagonal elements by sqrt(2) to preserve Riemannian metric
    off_diag_mask = idxs[0] != idxs[1]
    feat[:, off_diag_mask] *= np.sqrt(2.0, dtype=np.float32)
    
    return feat.astype(np.float32, copy=False)


def run_tangent_space_pipeline(
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
) -> TangentSpaceResult:
    """Run covariance → tangent space → logistic regression backbone.

    Parameters
    ----------
    denoising : {\"baseline\", \"icalabel\", \"gedai\"}
        Denoising strategy to apply before covariance computation.
    """
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

        cov_train = _covariance_matrices(X_train_proc)
        cov_test = _covariance_matrices(X_test_proc)
        C_ref = _riemannian_mean(cov_train)

        X_train_feat = _tangent_space_projection(cov_train, C_ref)
        X_test_feat = _tangent_space_projection(cov_test, C_ref)

        clf = LogisticRegression(
            solver="lbfgs", max_iter=1000, n_jobs=1
        )  # n_jobs=1 for memory constraint
        clf.fit(X_train_feat, y_train)
        acc = clf.score(X_test_feat, y_test)
        fold_accuracies.append(float(acc))

    return TangentSpaceResult(fold_accuracies=fold_accuracies)


def build_tangent_features_for_splits(
    X_proc: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Precompute tangent-space features once per split for repeated label tests."""
    X_proc = np.asarray(X_proc, dtype=np.float32)
    fold_features: List[Tuple[np.ndarray, np.ndarray]] = []

    for train_idx, test_idx in cv_splits:
        X_train = X_proc[train_idx]
        X_test = X_proc[test_idx]

        cov_train = _covariance_matrices(X_train)
        cov_test = _covariance_matrices(X_test)
        C_ref = _riemannian_mean(cov_train)
        X_train_feat = _tangent_space_projection(cov_train, C_ref)
        X_test_feat = _tangent_space_projection(cov_test, C_ref)
        fold_features.append((X_train_feat, X_test_feat))

    return fold_features


def run_tangent_cv_precomputed_features(
    y: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    fold_features: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> TangentSpaceResult:
    """Run logistic-regression CV using precomputed tangent features."""
    y = np.asarray(y)
    fold_accuracies: List[float] = []

    for (train_idx, test_idx), (X_train_feat, X_test_feat) in zip(cv_splits, fold_features):
        y_train = y[train_idx]
        y_test = y[test_idx]
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(X_train_feat, y_train)
        acc = clf.score(X_test_feat, y_test)
        fold_accuracies.append(float(acc))

    return TangentSpaceResult(fold_accuracies=fold_accuracies)


def run_tangent_cv_preprocessed(
    X_proc: np.ndarray,
    y: np.ndarray,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> TangentSpaceResult:
    """Run tangent-space CV on already-preprocessed data for fixed splits."""
    fold_features = build_tangent_features_for_splits(X_proc, cv_splits)
    return run_tangent_cv_precomputed_features(y, cv_splits, fold_features)


def fit_tangent_model(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    denoising: str,
) -> dict:
    """Train tangent-space + LogReg on full data; return serializable model dict."""
    X = np.asarray(X, dtype=np.float32)
    X_bp = bandpass_filter(X, sfreq, l_freq, h_freq)
    if denoising == "baseline":
        X_proc = X_bp
    elif denoising == "icalabel":
        X_proc = apply_icalabel(X_bp, sfreq, ch_names, l_freq, h_freq)
    else:
        X_proc = apply_gedai(X_bp, sfreq, ch_names)
    covs = _covariance_matrices(X_proc)
    C_ref = _riemannian_mean(covs)
    X_feat = _tangent_space_projection(covs, C_ref)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X_feat, y)
    return {
        "backbone": "tangent",
        "C_ref": C_ref,
        "clf": clf,
        "denoising": denoising,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "l_freq": l_freq,
        "h_freq": h_freq,
    }


def fit_tangent_model_preprocessed(
    X_proc: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    denoising: str,
) -> dict:
    """Train tangent-space + LogReg on already-preprocessed data."""
    X_proc = np.asarray(X_proc, dtype=np.float32)
    covs = _covariance_matrices(X_proc)
    C_ref = _riemannian_mean(covs)
    X_feat = _tangent_space_projection(covs, C_ref)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X_feat, y)
    return {
        "backbone": "tangent",
        "C_ref": C_ref,
        "clf": clf,
        "denoising": denoising,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "l_freq": l_freq,
        "h_freq": h_freq,
    }
