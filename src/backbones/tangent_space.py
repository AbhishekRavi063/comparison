from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import logm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..denoising.pipelines import (
    apply_asr,
    apply_gedai,
    apply_icalabel,
    apply_pylossless,
    bandpass_filter,
)


@dataclass
class TangentSpaceResult:
    fold_accuracies: List[float]
    pooled_test_correct: int = 0
    pooled_test_total: int = 0
    fold_aucs: List[float] = field(default_factory=list)


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
    # Stronger loading for short-epoch / high-channel consumer EEG (avoids singular logm).
    tr = float(np.trace(C))
    n = C.shape[0]
    scale = tr / n if tr > 1e-12 else 1.0
    eps = max(1e-10 * scale, 1e-5 * scale)
    C = (C + eps * np.eye(n, dtype=np.float32)).astype(np.float32)
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
        # Symmetrize + jitter so logm is defined (trial covs can be near-singular after whitening).
        Ti = T[i].astype(np.float64)
        Ti = 0.5 * (Ti + Ti.T)
        n_ch = Ti.shape[0]
        tr = float(np.trace(Ti))
        scale = tr / n_ch if tr > 1e-12 else 1.0
        Ti = Ti + (1e-8 * scale) * np.eye(n_ch, dtype=np.float64)
        l = logm(Ti).real
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
    denoising : {\"baseline\", \"icalabel\", \"gedai\", \"asr\", \"pylossless\"}
        Denoising strategy to apply before covariance computation.
    """
    if denoising not in {"baseline", "icalabel", "gedai", "asr", "pylossless"}:
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
        else:
            X_all = np.concatenate([X_train_bp, X_test_bp], axis=0)
            if denoising == "gedai":
                X_all_clean = apply_gedai(
                    X_all, sfreq, ch_names, l_freq, h_freq
                )
            elif denoising == "asr":
                X_all_clean = apply_asr(X_all, sfreq, ch_names, l_freq, h_freq)
            else:
                X_all_clean = apply_pylossless(
                    X_all, sfreq, ch_names, l_freq, h_freq
                )
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

    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        cov_est = Covariances(estimator="oas")
        for train_idx, test_idx in cv_splits:
            X_train = X_proc[train_idx]
            X_test = X_proc[test_idx]
            cov_train = cov_est.fit_transform(X_train)
            cov_test = cov_est.transform(X_test)
            ts = TangentSpace()
            X_train_feat = ts.fit_transform(cov_train)
            X_test_feat = ts.transform(cov_test)
            fold_features.append(
                (
                    X_train_feat.astype(np.float32, copy=False),
                    X_test_feat.astype(np.float32, copy=False),
                )
            )
        return fold_features
    except Exception:
        pass

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
    fold_aucs: List[float] = []
    pooled_correct = 0
    pooled_total = 0

    for (train_idx, test_idx), (X_train_feat, X_test_feat) in zip(cv_splits, fold_features):
        y_train = y[train_idx]
        y_test = y[test_idx]
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(X_train_feat, y_train)
        pred = clf.predict(X_test_feat)
        pooled_correct += int(np.sum(pred == y_test))
        pooled_total += int(len(y_test))
        fold_accuracies.append(float(np.mean(pred == y_test)))
        try:
            proba = clf.predict_proba(X_test_feat)[:, 1]
            fold_aucs.append(float(roc_auc_score(y_test, proba)))
        except Exception:
            fold_aucs.append(0.5)

    return TangentSpaceResult(
        fold_accuracies=fold_accuracies,
        pooled_test_correct=pooled_correct,
        pooled_test_total=pooled_total,
        fold_aucs=fold_aucs,
    )


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
    elif denoising == "gedai":
        X_proc = apply_gedai(X_bp, sfreq, ch_names, l_freq, h_freq)
    elif denoising == "asr":
        X_proc = apply_asr(X_bp, sfreq, ch_names, l_freq, h_freq)
    else:
        X_proc = apply_pylossless(X_bp, sfreq, ch_names, l_freq, h_freq)
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
    C_ref = None
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        covs = Covariances(estimator="oas").fit_transform(X_proc)
        ts = TangentSpace()
        X_feat = ts.fit_transform(covs).astype(np.float32, copy=False)
        C_ref = getattr(ts, "reference_", None)
    except Exception:
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
