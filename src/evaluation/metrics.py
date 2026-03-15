from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import permutation_test


@dataclass
class SubjectPerformance:
    subject_id: int
    backbone: str
    pipeline: str
    fold_accuracies: List[float]
    mean_accuracy: float
    std_accuracy: float
    p_empirical: float
    alpha_ratio: float = 1.0
    beta_ratio: float = 1.0


def empirical_chance_p_value(
    acc_real: float,
    null_accuracies: Sequence[float],
) -> float:
    """Compute empirical p-value vs chance from a null distribution."""
    null_arr = np.asarray(null_accuracies)
    return float(np.mean(null_arr >= acc_real))


def cohen_d_pooled(
    scores1: Sequence[float],
    scores2: Sequence[float],
) -> float:
    """Cohen's d using pooled standard deviation."""
    x1 = np.asarray(scores1, dtype=float)
    x2 = np.asarray(scores2, dtype=float)
    n1, n2 = len(x1), len(x2)
    sd1 = x1.std(ddof=1)
    sd2 = x2.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((x1.mean() - x2.mean()) / pooled_sd)


def paired_permutation_p_value(
    scores1: Sequence[float],
    scores2: Sequence[float],
    n_resamples: int,
) -> float:
    """Paired permutation test on fold-level scores using scipy.stats.permutation_test."""
    x1 = np.asarray(scores1, dtype=float)
    x2 = np.asarray(scores2, dtype=float)

    def statistic(a, b):
        return np.mean(a - b)

    res = permutation_test(
        (x1, x2),
        statistic,
        permutation_type="pairings",
        n_resamples=n_resamples,
        vectorized=False,
        alternative="two-sided",
    )
    return float(res.pvalue)

def delong_auc_p_value(
    y_true: np.ndarray,
    y_score1: np.ndarray,
    y_score2: np.ndarray,
) -> float:
    """Compute the two-sided p-value for the difference in AUCs between two models using the DeLong test."""
    try:
        from MLstatkit import Delong_test
    except ImportError:
        import warnings
        warnings.warn("MLstatkit not installed. Cannot compute DeLong test. Returning 1.0.")
        return 1.0
        
    y_true_bin = (y_true > 0).astype(int)
    z, p_value = Delong_test(
        true=y_true_bin,
        prob_A=y_score1,
        prob_B=y_score2,
        return_ci=False,
        return_auc=False,
        verbose=0
    )
    return float(p_value)


def compute_band_power(
    X: np.ndarray, sfreq: float, lo: float, hi: float, ch_idx: int = 0
) -> float:
    """Compute trial-averaged power in [lo, hi] Hz using Welch on a specific channel."""
    from scipy.signal import welch

    # Trial-average the signal first to get consistent spectral power
    sig = X[:, ch_idx, :].mean(axis=0)
    nperseg = min(256, len(sig))
    f, p = welch(sig, fs=sfreq, nperseg=nperseg)
    mask = (f >= lo) & (f <= hi)
    if not np.any(mask):
        return 0.0
    # Use np.trapezoid (NumPy 2.0+) if available, else fallback
    try:
        return float(np.trapezoid(p[mask], f[mask]))
    except AttributeError:
        return float(np.trapz(p[mask], f[mask]))

