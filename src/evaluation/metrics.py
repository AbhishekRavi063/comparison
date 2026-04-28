from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, permutation_test, wilcoxon


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
    # Filled when subject_chance_method == binomial (exact test on pooled CV test predictions).
    pooled_test_correct: int | None = None
    pooled_test_total: int | None = None
    p_vs_chance_method: str = "permutation"
    mean_auc: float = 0.0
    fold_aucs: List[float] = field(default_factory=list)
    # Alpha lateralization index — proof of signal preservation.
    # LI > 0: ipsilateral alpha > contralateral alpha (expected auditory attention pattern).
    # LI closer to 0 after GEDAI vs baseline would indicate signal destruction.
    # LI higher after GEDAI indicates GEDAI sharpened the neural lateralization.
    lateralization_index: float = 0.0


def empirical_chance_p_value(
    acc_real: float,
    null_accuracies: Sequence[float],
) -> float:
    """Compute empirical p-value vs chance from a null distribution."""
    null_arr = np.asarray(null_accuracies)
    return float(np.mean(null_arr >= acc_real))


def binomial_vs_chance_p_value(
    n_correct: int,
    n_total: int,
    n_classes: int,
) -> float:
    """Exact binomial test: accuracy vs uniform chance (1/K). One-sided greater.

    Appropriate when the number of pooled test predictions is large (professor guidance:
    label-shuffle permutations can be skipped).
    """
    if n_total < 1 or n_classes < 2:
        return 1.0
    p0 = 1.0 / float(n_classes)
    r = binomtest(int(n_correct), int(n_total), p=p0, alternative="greater")
    return float(r.pvalue)


def mann_whitney_pipeline_p_value(
    scores1: Sequence[float],
    scores2: Sequence[float],
) -> float:
    """Two-sample Mann–Whitney U (independent groups). Same length does not imply pairing."""
    x1 = np.asarray(scores1, dtype=float)
    x2 = np.asarray(scores2, dtype=float)
    if len(x1) < 2 or len(x2) < 2:
        return 1.0
    r = mannwhitneyu(x1, x2, alternative="two-sided")
    return float(r.pvalue)


def wilcoxon_paired_pipeline_p_value(
    scores1: Sequence[float],
    scores2: Sequence[float],
) -> float:
    """Paired Wilcoxon signed-rank on subject-level means (same subjects, two pipelines)."""
    x1 = np.asarray(scores1, dtype=float)
    x2 = np.asarray(scores2, dtype=float)
    n = min(len(x1), len(x2))
    if n < 2:
        return 1.0
    x1, x2 = x1[:n], x2[:n]
    diff = x1 - x2
    if np.allclose(diff, 0):
        return 1.0
    try:
        r = wilcoxon(x1, x2, alternative="two-sided", zero_method="wilcox")
    except TypeError:
        r = wilcoxon(x1, x2, alternative="two-sided")
    return float(r.pvalue)


def cohen_d_pooled(
    scores1: Sequence[float],
    scores2: Sequence[float],
) -> float:
    """Cohen's d using pooled standard deviation."""
    x1 = np.asarray(scores1, dtype=float)
    x2 = np.asarray(scores2, dtype=float)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    sd1 = x1.std(ddof=1)
    sd2 = x2.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
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

    if len(x1) < 2 or len(x2) < 2:
        return 1.0

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

