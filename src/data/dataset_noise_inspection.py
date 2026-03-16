from __future__ import annotations

"""
Core noise diagnostics for EEG motor-imagery datasets.

Given an X array (n_trials, n_channels, n_times) and sfreq, computes:
  - Amplitude statistics (mean |amplitude|, std, peak)
  - Welch PSD per channel, averaged across trials
  - High-frequency power (>30 Hz) as noise proxy
  - Channel variance (mean across trials)
  - Spike count: samples exceeding ±5 sigma (z-score > 5)
  - Noise score (raw, before normalisation): hf_power + channel_variance

Also generates and saves:
  - raw_overlay_subj<N>.png   — first 10 channels, first trial, time domain
  - psd_subj<N>.png           — mean PSD across all channels + trial avg
  - noise_band_comparison.png — per-dataset HF power bar chart (written by caller)
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch


# ------------------------------------------------------------------
# Diagnostic computation
# ------------------------------------------------------------------

HF_CUTOFF_HZ = 30.0   # above this = noise band
SPIKE_SIGMA = 5.0      # z-score threshold for spike detection


def compute_noise_diagnostics(
    X: np.ndarray,
    sfreq: float,
    ch_names: Optional[List[str]] = None,
) -> Dict:
    """Compute all noise statistics for a single subject's epoch array.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    sfreq : float
    ch_names : optional list of channel names

    Returns
    -------
    dict with keys:
        amp_mean, amp_std, amp_peak,
        hf_power_db,         mean HF (>30 Hz) power in dB relative to mean total
        channel_variance,    mean variance per channel, averaged across channels (µV² or arb)
        spike_count,         number of samples exceeding ±5σ
        noise_score_raw      hf_power (abs) + channel_variance (abs), before normalisation
    """
    X = np.asarray(X, dtype=np.float64)
    n_trials, n_ch, n_times = X.shape

    # ── Amplitude statistics ──────────────────────────────────
    amp_mean = float(np.mean(np.abs(X)))
    amp_std = float(np.std(X))
    amp_peak = float(np.max(np.abs(X)))

    # ── Welch PSD: average over trials then channels ──────────
    nperseg = min(256, n_times)
    # Shape: (n_trials, n_ch, n_freqs)
    f, psd_all = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)
    psd_mean = psd_all.mean(axis=0)        # (n_ch, n_freqs) mean over trials
    psd_channel_mean = psd_mean.mean(axis=0)  # (n_freqs,) mean over channels

    # ── High-frequency power (>30 Hz) ───────────────────────────
    hf_mask = f > HF_CUTOFF_HZ
    lf_mask = f <= HF_CUTOFF_HZ
    hf_power = float(np.trapezoid(psd_channel_mean[hf_mask], f[hf_mask])) if hf_mask.any() else 0.0
    lf_power = float(np.trapezoid(psd_channel_mean[lf_mask], f[lf_mask])) if lf_mask.any() else 1.0
    # Normalise as ratio: HF/total (×100 → percentage)
    total_power = hf_power + lf_power
    hf_ratio = hf_power / max(total_power, 1e-30)    # 0–1

    # ── Channel variance ──────────────────────────────────────────
    # Var across time for each trial/channel, then mean
    ch_var = float(np.mean(np.var(X, axis=-1)))   # scalar

    # ── Spike detection ───────────────────────────────────────
    flat = X.reshape(-1)
    mu, sigma = flat.mean(), flat.std()
    spikes = int(np.sum(np.abs(flat - mu) > SPIKE_SIGMA * sigma))

    # ── Raw noise score (unnormalised) ───────────────────
    # Use HF ratio (0–1) + normalised channel variance (unit: std of signal)
    noise_score_raw = hf_ratio + ch_var  # both are positive; will normalise across datasets

    return dict(
        f=f,
        psd_mean=psd_mean,        # (n_ch, n_freqs)
        psd_channel_mean=psd_channel_mean,  # (n_freqs,)
        amp_mean=amp_mean,
        amp_std=amp_std,
        amp_peak=amp_peak,
        hf_power_abs=hf_power,
        hf_ratio=hf_ratio,
        channel_variance=ch_var,
        spike_count=spikes,
        noise_score_raw=noise_score_raw,
        n_trials=n_trials,
        n_ch=n_ch,
        n_times=n_times,
        sfreq=sfreq,
    )


# ------------------------------------------------------------------
# Plotting utilities
# ------------------------------------------------------------------

def plot_raw_overlay(
    X: np.ndarray,
    sfreq: float,
    ch_names: Optional[List[str]],
    out_path: Path,
    subject_id: int = 1,
    n_channels: int = 10,
    trial_idx: int = 0,
    dataset_label: str = "",
) -> None:
    """Plot raw signal overlay for first `n_channels` channels, one trial."""
    X = np.asarray(X, dtype=np.float64)
    n_ch_plot = min(n_channels, X.shape[1])
    x_trial = X[trial_idx, :n_ch_plot, :]   # (n_ch, n_times)
    t = np.arange(x_trial.shape[-1]) / sfreq

    fig, axes = plt.subplots(n_ch_plot, 1, figsize=(14, 1.5 * n_ch_plot), sharex=True)
    if n_ch_plot == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, n_ch_plot))

    for i, ax in enumerate(axes):
        ch = ch_names[i] if ch_names and i < len(ch_names) else f"Ch{i}"
        ax.plot(t, x_trial[i], lw=0.6, color=colors[i])
        ax.set_ylabel(ch, fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(
        f"{dataset_label} — Raw EEG overlay | Subject {subject_id} | Trial {trial_idx}",
        fontsize=10,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_psd(
    f: np.ndarray,
    psd_channel_mean: np.ndarray,
    out_path: Path,
    subject_id: int = 1,
    sfreq: float = 250.0,
    dataset_label: str = "",
) -> None:
    """Plot mean PSD with alpha/beta/HF bands highlighted."""
    fig, ax = plt.subplots(figsize=(10, 4))
    psd_db = 10 * np.log10(np.maximum(psd_channel_mean, 1e-30))

    ax.plot(f, psd_db, color="steelblue", lw=1.4, label="Mean PSD (all channels)")

    # Band shading
    ax.axvspan(8, 12, alpha=0.12, color="green", label="Alpha (8–12 Hz)")
    ax.axvspan(13, 30, alpha=0.12, color="orange", label="Beta (13–30 Hz)")
    ax.axvspan(30, f[-1], alpha=0.08, color="red", label=f"HF noise (>30 Hz)")
    ax.axvline(30, color="red", lw=0.8, ls="--")

    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Power (dB)", fontsize=9)
    ax.set_title(
        f"{dataset_label} — Mean PSD | Subject {subject_id} | sfreq={sfreq:.0f} Hz",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(sfreq / 2, 120))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_noise_band_comparison(
    dataset_labels: List[str],
    hf_ratios: List[float],
    ch_variances: List[float],
    noise_scores: List[float],
    out_path: Path,
) -> None:
    """Bar chart comparing HF power ratio, channel variance, and final noise score."""
    x = np.arange(len(dataset_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    # Normalise each metric for visual comparison
    _norm = lambda v: np.array(v) / (max(max(v), 1e-30))
    hf_n = _norm(hf_ratios)
    cv_n = _norm(ch_variances)
    ns_n = _norm(noise_scores)

    bars1 = ax.bar(x - width, hf_n, width, label="HF power ratio (>30 Hz, norm.)", color="tomato", alpha=0.85)
    bars2 = ax.bar(x, cv_n, width, label="Channel variance (norm.)", color="steelblue", alpha=0.85)
    bars3 = ax.bar(x + width, ns_n, width, label="Noise score (norm.)", color="mediumpurple", alpha=0.85)

    # Annotate with raw noise scores
    for i, (b, ns) in enumerate(zip(bars3, noise_scores)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{ns:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=10)
    ax.set_ylabel("Normalised metric (0–1, higher = noisier)", fontsize=9)
    ax.set_title("Dataset Noise Band Comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_denoising_comparison_overlay(
    X_in: np.ndarray,
    X_out: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    out_path: Path,
    subject_id: int,
    pipeline_name: str,
    n_channels: int = 10,
    trial_idx: int = 0,
) -> None:
    """Compare Time-Domain signal 'In' vs 'Out' using official GEDAI style.
    
    Aesthetic:
    - MNE-style stacking: single axis with vertical offsets.
    - Colors: Red for 'In' (noisy baseline), Blue for 'Out' (denoised).
    """
    X_in = np.asarray(X_in, dtype=np.float64)
    X_out = np.asarray(X_out, dtype=np.float64)
    n_ch_plot = min(n_channels, X_in.shape[1])
    t = np.arange(X_in.shape[-1]) / sfreq
    
    fig, ax = plt.subplots(figsize=(14, 0.8 * n_ch_plot + 2))
    
    # Calculate spacing (MNE-style)
    # peak-to-peak of the noisy signal as reference
    spacing = np.max(np.ptp(X_in[trial_idx, :n_ch_plot, :], axis=1)) * 1.1
    if spacing == 0: spacing = 1.0
    offsets = np.arange(n_ch_plot)[::-1] * spacing
    
    for i in range(n_ch_plot):
        ch = ch_names[i] if i < len(ch_names) else f"Ch{i}"
        
        # In (Noisy) -> Red
        ax.plot(t, X_in[trial_idx, i, :] + offsets[i], color="red", alpha=0.6, lw=0.7, 
                label="Noisy (In)" if i == 0 else None)
        
        # Out (Cleaned) -> Blue
        ax.plot(t, X_out[trial_idx, i, :] + offsets[i], color="blue", alpha=0.8, lw=0.9, 
                label=f"Cleaned (Out - {pipeline_name})" if i == 0 else None)
        
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names[:n_ch_plot], fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"EEG Comparison (MNE-Style) | Sub {subject_id} | Trial {trial_idx} | {pipeline_name}")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(t[0], t[-1])
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_denoising_psd_comparison(
    X_in: np.ndarray,
    X_out: np.ndarray,
    sfreq: float,
    out_path: Path,
    subject_id: int,
    pipeline_name: str,
) -> None:
    """Compare trial-averaged PSD 'In' vs 'Out' (averaged across all channels)."""
    nperseg = min(256, X_in.shape[-1])
    f, p_in = welch(X_in, fs=sfreq, nperseg=nperseg, axis=-1)
    _, p_out = welch(X_out, fs=sfreq, nperseg=nperseg, axis=-1)
    
    # Average over trials and channels
    p_in_mean = 10 * np.log10(np.maximum(p_in.mean(axis=(0, 1)), 1e-30))
    p_out_mean = 10 * np.log10(np.maximum(p_out.mean(axis=(0, 1)), 1e-30))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f, p_in_mean, color="gray", lw=1.5, label="In (Bandpass)")
    ax.plot(f, p_out_mean, color="blue", lw=2.0, label=f"Out ({pipeline_name})")
    
    ax.axvspan(8, 12, alpha=0.1, color="green", label="Alpha Band")
    ax.axvspan(13, 30, alpha=0.1, color="orange", label="Beta Band")
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"PSD Comparison: Bandpass vs {pipeline_name} | Subject {subject_id}")
    ax.legend()
    ax.set_xlim(0, min(100, sfreq/2))
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
