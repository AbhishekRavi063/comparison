from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def plot_prepost_overlay_static(
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
    ch_names: List[str],
    times: np.ndarray,
    title: str,
    out_path: Path,
    duration: Optional[float] = None,
    mag_scale: float = 5.0,
) -> None:
    """Pre–post denoising overlay: original vs denoised trace for each channel independently.

    Matches the style of GEDAI's compare.py (github.com/neurotuning/gedai): each channel
    is plotted with vertical offset; red = original (noisy), blue = denoised (clean).
    Saves a static PNG (no interactive window).

    Parameters
    ----------
    noisy_data : np.ndarray
        Shape (n_channels, n_times), original (e.g. bandpassed) data.
    clean_data : np.ndarray
        Shape (n_channels, n_times), denoised data.
    ch_names : list of str
        Channel names for y-axis labels.
    times : np.ndarray
        Time vector in seconds, length n_times.
    title : str
        Figure title.
    out_path : Path
        Where to save the PNG.
    duration : float, optional
        Plot only the first `duration` seconds. If None, plot full length.
    mag_scale : float
        Scale factor for trace amplitude (for visibility).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_chans = noisy_data.shape[0]
    if duration is not None:
        end_idx = np.searchsorted(times, duration)
        end_idx = min(end_idx, noisy_data.shape[1])
    else:
        end_idx = noisy_data.shape[1]
    start_idx = 0
    t_plot = times[start_idx:end_idx]
    noisy_plot = noisy_data[:, start_idx:end_idx]
    clean_plot = clean_data[:, start_idx:end_idx]

    spacing = np.max(np.ptp(noisy_plot, axis=1)) * mag_scale * 1.2
    if spacing <= 0:
        spacing = np.max(np.ptp(clean_plot, axis=1)) * mag_scale * 1.2
    if spacing <= 0:
        spacing = 1.0 * mag_scale
    offsets = np.arange(n_chans) * spacing
    ylim = (-spacing, offsets[-1] + spacing)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * n_chans)))
    ax.set_ylim(ylim)

    # Use different line styles so both traces are visible even when identical (e.g. retention fallback).
    # Red = original (dashed), blue = denoised (solid); dashed shows through when they overlap.
    for ch in range(n_chans):
        ax.plot(
            t_plot,
            noisy_plot[ch] * mag_scale + offsets[ch],
            color="red",
            alpha=0.5,
            linewidth=1.5,
            linestyle="-",
            label="Original" if ch == 0 else None,
        )
        ax.plot(
            t_plot,
            clean_plot[ch] * mag_scale + offsets[ch],
            color="blue",
            alpha=0.8,
            linewidth=1.0,
            linestyle="-",
            label="Denoised" if ch == 0 else None,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlim(t_plot[0], t_plot[-1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_signal_overlays(
    t: np.ndarray,
    signals: Dict[str, np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    """Plot time-domain overlays for different denoising methods."""
    plt.figure(figsize=(8, 4))
    for label, sig in signals.items():
        plt.plot(t, sig, label=label, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_removed_noise(
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
    ch_names: List[str],
    times: np.ndarray,
    title: str,
    out_path: Path,
    duration: Optional[float] = None,
    mag_scale: float = 5.0,
) -> None:
    """Plot the noise that was removed (EEG_in - EEG_out) for each channel."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_chans = noisy_data.shape[0]
    if duration is not None:
        end_idx = np.searchsorted(times, duration)
        end_idx = min(end_idx, noisy_data.shape[1])
    else:
        end_idx = noisy_data.shape[1]
    start_idx = 0
    t_plot = times[start_idx:end_idx]
    
    # Calculate the exact noise removed by the algorithm (EEG_in - EEG_out)
    removed_noise = noisy_data[:, start_idx:end_idx] - clean_data[:, start_idx:end_idx]

    spacing = np.max(np.ptp(removed_noise, axis=1)) * mag_scale * 1.2
    if spacing <= 0:
        spacing = 1.0 * mag_scale
    offsets = np.arange(n_chans) * spacing
    ylim = (-spacing, offsets[-1] + spacing)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * n_chans)))
    ax.set_ylim(ylim)

    for ch in range(n_chans):
        ax.plot(
            t_plot,
            removed_noise[ch] * mag_scale + offsets[ch],
            color="black",
            alpha=0.7,
            linewidth=1.0,
            linestyle="-",
            label="Removed Noise (In - Out)" if ch == 0 else None,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlim(t_plot[0], t_plot[-1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_psd_comparison(
    signals: Dict[str, np.ndarray],
    sfreq: float,
    bands: Dict[str, tuple],
    title: str,
    out_path: Path,
) -> None:
    """Plot PSD comparison for different denoising methods."""
    plt.figure(figsize=(8, 4))
    for label, sig in signals.items():
        freqs, psd = welch(sig, fs=sfreq, nperseg=min(1024, len(sig)))
        plt.semilogy(freqs, psd, label=label)

    for band_name, (fmin, fmax) in bands.items():
        plt.axvspan(fmin, fmax, color="grey", alpha=0.1, label=f"{band_name} band")

    handles, labels = plt.gca().get_legend_handles_labels()
    # Deduplicate legend entries
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (a.u.)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

