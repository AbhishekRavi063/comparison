from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYGEDAI_FORCE_CPU", "1")

from src.data.prepare_eeg_emg_mrcp import (  # noqa: E402
    KNOWN_EEG_CHANNELS,
    _find_eeg_columns,
    _find_trigger_column,
    _session_files,
    _subject_dir,
)
from src.denoising.pipelines import (  # noqa: E402
    MRCP_GEDAI_HPF_HZ,
    MRCP_GEDAI_NOISE_MULTIPLIER,
    MRCP_GEDAI_RESAMPLE_HZ,
    MRCP_GEDAI_WAVELET_LEVEL,
    MRCP_MOTOR_CHANNELS,
)
from gedai import Gedai  # noqa: E402


def _load_raw(eeg_path: Path, sfreq: float) -> mne.io.RawArray:
    df = pd.read_csv(eeg_path)
    trigger_col = _find_trigger_column(df)
    eeg_cols = _find_eeg_columns(df, trigger_col)
    X = df[eeg_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32).T
    ch_names = list(KNOWN_EEG_CHANNELS) if X.shape[0] == len(KNOWN_EEG_CHANNELS) else [str(c) for c in eeg_cols]
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types="eeg")
    try:
        info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore", verbose=False)
    except Exception:
        pass
    return mne.io.RawArray(X * 1e-6, info, verbose="ERROR")


def _motor_indices(ch_names: list[str]) -> list[int]:
    wanted = {c.upper() for c in MRCP_MOTOR_CHANNELS}
    return [i for i, c in enumerate(ch_names) if c.upper() in wanted]


def _plot_overlay(times: np.ndarray, ref: np.ndarray, clean: np.ndarray, removed: np.ndarray, title: str, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(times, ref, label="Reference", lw=1.3, color="#1f77b4")
    axes[0].plot(times, clean, label="GEDAI", lw=1.1, color="#d62728", alpha=0.9)
    axes[0].set_title(title)
    axes[0].set_ylabel("uV")
    axes[0].legend(loc="upper right")

    axes[1].plot(times, removed, lw=1.0, color="#2ca02c")
    axes[1].set_ylabel("uV")
    axes[1].set_title("Removed Component")

    axes[2].plot(times, ref - ref.mean(), lw=1.0, color="#1f77b4", alpha=0.7, label="Reference (demeaned)")
    axes[2].plot(times, clean - clean.mean(), lw=1.0, color="#d62728", alpha=0.7, label="GEDAI (demeaned)")
    axes[2].set_ylabel("uV")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_psd(ref: np.ndarray, clean: np.ndarray, removed: np.ndarray, sfreq: float, title: str, out: Path) -> None:
    nper = max(128, int(min(len(ref), round(sfreq * 8))))
    freqs, psd_ref = welch(ref, fs=sfreq, nperseg=nper)
    _, psd_clean = welch(clean, fs=sfreq, nperseg=nper)
    _, psd_removed = welch(removed, fs=sfreq, nperseg=nper)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs, psd_ref, label="Reference", color="#1f77b4", lw=1.3)
    ax.semilogy(freqs, psd_clean, label="GEDAI", color="#d62728", lw=1.1)
    ax.semilogy(freqs, psd_removed, label="Removed", color="#2ca02c", lw=1.0)
    for x in (0.1, 0.5, 1.0, 8.0, 12.0, 30.0):
        ax.axvline(x, color="0.75", lw=0.8, ls="--")
    ax.set_xlim(0, 35)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(loc="upper right")
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100])
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=2)
    parser.add_argument("--files", type=int, nargs="*", default=[1, 3])
    parser.add_argument("--raw-root", type=str, default="data/EEG and EMG Dataset for Analyzing Movement-Related")
    parser.add_argument("--out-dir", type=str, default="results/mrcp_fix_plots")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir / f"subject_{args.subject:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_dir = _subject_dir(Path(args.raw_root), args.subject)
    eeg_files = _session_files(subject_dir)

    for file_idx in args.files:
        eeg_path = eeg_files[file_idx - 1]
        raw = _load_raw(eeg_path, 128.0)
        raw.resample(MRCP_GEDAI_RESAMPLE_HZ, npad="auto", verbose=False)
        raw.filter(l_freq=MRCP_GEDAI_HPF_HZ, h_freq=None, method="fir", fir_design="firwin", picks="eeg", verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        raw_ref = raw.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gedai = Gedai(
                wavelet_type="haar",
                wavelet_level=MRCP_GEDAI_WAVELET_LEVEL,
                wavelet_low_cutoff=0.1,
                epoch_size_in_cycles=12,
                highpass_cutoff=0.05,
                signal_type="eeg",
                preliminary_broadband_noise_multiplier=None,
            )
            gedai.fit_raw(raw, noise_multiplier=float(MRCP_GEDAI_NOISE_MULTIPLIER), n_jobs=1, verbose=False)
            raw_clean = gedai.transform_raw(raw, n_jobs=1, verbose=False)

        fit_figs = gedai.plot_fit()
        for i, fig in enumerate(fit_figs, start=1):
            fig.savefig(out_dir / f"{eeg_path.stem}_fit_band_{i}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        raw_ref.resample(128.0, npad="auto", verbose=False)
        raw_clean.resample(128.0, npad="auto", verbose=False)
        raw_ref.filter(l_freq=0.1, h_freq=1.0, method="fir", fir_design="firwin", picks="eeg", verbose=False)
        raw_clean.filter(l_freq=0.1, h_freq=1.0, method="fir", fir_design="firwin", picks="eeg", verbose=False)

        Xref = raw_ref.get_data().astype(np.float32) * 1e6
        Xclean = raw_clean.get_data().astype(np.float32) * 1e6
        Xremoved = Xref - Xclean
        idx = _motor_indices(raw_ref.ch_names)
        motor_ref = np.median(Xref[idx], axis=0)
        motor_clean = np.median(Xclean[idx], axis=0)
        motor_removed = np.median(Xremoved[idx], axis=0)
        times = np.arange(motor_ref.shape[0]) / 128.0

        title = f"Subject {args.subject:02d} {eeg_path.stem} Motor-Channel Median (0.1-1 Hz)"
        _plot_overlay(times, motor_ref, motor_clean, motor_removed, title, out_dir / f"{eeg_path.stem}_overlay_motor.png")
        _plot_psd(motor_ref, motor_clean, motor_removed, 128.0, title, out_dir / f"{eeg_path.stem}_psd_motor.png")


if __name__ == "__main__":
    main()
