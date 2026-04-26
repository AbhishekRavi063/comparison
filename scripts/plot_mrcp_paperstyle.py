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
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYGEDAI_FORCE_CPU", "1")

from gedai import Gedai  # noqa: E402
from src.data.prepare_eeg_emg_mrcp import (  # noqa: E402
    KNOWN_EEG_CHANNELS,
    _contiguous_trigger_onsets,
    _extract_epochs,
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
)

PAPER_CHANNELS = ["FC3", "FC1", "FCZ", "C3", "C1", "CZ", "CP3", "CP1", "CP2"]


def _load_and_clean_session(eeg_path: Path, sfreq: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(eeg_path)
    trigger_col = _find_trigger_column(df)
    eeg_cols = _find_eeg_columns(df, trigger_col)
    trigger = pd.to_numeric(df[trigger_col], errors="coerce").fillna(0).astype(int).to_numpy()
    X_cont = df[eeg_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32).T

    ch_names = list(KNOWN_EEG_CHANNELS) if X_cont.shape[0] == len(KNOWN_EEG_CHANNELS) else [str(c) for c in eeg_cols]
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types="eeg")
    try:
        info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore", verbose=False)
    except Exception:
        pass

    raw = mne.io.RawArray(X_cont * 1e-6, info, verbose="ERROR")
    orig_sfreq = float(sfreq)
    if abs(raw.info["sfreq"] - MRCP_GEDAI_RESAMPLE_HZ) > 0.1:
        raw.resample(MRCP_GEDAI_RESAMPLE_HZ, npad="auto", verbose=False)
    raw.filter(l_freq=MRCP_GEDAI_HPF_HZ, h_freq=None, method="fir", fir_design="firwin", picks="eeg", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    raw_ref = raw.copy()
    raw_ref.filter(l_freq=0.1, h_freq=1.0, method="fir", fir_design="firwin", picks="eeg", verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gedai = Gedai(
            wavelet_type="haar",
            wavelet_level=MRCP_GEDAI_WAVELET_LEVEL,
            wavelet_low_cutoff=0.1,
            epoch_size_in_cycles=12,
            highpass_cutoff=MRCP_GEDAI_HPF_HZ,
            signal_type="eeg",
            preliminary_broadband_noise_multiplier=None,
        )
        gedai.fit_raw(raw, noise_multiplier=float(MRCP_GEDAI_NOISE_MULTIPLIER), n_jobs=1, verbose=False)
        raw_clean = gedai.transform_raw(raw, n_jobs=1, verbose=False)

    if abs(raw_ref.info["sfreq"] - orig_sfreq) > 0.1:
        raw_ref.resample(orig_sfreq, npad="auto", verbose=False)
    if abs(raw_clean.info["sfreq"] - orig_sfreq) > 0.1:
        raw_clean.resample(orig_sfreq, npad="auto", verbose=False)

    raw_clean.filter(l_freq=0.1, h_freq=1.0, method="fir", fir_design="firwin", picks="eeg", verbose=False)

    X_ref = raw_ref.get_data().astype(np.float32, copy=False) * 1e6
    X_clean = raw_clean.get_data().astype(np.float32, copy=False) * 1e6
    return trigger, X_ref, X_clean, ch_names


def _mean_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(axis=0), X.std(axis=0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1], dtype=np.float32)


def _apply_surface_laplacian(X: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    montage = mne.channels.make_standard_montage("standard_1005")
    canonical = {name.upper(): name for name in montage.ch_names}
    ch_names_use = [canonical.get(name.upper(), name) for name in ch_names]
    info = mne.create_info(ch_names=ch_names_use, sfreq=float(sfreq), ch_types="eeg")
    try:
        info.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception:
        return X

    epochs = mne.EpochsArray(X * 1e-6, info, tmin=0.0, verbose="ERROR")
    try:
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs, verbose=False)
    except Exception as e:
        warnings.warn(f"Surface Laplacian failed ({e}); using non-Laplacian epochs.")
        return X
    return (epochs_csd.get_data().astype(np.float32, copy=False) * 1e6).astype(np.float32, copy=False)


def _zero_mean_and_detrend(X: np.ndarray) -> np.ndarray:
    # Professor's suggestion: remove per-trial linear trend and enforce zero mean
    # before averaging so slow offsets do not masquerade as MRCP structure.
    X = detrend(X, axis=-1, type="linear")
    X = X - X.mean(axis=-1, keepdims=True)
    return X.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style extended MRCP plots.")
    parser.add_argument("--subject", type=int, default=2)
    parser.add_argument("--raw-root", default="data/EEG and EMG Dataset for Analyzing Movement-Related")
    parser.add_argument("--mode", choices=("reference", "gedai"), default="reference")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--sfreq", type=float, default=128.0)
    parser.add_argument("--spatial", choices=("none", "laplacian"), default="laplacian")
    parser.add_argument(
        "--preprocess",
        choices=("none", "zero_mean", "detrend_zero_mean"),
        default="detrend_zero_mean",
    )
    parser.add_argument("--out", default="results/mrcp_fix_plots/subject_02/SUBJECT02_MRCP_paperstyle_reference.png")
    args = parser.parse_args()

    subject_dir = _subject_dir(ROOT / args.raw_root, args.subject)
    eeg_files = _session_files(subject_dir)
    if not eeg_files:
        raise FileNotFoundError(f"No EEG files found in {subject_dir}")

    move_all: list[np.ndarray] = []
    rest_all: list[np.ndarray] = []
    ch_names_final: list[str] | None = None

    for eeg_path in eeg_files:
        try:
            trigger, X_ref, X_clean, ch_names = _load_and_clean_session(eeg_path, args.sfreq)
            X_cont = X_ref if args.mode == "reference" else X_clean
            movement_onsets = _contiguous_trigger_onsets(trigger, 7711)
            rest_onsets = _contiguous_trigger_onsets(trigger, 771)
            X_move = _extract_epochs(X_cont, movement_onsets, sfreq=args.sfreq, tmin=args.tmin, tmax=args.tmax)
            X_rest = _extract_epochs(X_cont, rest_onsets, sfreq=args.sfreq, tmin=args.tmin, tmax=args.tmax)
            if len(X_move) == 0 or len(X_rest) == 0:
                continue
            move_all.append(X_move)
            rest_all.append(X_rest)
            ch_names_final = ch_names
        except Exception as e:
            warnings.warn(f"Skipping {eeg_path.name}: {e}")

    if not move_all or not rest_all or ch_names_final is None:
        raise ValueError("No usable epochs extracted for extended MRCP plot.")

    X_move = np.concatenate(move_all, axis=0)
    X_rest = np.concatenate(rest_all, axis=0)
    if args.spatial == "laplacian":
        X_move = _apply_surface_laplacian(X_move, ch_names_final, args.sfreq)
        X_rest = _apply_surface_laplacian(X_rest, ch_names_final, args.sfreq)
    if args.preprocess == "zero_mean":
        X_move = (X_move - X_move.mean(axis=-1, keepdims=True)).astype(np.float32, copy=False)
        X_rest = (X_rest - X_rest.mean(axis=-1, keepdims=True)).astype(np.float32, copy=False)
    elif args.preprocess == "detrend_zero_mean":
        X_move = _zero_mean_and_detrend(X_move)
        X_rest = _zero_mean_and_detrend(X_rest)

    channel_to_idx = {c.upper(): i for i, c in enumerate(ch_names_final)}
    selected = [c for c in PAPER_CHANNELS if c.upper() in channel_to_idx]
    idx = [channel_to_idx[c.upper()] for c in selected]
    times = np.arange(X_move.shape[-1]) / args.sfreq + args.tmin

    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True)
    axes = axes.ravel()
    for ax, ch_name, ch_idx in zip(axes, selected, idx):
        move_mean, move_std = _mean_std(X_move[:, ch_idx, :])
        rest_mean, rest_std = _mean_std(X_rest[:, ch_idx, :])
        ax.plot(times, move_mean, color="#4c6ef5", lw=1.8, label="Movement")
        ax.fill_between(times, move_mean - move_std, move_mean + move_std, color="#4c6ef5", alpha=0.23)
        ax.plot(times, rest_mean, color="#e03131", lw=1.6, label="Rest")
        ax.fill_between(times, rest_mean - rest_std, rest_mean + rest_std, color="#e03131", alpha=0.20)
        ax.axvline(0.0, color="0.45", lw=1.0, ls="--")
        ax.axhline(0.0, color="0.85", lw=0.8)
        ax.set_title(f"Electrode: {ch_name}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.15)
        ax.set_xlim(args.tmin, args.tmax)
        ax.tick_params(labelsize=8)

    for ax in axes[6:]:
        ax.set_xlabel("Time (s)")
    y_label = "CSD (a.u.)" if args.spatial == "laplacian" else "Amplitude (uV)"
    for ax in (axes[0], axes[3], axes[6]):
        ax.set_ylabel(y_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(
        f"Subject {args.subject:02d} Extended MRCP ({args.mode.title()}, 0.1-1 Hz, {args.spatial}, {args.preprocess})",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
