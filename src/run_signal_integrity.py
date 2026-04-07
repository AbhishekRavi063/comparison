from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import ExperimentConfig
from .denoising.pipelines import bandpass_filter, apply_icalabel, apply_gedai
from .io.dataset import NpzMotorImageryDataset
from .plots.signal_integrity import (
    plot_signal_overlays,
    plot_psd_comparison,
    plot_prepost_overlay_static,
    plot_removed_noise,
)


def _find_channel_indices(ch_names: List[str], targets: List[str]) -> List[int]:
    indices: List[int] = []
    lower_map = {ch.lower(): i for i, ch in enumerate(ch_names)}
    for t in targets:
        idx = lower_map.get(t.lower())
        if idx is not None:
            indices.append(idx)
    if not indices:
        # Fallback: first channel
        indices = [0]
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run signal integrity analysis (time + PSD) for one subject."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_alljoined_smoke_1sub.yml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help="Subject ID to use (defaults to first in config.subjects).",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial index to visualize (default: 0).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="If set and GEDAI is enabled, also open GEDAI's interactive overlay (gedai.viz.compare).",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dataset = NpzMotorImageryDataset(
        data_root=cfg.data_root,
        subjects=cfg.subjects,
        float_dtype=cfg.memory.float_dtype,
    )

    # Select subject
    subj_id = args.subject or cfg.subjects[0]
    subj_data = None
    for sid, data in dataset.iter_subjects():
        if sid == subj_id:
            subj_data = data
            break
    if subj_data is None:
        raise ValueError(f"Subject {subj_id} not found in dataset.")

    X, sfreq, ch_names = subj_data.X, subj_data.sfreq, subj_data.ch_names
    trial_idx = int(np.clip(args.trial, 0, X.shape[0] - 1))
    n_times = X.shape[2]
    times = np.arange(n_times) / sfreq

    # Channel selection for single-channel plots
    target_chs = cfg.signal_integrity.channels_of_interest
    ch_idxs = _find_channel_indices(ch_names, target_chs)
    ch_idx = ch_idxs[0]
    ch_name = ch_names[ch_idx]

    # Bandpassed data for selected trial (all channels) — "original" for pre–post overlay
    X_bp = bandpass_filter(
        X[trial_idx : trial_idx + 1],
        sfreq=sfreq,
        l_freq=cfg.bandpass.l_freq,
        h_freq=cfg.bandpass.h_freq,
    )
    # (1, n_channels, n_times) -> (n_channels, n_times)
    original_all_chs = X_bp[0]
    bp_sig = original_all_chs[ch_idx]
    raw_sig = X[trial_idx, ch_idx]

    signals_time: Dict[str, np.ndarray] = {"raw": raw_sig, "bandpass": bp_sig}

    results_dir = cfg.results_root / "figures" / "signal_integrity"
    results_dir.mkdir(parents=True, exist_ok=True)
    segment_duration = cfg.signal_integrity.segment_duration_s

    # ----- Pre–post denoising overlay (professor-recommended: original vs denoised per channel) -----
    # Style matches GEDAI compare.py: each channel independently, red=original, blue=denoised.
    if cfg.denoising.use_icalabel:
        X_ica = apply_icalabel(
            X,
            sfreq=sfreq,
            ch_names=ch_names,
            l_freq=cfg.bandpass.l_freq,
            h_freq=cfg.bandpass.h_freq,
        )
        clean_icalabel = X_ica[trial_idx]  # (n_channels, n_times)
        plot_prepost_overlay_static(
            noisy_data=original_all_chs,
            clean_data=clean_icalabel,
            ch_names=ch_names,
            times=times,
            title=f"Pre–post denoising (ICALabel) – subj {subj_id}, trial {trial_idx}",
            out_path=results_dir / f"prepost_overlay_subj{subj_id}_trial{trial_idx}_icalabel.png",
            duration=segment_duration,
        )
        plot_removed_noise(
            noisy_data=original_all_chs,
            clean_data=clean_icalabel,
            ch_names=ch_names,
            times=times,
            title=f"Removed Noise (ICALabel) – subj {subj_id}, trial {trial_idx}",
            out_path=results_dir / f"removed_noise_subj{subj_id}_trial{trial_idx}_icalabel.png",
            duration=segment_duration,
        )
        signals_time["icalabel"] = clean_icalabel[ch_idx]

    if cfg.denoising.use_gedai:
        X_gd = apply_gedai(
            X,
            sfreq=sfreq,
            ch_names=ch_names,
            l_freq=cfg.bandpass.l_freq,
            h_freq=cfg.bandpass.h_freq,
        )
        clean_gedai = X_gd[trial_idx]  # (n_channels, n_times)
        plot_prepost_overlay_static(
            noisy_data=original_all_chs,
            clean_data=clean_gedai,
            ch_names=ch_names,
            times=times,
            title=f"Pre–post denoising (GEDAI) – subj {subj_id}, trial {trial_idx}",
            out_path=results_dir / f"prepost_overlay_subj{subj_id}_trial{trial_idx}_gedai.png",
            duration=segment_duration,
        )
        plot_removed_noise(
            noisy_data=original_all_chs,
            clean_data=clean_gedai,
            ch_names=ch_names,
            times=times,
            title=f"Removed Noise (GEDAI) – subj {subj_id}, trial {trial_idx}",
            out_path=results_dir / f"removed_noise_subj{subj_id}_trial{trial_idx}_gedai.png",
            duration=segment_duration,
        )
        signals_time["gedai"] = clean_gedai[ch_idx]
        if args.interactive:
            try:
                import mne
                from gedai.viz.compare import plot_mne_style_overlay_interactive
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
                raw_noisy = mne.io.RawArray(original_all_chs, info, verbose="ERROR")
                raw_clean = mne.io.RawArray(clean_gedai, info, verbose="ERROR")
                plot_mne_style_overlay_interactive(
                    raw_noisy, raw_clean,
                    title=f"GEDAI pre–post (interactive) – subj {subj_id}",
                    duration=segment_duration,
                )
            except Exception as e:
                print(f"Interactive GEDAI overlay skipped: {e}")

    # Single-channel time overlay (all conditions on one plot)
    title = f"Time-domain overlays – subj {subj_id}, trial {trial_idx}, ch {ch_name}"
    plot_signal_overlays(
        t=times,
        signals=signals_time,
        title=title,
        out_path=results_dir / f"time_subj{subj_id}_trial{trial_idx}_{ch_name}.png",
    )

    bands = {
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
    }
    title_psd = f"PSD comparison – subj {subj_id}, ch {ch_name}"
    plot_psd_comparison(
        signals=signals_time,
        sfreq=sfreq,
        bands=bands,
        title=title_psd,
        out_path=results_dir / f"psd_subj{subj_id}_{ch_name}.png",
    )


if __name__ == "__main__":
    main()

