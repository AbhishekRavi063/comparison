#!/usr/bin/env python3
"""
After a full run (pass 1 or merged): check whether any pipeline removed actual brain signal.
Runs preservation check on N subjects, then writes a report:
- % subjects with over-removal (alpha or beta ratio < 0.75)
- Which frequency band was most reduced (alpha 8-12 Hz vs beta 13-30 Hz)
- Mean/min ratio per pipeline, per band
- Details per subject if requested.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.denoising.pipelines import bandpass_filter, apply_icalabel, apply_gedai
from src.io.dataset import NpzMotorImageryDataset

ALPHA_BAND = (8.0, 12.0)
BETA_BAND = (13.0, 30.0)
THRESHOLD = 0.75  # ratio below this = possible over-removal


def band_power(sig: np.ndarray, sfreq: float, lo: float, hi: float) -> float:
    nperseg = min(256, len(sig) // 4, 1024)
    f, p = welch(sig, fs=sfreq, nperseg=nperseg)
    mask = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[mask], f[mask]))


def check_one_subject(
    cfg: ExperimentConfig,
    dataset: NpzMotorImageryDataset,
    subject_id: int,
    channel: str,
) -> dict:
    """Return dict with pipeline -> {alpha_ratio, beta_ratio, over_removed_alpha, over_removed_beta}."""
    subj_data = None
    for sid, data in dataset.iter_subjects():
        if sid == subject_id:
            subj_data = data
            break
    if subj_data is None:
        return {}
    X = subj_data.X
    sfreq = subj_data.sfreq
    ch_names = list(subj_data.ch_names)
    ch_upper = [c.upper() for c in ch_names]
    try:
        ch_idx = ch_upper.index(channel.upper())
    except ValueError:
        ch_idx = 0
    l_freq, h_freq = cfg.bandpass.l_freq, cfg.bandpass.h_freq
    X_bp = bandpass_filter(X, sfreq, l_freq, h_freq)
    ref_alpha = band_power(X_bp[:, ch_idx, :].mean(axis=0), sfreq, *ALPHA_BAND)
    ref_beta = band_power(X_bp[:, ch_idx, :].mean(axis=0), sfreq, *BETA_BAND)
    out = {}
    if cfg.denoising.use_icalabel:
        X_ica = apply_icalabel(X, sfreq, ch_names, l_freq, h_freq)
        sig = X_ica[:, ch_idx, :].mean(axis=0)
        r_a = (band_power(sig, sfreq, *ALPHA_BAND) / ref_alpha) if ref_alpha > 0 else 1.0
        r_b = (band_power(sig, sfreq, *BETA_BAND) / ref_beta) if ref_beta > 0 else 1.0
        out["icalabel"] = {"alpha_ratio": r_a, "beta_ratio": r_b}
    if cfg.denoising.use_gedai:
        X_gd = apply_gedai(X, sfreq, ch_names, l_freq, h_freq)
        sig = X_gd[:, ch_idx, :].mean(axis=0)
        r_a = (band_power(sig, sfreq, *ALPHA_BAND) / ref_alpha) if ref_alpha > 0 else 1.0
        r_b = (band_power(sig, sfreq, *BETA_BAND) / ref_beta) if ref_beta > 0 else 1.0
        out["gedai"] = {"alpha_ratio": r_a, "beta_ratio": r_b}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report brain signal removal across subjects (alpha/beta preservation)."
    )
    parser.add_argument(
        "--config",
        default="config/config_real_physionet_full.yml",
        help="Config used for the run (same data_root, denoising flags).",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=10,
        help="Number of subjects to check (first N from config). Use 109 for full.",
    )
    parser.add_argument(
        "--channel",
        default="C3",
        help="Channel for band power (e.g. C3, C4).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write report to this path (e.g. results/physionet_full/brain_signal_removal_report.md).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-subject ratios to stdout.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    subject_list = list(cfg.subjects)[: args.n_subjects]
    dataset = NpzMotorImageryDataset(
        data_root=cfg.data_root,
        subjects=subject_list,
        float_dtype=cfg.memory.float_dtype,
    )

    # Collect ratios per subject, per pipeline
    results = {}  # pipeline -> list of (alpha_ratio, beta_ratio) per subject
    for p in ["icalabel", "gedai"]:
        if (p == "icalabel" and not cfg.denoising.use_icalabel) or (
            p == "gedai" and not cfg.denoising.use_gedai
        ):
            continue
        results[p] = []
    subject_ids_checked = []

    for sid in subject_list:
        row = check_one_subject(cfg, dataset, sid, args.channel)
        subject_ids_checked.append(sid)
        for p, vals in row.items():
            if p not in results:
                results[p] = []
            results[p].append((vals["alpha_ratio"], vals["beta_ratio"]))
            if args.verbose:
                print(f"  Subject {sid} {p}: alpha={vals['alpha_ratio']:.4f} beta={vals['beta_ratio']:.4f}")

    # Build report
    lines = [
        "# Brain signal removal report",
        "",
        f"**Config:** {args.config}",
        f"**Subjects checked:** {len(subject_ids_checked)} (IDs: {subject_ids_checked[:5]}{'...' if len(subject_ids_checked) > 5 else ''})",
        f"**Channel:** {args.channel}",
        f"**Threshold:** ratio < {THRESHOLD} = possible over-removal",
        "",
        "## Frequency bands",
        "- **Alpha (8–12 Hz):** mu/alpha rhythm",
        "- **Beta (13–30 Hz):** beta band",
        "",
        "## Summary",
        "",
    ]

    for pipeline in ["icalabel", "gedai"]:
        if pipeline not in results:
            continue
        arr = np.array(results[pipeline])
        alpha_ratios = arr[:, 0]
        beta_ratios = arr[:, 1]
        n = len(alpha_ratios)
        pct_alpha_removed = 100.0 * (alpha_ratios < THRESHOLD).sum() / n
        pct_beta_removed = 100.0 * (beta_ratios < THRESHOLD).sum() / n
        any_removed = (alpha_ratios < THRESHOLD) | (beta_ratios < THRESHOLD)
        pct_any_removed = 100.0 * any_removed.sum() / n

        lines.append(f"### {pipeline.upper()}")
        lines.append("")
        lines.append(f"- **% subjects with alpha (8–12 Hz) over-removal (ratio < {THRESHOLD}):** {pct_alpha_removed:.1f}%")
        lines.append(f"- **% subjects with beta (13–30 Hz) over-removal (ratio < {THRESHOLD}):** {pct_beta_removed:.1f}%")
        lines.append(f"- **% subjects with any over-removal (alpha or beta):** {pct_any_removed:.1f}%")
        lines.append(f"- **Mean ratio alpha:** {float(np.mean(alpha_ratios)):.4f}  |  **Mean ratio beta:** {float(np.mean(beta_ratios)):.4f}")
        lines.append(f"- **Min ratio alpha:** {float(np.min(alpha_ratios)):.4f}  |  **Min ratio beta:** {float(np.min(beta_ratios)):.4f}")
        if (alpha_ratios < THRESHOLD).any() or (beta_ratios < THRESHOLD).any():
            worst_alpha = subject_ids_checked[np.argmin(alpha_ratios)]
            worst_beta = subject_ids_checked[np.argmin(beta_ratios)]
            lines.append(f"- **Worst alpha (subject):** {worst_alpha} (ratio {float(np.min(alpha_ratios)):.4f})")
            lines.append(f"- **Worst beta (subject):** {worst_beta} (ratio {float(np.min(beta_ratios)):.4f})")
        lines.append("")
        lines.append("**Interpretation:** Ratio 1.0 = full preservation; < 0.75 suggests possible removal of brain signal in that band.")
        lines.append("")

    lines.append("---")
    lines.append("**Conclusion:** If % over-removal is 0% and mean/min ratios are ≥ 0.75, pipelines did not remove actual brain signal in alpha/beta. If > 0%, inspect overlay/PSD plots for those subjects.")
    lines.append("")

    report_text = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report_text, encoding="utf-8")
        print(f"Report written to: {args.out}")
    print(report_text)


if __name__ == "__main__":
    main()
