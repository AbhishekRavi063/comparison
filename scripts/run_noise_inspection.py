#!/usr/bin/env python3
"""
run_noise_inspection.py
=======================
End-to-end noise inspection and preprocessing runner for three EEG datasets.

Steps:
  1. Download N subjects from each dataset (OpenBMI / Cho2017 / BNCI2014_001)
  2. Run noise diagnostics (amplitude, PSD, HF power, channel variance, spikes)
  3. Generate and save diagnostic plots per dataset
  4. Normalise + rank datasets by noise score
  5. Print DATASET NOISE SUMMARY and save to results/dataset_noise_inspection/noise_summary.txt
  6. (Optional) Preprocess and export ALL subjects to .npz at 250 Hz

Usage — quick inspection (3 subjects each, no full preprocessing):
    cd /Users/abhishekr/Documents/EEG/comparison
    source .venv/bin/activate
    export HOME="$PWD/.mne_home"  && export MPLBACKEND=Agg
    python scripts/run_noise_inspection.py --n-subjects 3

Usage — inspection + full preprocessing of all subjects:
    python scripts/run_noise_inspection.py --n-subjects 3 --preprocess-all
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── project root on PYTHONPATH ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset_noise_inspection"

DATASETS = {
    "openbmi": {
        "label": "OpenBMI (Lee2019_MI)",
        "moabb_cls": "Lee2019_MI",
        "processed_dir": PROJECT_ROOT / "data" / "openbmi" / "processed",
        "raw_dir": PROJECT_ROOT / "data" / "openbmi" / "raw",
        "sfreq_orig": 1000,
        "n_ch": 62,
        "n_subjects_full": 54,
        "prepare_module": "src.data.prepare_openbmi",
    },
    "cho2017": {
        "label": "Cho2017 (GigaDB)",
        "moabb_cls": "Cho2017",
        "processed_dir": PROJECT_ROOT / "data" / "cho2017" / "processed",
        "raw_dir": PROJECT_ROOT / "data" / "cho2017" / "raw",
        "sfreq_orig": 512,
        "n_ch": 64,
        "n_subjects_full": 52,
        "prepare_module": "src.data.prepare_cho2017",
    },
    "gigadb": {
        "label": "BNCI2014_001 (reference)",
        "moabb_cls": "BNCI2014_001",
        "processed_dir": PROJECT_ROOT / "data" / "gigadb" / "processed",
        "raw_dir": PROJECT_ROOT / "data" / "gigadb" / "raw",
        "sfreq_orig": 250,
        "n_ch": 22,
        "n_subjects_full": 9,
        "prepare_module": "src.data.prepare_bnci2014_001_for_noise",
    },
}


# ── helper: load a processed npz subject ─────────────────────────────────────

def _load_npz(path: Path):
    with np.load(path, allow_pickle=True) as d:
        X = d["X"].astype(np.float64, copy=False)
        y = d["y"].astype(int)
        sfreq = float(d["sfreq"].item() if np.ndim(d["sfreq"]) else d["sfreq"])
        ch_names_raw = d["ch_names"]
    ch_names = [str(c) for c in (ch_names_raw.tolist() if isinstance(ch_names_raw, np.ndarray) else ch_names_raw)]
    return X, y, sfreq, ch_names


# ── step 1+2: download + inspect N subjects per dataset ──────────────────────

def _inspect_dataset(ds_key: str, ds: dict, n_subjects: int) -> dict:
    """Prepare n_subjects, run noise diagnostics on each, return aggregated stats."""
    from src.data.dataset_noise_inspection import (
        compute_noise_diagnostics,
        plot_raw_overlay,
        plot_psd,
    )

    label = ds["label"]
    processed_dir = ds["processed_dir"]
    raw_dir = ds["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_dir = RESULTS_DIR / ds_key
    out_dir.mkdir(parents=True, exist_ok=True)

    prepare_fn = _get_prepare_fn(ds["prepare_module"])

    all_hf_ratio: List[float] = []
    all_ch_var: List[float] = []
    all_spikes: List[int] = []
    all_amp_mean: List[float] = []
    all_noise_raw: List[float] = []
    sfreq_final = 250.0
    ch_names_out = None

    for sid in range(1, n_subjects + 1):
        npz_path = processed_dir / f"subject_{sid}.npz"
        if not npz_path.exists():
            print(f"  Downloading {label} subject {sid}...", flush=True)
            try:
                prepare_fn(subject=sid, out_root=processed_dir)
            except Exception as e:
                warnings.warn(f"  ✗ subject {sid} download failed: {e}")
                continue
        else:
            print(f"  Found cached {label} subject {sid} — skipping download.", flush=True)

        X, y, sfreq, ch_names = _load_npz(npz_path)
        sfreq_final = sfreq
        if ch_names_out is None:
            ch_names_out = ch_names

        print(f"  Running diagnostics on {label} subject {sid} (shape={X.shape})...", flush=True)
        diag = compute_noise_diagnostics(X, sfreq, ch_names)

        all_hf_ratio.append(diag["hf_ratio"])
        all_ch_var.append(diag["channel_variance"])
        all_spikes.append(diag["spike_count"])
        all_amp_mean.append(diag["amp_mean"])
        all_noise_raw.append(diag["noise_score_raw"])

        # plots for first subject only
        if sid == 1:
            plot_raw_overlay(
                X, sfreq, ch_names,
                out_path=out_dir / f"raw_overlay_subj{sid}.png",
                subject_id=sid,
                dataset_label=label,
            )
            plot_psd(
                diag["f"], diag["psd_channel_mean"],
                out_path=out_dir / f"psd_subj{sid}.png",
                subject_id=sid,
                sfreq=sfreq,
                dataset_label=label,
            )
            print(f"  ✓ Plots saved to {out_dir}", flush=True)

        del X, y
        gc.collect()

    if not all_hf_ratio:
        return {}

    return {
        "label": label,
        "sfreq": sfreq_final,
        "n_channels": len(ch_names_out) if ch_names_out else ds["n_ch"],
        "n_subjects_inspected": len(all_hf_ratio),
        "mean_hf_ratio": float(np.mean(all_hf_ratio)),
        "mean_ch_var": float(np.mean(all_ch_var)),
        "mean_spikes": float(np.mean(all_spikes)),
        "mean_amp_mean": float(np.mean(all_amp_mean)),
        "noise_score_raw": float(np.mean(all_noise_raw)),
    }


def _get_prepare_fn(module_path: str):
    """Dynamically import _prepare_subject from a module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return mod._prepare_subject


# ── step 3: normalise + rank ──────────────────────────────────────────────────

def _normalise_and_rank(results: Dict[str, dict]) -> List[tuple]:
    """Normalise noise_score_raw across datasets and rank (highest first)."""
    keys = [k for k, v in results.items() if v]
    raw_scores = np.array([results[k]["noise_score_raw"] for k in keys])
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s - min_s < 1e-12:
        norm_scores = np.ones(len(keys))
    else:
        norm_scores = (raw_scores - min_s) / (max_s - min_s)
    ranked = sorted(
        zip(keys, norm_scores, raw_scores),
        key=lambda t: t[2],
        reverse=True,
    )
    return ranked


# ── step 4: noise band comparison plot ───────────────────────────────────────

def _plot_comparison(results: Dict[str, dict]) -> None:
    from src.data.dataset_noise_inspection import plot_noise_band_comparison

    labels = [v["label"] for v in results.values() if v]
    hf = [v["mean_hf_ratio"] for v in results.values() if v]
    cv = [v["mean_ch_var"] for v in results.values() if v]
    ns = [v["noise_score_raw"] for v in results.values() if v]

    if not labels:
        return

    out_path = RESULTS_DIR / "noise_band_comparison.png"
    plot_noise_band_comparison(labels, hf, cv, ns, out_path)
    print(f"\n  ✓ Noise band comparison plot saved: {out_path}", flush=True)


# ── step 5: print + save summary ─────────────────────────────────────────────

def _print_summary(results: Dict[str, dict], ranked: List[tuple]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("DATASET NOISE SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    for k, norm_score, raw_score in ranked:
        v = results[k]
        lines.append(f"{v['label']}:")
        lines.append(f"  sampling rate        : {v['sfreq']:.0f} Hz")
        lines.append(f"  channels             : {v['n_channels']}")
        lines.append(f"  subjects inspected   : {v['n_subjects_inspected']}")
        lines.append(f"  mean HF ratio (>30Hz): {v['mean_hf_ratio']:.5f}")
        lines.append(f"  mean channel variance: {v['mean_ch_var']:.5e}")
        lines.append(f"  mean spike count     : {v['mean_spikes']:.1f}")
        lines.append(f"  noise score (raw)    : {raw_score:.5f}")
        lines.append(f"  noise score (norm.)  : {norm_score:.4f}  ← 0=cleanest, 1=noisiest")
        lines.append("")

    lines.append("-" * 60)
    lines.append("RANKING  (most noisy → least noisy):")
    for rank, (k, norm_score, raw_score) in enumerate(ranked, 1):
        lines.append(f"  {rank}. {results[k]['label']}  (noise score: {norm_score:.4f})")
    lines.append("")
    lines.append(f"Conclusion:")
    if ranked:
        most_noisy = results[ranked[0][0]]["label"]
        least_noisy = results[ranked[-1][0]]["label"]
        lines.append(f"  Most noisy dataset  = {most_noisy}")
        lines.append(f"  Least noisy dataset = {least_noisy}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    return report


# ── step 6: preprocess all subjects ──────────────────────────────────────────

def _preprocess_all(ds_key: str, ds: dict) -> None:
    prepare_fn = _get_prepare_fn(ds["prepare_module"])
    processed_dir = ds["processed_dir"]
    n_total = ds["n_subjects_full"]
    label = ds["label"]

    print(f"\nPreprocessing all {n_total} subjects for {label}...", flush=True)
    for sid in range(1, n_total + 1):
        npz_path = processed_dir / f"subject_{sid}.npz"
        if npz_path.exists():
            print(f"  subject {sid:3d}: cached — skip", flush=True)
            continue
        try:
            prepare_fn(subject=sid, out_root=processed_dir)
            print(f"  subject {sid:3d}: ✓", flush=True)
        except Exception as e:
            warnings.warn(f"  subject {sid:3d}: ✗ ({e})")
        gc.collect()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # MNE data dir
    project_root = Path(__file__).resolve().parents[1]
    default_mne = project_root / ".mne_home" / "MNE-data"
    default_mne.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DATA", str(default_mne))
    os.environ.setdefault("MPLBACKEND", "Agg")

    parser = argparse.ArgumentParser(
        description="Download, inspect, and optionally preprocess three MI EEG datasets."
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=3,
        metavar="N",
        help="Number of subjects to download for noise inspection (default: 3).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASETS.keys()),
        help=f"Which datasets to inspect. Choices: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--preprocess-all",
        action="store_true",
        help="After inspection, download and preprocess all subjects for each dataset.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = {k: DATASETS[k] for k in args.datasets if k in DATASETS}
    if not selected:
        print("No valid datasets selected. Choices:", list(DATASETS.keys()))
        sys.exit(1)

    print("\n" + "=" * 60, flush=True)
    print("EEG DATASET NOISE INSPECTION", flush=True)
    print(f"Datasets   : {list(selected.keys())}", flush=True)
    print(f"N subjects  : {args.n_subjects} (inspection)", flush=True)
    print(f"Results dir : {RESULTS_DIR}", flush=True)
    print("=" * 60 + "\n", flush=True)

    results: Dict[str, dict] = {}
    for ds_key, ds in selected.items():
        print(f"\n{'─'*50}", flush=True)
        print(f"Dataset: {ds['label']}", flush=True)
        print(f"{'─'*50}", flush=True)
        t0 = time.time()
        res = _inspect_dataset(ds_key, ds, n_subjects=args.n_subjects)
        elapsed = time.time() - t0
        if res:
            print(f"  ✓ Inspection done in {elapsed:.1f}s", flush=True)
        else:
            print(f"  ✗ Inspection produced no results.", flush=True)
        results[ds_key] = res

    # Rank + compare
    ranked = _normalise_and_rank(results)
    _plot_comparison(results)

    # Summary
    report = _print_summary(results, ranked)
    summary_path = RESULTS_DIR / "noise_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    print(f"\n✓ Noise summary saved: {summary_path}", flush=True)

    # Full preprocessing (optional)
    if args.preprocess_all:
        print("\n" + "=" * 60, flush=True)
        print("PREPROCESSING ALL SUBJECTS", flush=True)
        print("=" * 60, flush=True)
        for ds_key, ds in selected.items():
            _preprocess_all(ds_key, ds)
        print("\n✓ All subjects preprocessed.", flush=True)
    else:
        print(
            "\n[INFO] Full preprocessing skipped. "
            "Re-run with --preprocess-all to export all subjects.",
            flush=True,
        )


if __name__ == "__main__":
    main()
