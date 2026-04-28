"""Post-experiment AASD analysis: preprint figures + signal preservation proof.

Run AFTER src.run_all completes:
    python -m src.run_aasd_analysis --config config/config_aasd.yml

Generates
---------
results/aasd_full/figures/
    auc_violin_csp.png
    auc_violin_tangent.png
    lateralization_index.png       ← KEY: proves GEDAI sharpens, not removes, brain signal
    alpha_retention.png            ← Shows targeted denoising (beta drops more than alpha)
    psd_all_pipelines_sub1.png    ← All pipelines on same PSD axes
    topomap_alpha_sub1.png        ← Spatial alpha pattern preserved
    summary_table.csv              ← Mean ± std AUC + LI per pipeline × backbone
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict

import numpy as np

from .config import ExperimentConfig
from .denoising.pipelines import preprocess_subject_data
from .io.dataset import NpzMotorImageryDataset
from .plots.aasd_preprint import generate_all_preprint_figures


def _load_denoised_for_subject(
    cfg: ExperimentConfig,
    subject_id: int,
    pipelines: list[str],
) -> tuple[Dict[str, np.ndarray], float, list[str], np.ndarray]:
    """Re-run denoising for one subject and return {pipeline: X_proc}.

    Used only to generate PSD and topomap figures — denoising is fast for a single subject.
    """
    dataset = NpzMotorImageryDataset(
        data_root=cfg.data_root,
        subjects=[subject_id],
        float_dtype=cfg.memory.float_dtype,
    )
    subj_data = None
    for sid, sd in dataset.iter_subjects():
        if sid == subject_id:
            subj_data = sd
            break
    if subj_data is None:
        raise RuntimeError(f"Subject {subject_id} not found in dataset.")

    X, y = subj_data.X, subj_data.y

    X_by_pipeline: Dict[str, np.ndarray] = {}
    for pipeline in pipelines:
        try:
            X_proc = preprocess_subject_data(
                X=X,
                y=y,
                sfreq=subj_data.sfreq,
                ch_names=subj_data.ch_names,
                l_freq=cfg.bandpass.l_freq,
                h_freq=cfg.bandpass.h_freq,
                denoising=pipeline,
                subject_id=subject_id,
                dataset_name=cfg.dataset_label or "",
                gedai_n_jobs=cfg.memory.n_jobs,
                data_root=cfg.data_root,
            )
            X_by_pipeline[pipeline] = X_proc
            print(f"  [analysis] {pipeline} denoising done for subject {subject_id}")
        except Exception as e:
            warnings.warn(f"[analysis] {pipeline} failed for subject {subject_id}: {e}")

    return X_by_pipeline, float(subj_data.sfreq), list(subj_data.ch_names), y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AASD preprint figures from completed experiment results."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_aasd.yml",
        help="YAML config used for the original run_all experiment.",
    )
    parser.add_argument(
        "--psd-subject",
        type=int,
        default=1,
        metavar="N",
        help="Subject ID to use for PSD + topomap plots (default: 1). "
             "Set to 0 to skip PSD/topomap (faster).",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    csv_path  = cfg.results_root / "tables" / "subject_level_performance.csv"
    out_dir   = cfg.results_root / "figures"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {csv_path}\n"
            "Run `python -m src.run_all --config {args.config}` first."
        )

    # Determine which pipelines are present in the CSV
    import pandas as pd
    df = pd.read_csv(csv_path)
    pipelines_in_csv = list(df["pipeline"].unique())

    # Optional: re-run denoising for PSD + topomap on one subject
    X_by_pipeline = None
    sfreq = 250.0
    ch_names = None
    psd_sid = int(args.psd_subject)

    if psd_sid > 0 and psd_sid in cfg.subjects:
        print(f"\n[analysis] Re-running denoising for subject {psd_sid} (PSD + topomap)...")
        try:
            X_by_pipeline, sfreq, ch_names, _ = _load_denoised_for_subject(
                cfg, psd_sid, pipelines_in_csv
            )
        except Exception as e:
            warnings.warn(f"[analysis] PSD/topomap skipped: {e}")
            X_by_pipeline = None

    generate_all_preprint_figures(
        csv_path=csv_path,
        out_dir=out_dir,
        X_by_pipeline=X_by_pipeline,
        sfreq=sfreq,
        ch_names=ch_names,
        subject_id=psd_sid,
    )


if __name__ == "__main__":
    main()
