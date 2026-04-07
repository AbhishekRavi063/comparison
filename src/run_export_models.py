from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from .config import ExperimentConfig
from .denoising.pipelines import preprocess_subject_data
from .evaluation.experiment import _apply_max_trials_smoke
from .io.dataset import NpzMotorImageryDataset
from .backbones.csp import fit_csp_model_preprocessed
from .backbones.tangent_space import fit_tangent_model_preprocessed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained full-data models only (no CV/permutations)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_alljoined_smoke_1sub.yml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    log = logging.getLogger("run_export_models")
    cfg = ExperimentConfig.from_yaml(args.config)
    dataset = NpzMotorImageryDataset(
        data_root=cfg.data_root,
        subjects=cfg.subjects,
        float_dtype=cfg.memory.float_dtype,
    )
    models_dir = Path(cfg.results_root) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    pipelines = []
    if cfg.denoising.use_baseline:
        pipelines.append("baseline")
    if cfg.denoising.use_icalabel:
        pipelines.append("icalabel")
    if cfg.denoising.use_gedai:
        pipelines.append("gedai")
    if getattr(cfg.denoising, "use_pylossless", False):
        pipelines.append("pylossless")
    if getattr(cfg.denoising, "use_asr", False):
        pipelines.append("asr")

    backbones = []
    if cfg.backbones.use_csp:
        backbones.append("csp")
    if cfg.backbones.use_tangent_space:
        backbones.append("tangent")

    for sid, subj_data in dataset.iter_subjects():
        X, y = subj_data.X, subj_data.y
        X, y = _apply_max_trials_smoke(cfg, X, y, sid, log)
        for pipeline in pipelines:
            X_proc = preprocess_subject_data(
                X=X,
                sfreq=subj_data.sfreq,
                ch_names=subj_data.ch_names,
                l_freq=cfg.bandpass.l_freq,
                h_freq=cfg.bandpass.h_freq,
                denoising=pipeline,
                subject_id=sid if pipeline in ("gedai", "pylossless") else None,
                dataset_name=cfg.dataset_label or "",
                gedai_n_jobs=cfg.memory.n_jobs,
                data_root=cfg.data_root,
            )
            for backbone in backbones:
                if backbone == "csp":
                    model = fit_csp_model_preprocessed(
                        X_proc=X_proc,
                        y=y,
                        sfreq=subj_data.sfreq,
                        ch_names=subj_data.ch_names,
                        l_freq=cfg.bandpass.l_freq,
                        h_freq=cfg.bandpass.h_freq,
                        denoising=pipeline,
                    )
                else:
                    model = fit_tangent_model_preprocessed(
                        X_proc=X_proc,
                        y=y,
                        sfreq=subj_data.sfreq,
                        ch_names=subj_data.ch_names,
                        l_freq=cfg.bandpass.l_freq,
                        h_freq=cfg.bandpass.h_freq,
                        denoising=pipeline,
                    )
                out_path = models_dir / f"subject_{sid}_{backbone}_{pipeline}.joblib"
                joblib.dump(model, out_path)


if __name__ == "__main__":
    main()
