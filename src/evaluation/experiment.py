from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def _log_memory_if_debug(log) -> None:
    """If EEG_MEMORY_DEBUG=1, log current process RSS (MB). Requires psutil."""
    if os.environ.get("EEG_MEMORY_DEBUG", "").strip() != "1":
        return
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 ** 2)
        log.info(f"RAM (MB): {rss_mb:.1f}")
    except Exception:
        pass


from ..config import ExperimentConfig
from ..io.dataset import NpzMotorImageryDataset
from ..backbones.csp import run_csp_cv_preprocessed, fit_csp_model_preprocessed
from ..backbones.tangent_space import (
    build_tangent_features_for_splits,
    run_tangent_cv_precomputed_features,
    fit_tangent_model_preprocessed,
)
from ..denoising.pipelines import preprocess_subject_data
from ..data.dataset_noise_inspection import (
    plot_denoising_comparison_overlay,
    plot_denoising_psd_comparison,
)
from .metrics import (
    SubjectPerformance,
    binomial_vs_chance_p_value,
    empirical_chance_p_value,
    cohen_d_pooled,
    mann_whitney_pipeline_p_value,
    paired_permutation_p_value,
    wilcoxon_paired_pipeline_p_value,
    compute_band_power,
)


def _is_borderline_p_value(p_value: float, cfg: ExperimentConfig) -> bool:
    low = float(cfg.permutation.borderline_low)
    high = float(cfg.permutation.borderline_high)
    return low <= p_value <= high


@dataclass
class ExperimentResult:
    subject_performances: List[SubjectPerformance]
    pipeline_comparisons: pd.DataFrame


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run the complete experiment over all subjects and pipelines.

    Per subject, pipelines are run in order: baseline, ICALabel, then GEDAI.
    Before GEDAI we call gc.collect() so ICA/Raw temporaries from ICALabel are
    freed, reducing peak RAM when GEDAI runs.
    """
    rng = np.random.RandomState(cfg.cv.random_state)
    np.random.set_state(rng.get_state())

    dataset = NpzMotorImageryDataset(
        data_root=cfg.data_root,
        subjects=cfg.subjects,
        float_dtype=cfg.memory.float_dtype,
    )

    results_root = cfg.results_root
    tables_dir = results_root / "tables"
    stats_dir = results_root / "stats"
    models_dir = results_root / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    if cfg.memory.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    if cfg.memory.save_denoised_npz:
        (results_root / cfg.memory.denoised_subdir).mkdir(parents=True, exist_ok=True)

    subject_performances: List[SubjectPerformance] = []

    pipelines = []
    if cfg.denoising.use_baseline:
        pipelines.append("baseline")
    # ASR (Artifact Subspace Reconstruction) – optional additional denoising pipeline.
    if getattr(cfg.denoising, "use_asr", False):
        pipelines.append("asr")
    if cfg.denoising.use_icalabel:
        pipelines.append("icalabel")
    if cfg.denoising.use_gedai:
        pipelines.append("gedai")
    if getattr(cfg.denoising, "use_pylossless", False):
        pipelines.append("pylossless")

    backbones = []
    if cfg.backbones.use_csp:
        backbones.append("csp")
    if cfg.backbones.use_tangent_space:
        backbones.append("tangent")

    log = logging.getLogger("run_full_test")
    n_subj = len(cfg.subjects)
    log.info(
        f"Experiment: {n_subj} subjects, pipelines={pipelines}, backbones={backbones}, "
        f"subject_chance={cfg.statistics.subject_chance_method}, "
        f"pipeline_cmp={cfg.statistics.pipeline_comparison_method}, "
        f"null_perm={cfg.permutation.n_subject_level}, pipeline_perm={cfg.permutation.n_pipeline_level}"
    )

    # Subject-level loop (sequential to respect memory constraints)
    for sid, subj_data in tqdm(
        dataset.iter_subjects(), desc="Subjects", total=len(cfg.subjects)
    ):
        log.info(f"Subject {sid}/{n_subj} started (n_trials={subj_data.X.shape[0]})")
        _log_memory_if_debug(log)
        X, y = subj_data.X, subj_data.y
        cv = StratifiedKFold(
            n_splits=cfg.cv.n_splits,
            shuffle=cfg.cv.shuffle,
            random_state=cfg.cv.random_state,
        )
        cv_splits = list(cv.split(X, y))

        # Pipeline order: baseline and ICALabel first, then GEDAI.
        for pipeline in pipelines:
            if pipeline in ("gedai", "pylossless") and (
                "baseline" in pipelines or "icalabel" in pipelines
            ):
                gc.collect()
                _log_memory_if_debug(log)
                log.info(
                    f"  (lighter pipelines done for this subject; running {pipeline} next)"
                )

            X_proc = preprocess_subject_data(
                X=X,
                sfreq=subj_data.sfreq,
                ch_names=subj_data.ch_names,
                l_freq=cfg.bandpass.l_freq,
                h_freq=cfg.bandpass.h_freq,
                denoising=pipeline,
                # GEDAI (PhysioNet): continuous EDF; PyLossless (Alljoined): optional manifest.
                subject_id=sid if pipeline in ("gedai", "pylossless") else None,
                dataset_name=cfg.dataset_label or "",
                gedai_n_jobs=cfg.memory.n_jobs,
                data_root=cfg.data_root,
            )

            if cfg.memory.save_denoised_npz:
                ddir = results_root / cfg.memory.denoised_subdir
                ch_arr = np.asarray(list(subj_data.ch_names), dtype=object)
                den_path = ddir / f"subject_{sid}_{pipeline}.npz"
                np.savez_compressed(
                    den_path,
                    X=np.asarray(X_proc, dtype=np.float32),
                    y=y,
                    sfreq=np.float32(subj_data.sfreq),
                    ch_names=ch_arr,
                )
                log.info(f"  Saved denoised data {den_path}")

            # --- Signal Preservation Calculation ---
            # We use the config's channels_of_interest (first one as primary)
            # Baseline power is calculated from X_bp (the first pipeline which is usually 'baseline')
            coi = cfg.signal_integrity.channels_of_interest[0]
            try:
                ch_idx = [c.upper() for c in subj_data.ch_names].index(coi.upper())
            except ValueError:
                ch_idx = 0
            
            alpha_band = (8.0, 12.0)
            beta_band = (13.0, 30.0)
            
            # Record power for this pipeline
            curr_alpha = compute_band_power(X_proc, subj_data.sfreq, *alpha_band, ch_idx=ch_idx)
            curr_beta = compute_band_power(X_proc, subj_data.sfreq, *beta_band, ch_idx=ch_idx)
            
            if pipeline == "baseline":
                ref_alpha = curr_alpha if curr_alpha > 0 else 1.0
                ref_beta = curr_beta if curr_beta > 0 else 1.0
                alpha_ratio = 1.0
                beta_ratio = 1.0
            else:
                # ref_alpha/ref_beta must have been set by the 'baseline' pass
                alpha_ratio = curr_alpha / ref_alpha
                beta_ratio = curr_beta / ref_beta

            # --- Automated Plots (Subject 1 only or if configured) ---
            if sid == cfg.subjects[0]:
                plots_dir = results_root / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # PSD Comparison
                psd_out = plots_dir / f"sub{sid}_{pipeline}_psd_comparison.png"
                # We need X_bp as the reference "In"
                # But X_proc is the current pipeline's "Out"
                # To make this clean, we'd need to keep X_baseline around or re-run bandpass.
                # However, X_proc for 'baseline' IS the bandpass.
                if pipeline == "baseline":
                    # Store baseline for comparison with future pipelines
                    X_baseline_ref = X_proc.copy()
                else:
                    # Plot current vs baseline
                    plot_denoising_psd_comparison(
                        X_in=X_baseline_ref,
                        X_out=X_proc,
                        sfreq=subj_data.sfreq,
                        out_path=psd_out,
                        subject_id=sid,
                        pipeline_name=pipeline
                    )
                    
                    # Time Domain Overlay
                    overlay_out = plots_dir / f"sub{sid}_{pipeline}_overlay_comparison.png"
                    plot_denoising_comparison_overlay(
                        X_in=X_baseline_ref,
                        X_out=X_proc,
                        sfreq=subj_data.sfreq,
                        ch_names=subj_data.ch_names,
                        out_path=overlay_out,
                        subject_id=sid,
                        pipeline_name=pipeline,
                        n_channels=5,
                        trial_idx=0
                    )

            tangent_fold_features = None
            use_chance_binom = cfg.statistics.subject_chance_method == "binomial"
            n_classes = int(len(np.unique(y)))

            for backbone in backbones:
                if backbone == "csp":
                    res = run_csp_cv_preprocessed(X_proc=X_proc, y=y, cv_splits=cv_splits)
                else:
                    if tangent_fold_features is None:
                        tangent_fold_features = build_tangent_features_for_splits(
                            X_proc=X_proc,
                            cv_splits=cv_splits,
                        )
                    res = run_tangent_cv_precomputed_features(
                        y=y,
                        cv_splits=cv_splits,
                        fold_features=tangent_fold_features,
                    )

                fold_acc = list(res.fold_accuracies)
                mean_acc = float(np.mean(fold_acc))
                std_acc = float(np.std(fold_acc, ddof=1))
                pooled_c = int(res.pooled_test_correct)
                pooled_t = int(res.pooled_test_total)

                if use_chance_binom:
                    p_emp = binomial_vs_chance_p_value(pooled_c, pooled_t, n_classes)
                    p_method = "binomial"
                else:
                    p_method = "permutation"
                    null_acc: List[float] = []
                    base_n = int(cfg.permutation.n_subject_level)
                    for _ in tqdm(
                        range(base_n),
                        desc=f"  null s{sid} {backbone} {pipeline}",
                        leave=False,
                        total=base_n,
                    ):
                        y_shuffled = np.random.permutation(y)
                        if backbone == "csp":
                            nres = run_csp_cv_preprocessed(
                                X_proc=X_proc,
                                y=y_shuffled,
                                cv_splits=cv_splits,
                            )
                        else:
                            nres = run_tangent_cv_precomputed_features(
                                y=y_shuffled,
                                cv_splits=cv_splits,
                                fold_features=tangent_fold_features,
                            )
                        null_acc.append(float(np.mean(nres.fold_accuracies)))
                        del nres

                    p_emp = empirical_chance_p_value(mean_acc, null_acc)

                    if (
                        cfg.permutation.adaptive_step_up
                        and _is_borderline_p_value(p_emp, cfg)
                        and int(cfg.permutation.step_up_n_subject_level) > base_n
                    ):
                        extra_n = int(cfg.permutation.step_up_n_subject_level) - base_n
                        log.info(
                            f"  Step-up permutations for s{sid} {backbone} {pipeline}: "
                            f"{base_n} -> {cfg.permutation.step_up_n_subject_level} (borderline p={p_emp:.4f})"
                        )
                        for _ in tqdm(
                            range(extra_n),
                            desc=f"  step-up s{sid} {backbone} {pipeline}",
                            leave=False,
                            total=extra_n,
                        ):
                            y_shuffled = np.random.permutation(y)
                            if backbone == "csp":
                                nres = run_csp_cv_preprocessed(
                                    X_proc=X_proc,
                                    y=y_shuffled,
                                    cv_splits=cv_splits,
                                )
                            else:
                                nres = run_tangent_cv_precomputed_features(
                                    y=y_shuffled,
                                    cv_splits=cv_splits,
                                    fold_features=tangent_fold_features,
                                )
                            null_acc.append(float(np.mean(nres.fold_accuracies)))
                            del nres
                        p_emp = empirical_chance_p_value(mean_acc, null_acc)
                    del null_acc

                del res

                subject_performances.append(
                    SubjectPerformance(
                        subject_id=sid,
                        backbone=backbone,
                        pipeline=pipeline,
                        fold_accuracies=fold_acc,
                        mean_accuracy=mean_acc,
                        std_accuracy=std_acc,
                        p_empirical=p_emp,
                        alpha_ratio=alpha_ratio,
                        beta_ratio=beta_ratio,
                        pooled_test_correct=pooled_c if use_chance_binom else None,
                        pooled_test_total=pooled_t if use_chance_binom else None,
                        p_vs_chance_method=p_method,
                    )
                )
                log.info(
                    f"  Subject {sid} | {backbone} | {pipeline} | acc={mean_acc:.3f} | p={p_emp:.4f} ({p_method})"
                )

                # Save trained model after each run (full-data fit) for reuse / deployment
                if cfg.memory.save_models:
                    model = None
                    try:
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
                        path = models_dir / f"subject_{sid}_{backbone}_{pipeline}.joblib"
                        joblib.dump(model, path)
                    except Exception as e:
                        import warnings

                        warnings.warn(
                            f"Model save failed for subject {sid} {backbone} {pipeline}: {e}"
                        )
                    finally:
                        del model

            del tangent_fold_features
            del X_proc
            gc.collect()

        # Explicit cleanup: no raw data or large objects persist beyond this subject
        del X, y, cv_splits, subj_data
        gc.collect()
        _log_memory_if_debug(log)
        log.info(f"Subject {sid}/{n_subj} done.")

    # Save subject-level table
    rows = []
    for sp in subject_performances:
        for fold_idx, acc in enumerate(sp.fold_accuracies):
            rows.append(
                {
                    "subject": sp.subject_id,
                    "backbone": sp.backbone,
                    "pipeline": sp.pipeline,
                    "fold": fold_idx,
                    "accuracy": acc,
                    "mean_accuracy": sp.mean_accuracy,
                    "std_accuracy": sp.std_accuracy,
                    "p_empirical": sp.p_empirical,
                    "p_vs_chance_method": sp.p_vs_chance_method,
                    "pooled_test_correct": sp.pooled_test_correct,
                    "pooled_test_total": sp.pooled_test_total,
                    "alpha_ratio": sp.alpha_ratio,
                    "beta_ratio": sp.beta_ratio,
                }
            )
    df_subject = pd.DataFrame(rows)
    df_subject.to_csv(tables_dir / "subject_level_performance.csv", index=False)
    log.info(
        "Subject-level table written. Computing between-pipeline tests "
        f"({cfg.statistics.pipeline_comparison_method})..."
    )

    # Between-pipeline comparisons (within backbone): GEDAI vs Bandpass, ICALabel vs Bandpass, GEDAI vs ICALabel
    comparison_rows: List[Dict] = []
    for backbone in backbones:
        for (p1, p2) in [
            ("gedai", "baseline"),
            ("icalabel", "baseline"),
            ("gedai", "icalabel"),
            ("pylossless", "baseline"),
            ("gedai", "pylossless"),
            ("icalabel", "pylossless"),
        ]:
            if p1 not in pipelines or p2 not in pipelines:
                continue

            scores1: List[float] = []
            scores2: List[float] = []
            # Aggregate over subjects using mean CV per subject
            for sp in subject_performances:
                if sp.backbone != backbone:
                    continue
                if sp.pipeline == p1:
                    scores1.append(sp.mean_accuracy)
                elif sp.pipeline == p2:
                    scores2.append(sp.mean_accuracy)

            if not scores1 or not scores2:
                continue

            pcm = cfg.statistics.pipeline_comparison_method
            if pcm == "mann_whitney":
                p_val = mann_whitney_pipeline_p_value(scores1, scores2)
            elif pcm == "wilcoxon":
                p_val = wilcoxon_paired_pipeline_p_value(scores1, scores2)
            else:
                p_val = paired_permutation_p_value(
                    scores1, scores2, n_resamples=cfg.permutation.n_pipeline_level
                )
            d_eff = cohen_d_pooled(scores1, scores2)

            comparison_rows.append(
                dict(
                    backbone=backbone,
                    comparison=f"{p1} - {p2}",
                    p_value=p_val,
                    cohen_d=d_eff,
                    mean_diff=float(np.mean(scores1) - np.mean(scores2)),
                    comparison_method=pcm,
                )
            )

    col_names = [
        "backbone",
        "comparison",
        "p_value",
        "cohen_d",
        "mean_diff",
        "comparison_method",
    ]
    df_comp = pd.DataFrame(comparison_rows)
    if df_comp.empty:
        df_comp = pd.DataFrame(columns=col_names)
    df_comp.to_csv(stats_dir / "pipeline_comparisons.csv", index=False)
    log.info("Pipeline comparisons and stats written.")

    return ExperimentResult(
        subject_performances=subject_performances,
        pipeline_comparisons=df_comp,
    )
