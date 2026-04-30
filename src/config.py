from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class BandpassConfig:
    l_freq: float
    h_freq: float


@dataclass
class CVConfig:
    n_splits: int
    shuffle: bool
    random_state: int


@dataclass
class PermutationConfig:
    n_subject_level: int
    n_pipeline_level: int
    adaptive_step_up: bool = True
    borderline_low: float = 0.04
    borderline_high: float = 0.06
    step_up_n_subject_level: int = 10000


@dataclass
class MemoryConfig:
    float_dtype: str
    n_jobs: int
    save_models: bool = True
    # Save full denoised trial arrays per (subject, pipeline) for offline/cloud reuse (large disk).
    save_denoised_npz: bool = False
    denoised_subdir: str = "denoised"


@dataclass
class StatisticsConfig:
    """Subject vs-chance and between-pipeline inference (professor: binomial + Mann–Whitney for large N)."""

    subject_chance_method: str = "binomial"  # permutation | binomial
    pipeline_comparison_method: str = "mann_whitney"  # permutation | mann_whitney | wilcoxon


@dataclass
class BackboneConfig:
    use_csp: bool
    use_tangent_space: bool
    use_time_lda: bool = False
    use_transformer: bool = False
    # Optional feature decimation target for time-domain LDA.
    # None (or <=0) keeps native sampling rate.
    time_lda_target_sfreq: float | None = None
    transformer_target_sfreq: float | None = None
    transformer_epochs: int = 50
    transformer_batch_size: int = 16
    transformer_learning_rate: float = 1e-4
    transformer_weight_decay: float = 1e-4
    transformer_dropout: float = 0.3
    transformer_d_model: int = 256
    transformer_n_heads: int = 8
    transformer_n_layers: int = 4
    transformer_ff_dim: int = 256
    transformer_val_fraction: float = 0.125
    transformer_patience: int = 5
    transformer_device: str = "cpu"
    # Paper-exact training mode for the AASD Transformer: switches off the
    # comparison's training improvements (AdamW, cosine LR, augmentation,
    # gradient clipping) so the configuration matches the AASD paper's
    # description (Adam, no scheduler, no augmentation). Use this for the
    # cleanest "GEDAI improves the published model" head-to-head.
    transformer_paper_exact: bool = False
    # CSP filters are defined for binary classification; default to skipping CSP
    # when labels are multi-class unless explicitly enabled.
    allow_csp_multiclass: bool = False


@dataclass
class DenoisingConfig:
    use_baseline: bool
    use_icalabel: bool
    use_gedai: bool
    # Optional additional pipelines (e.g. ASR, PyLossless); default False for older configs.
    use_asr: bool = False
    use_pylossless: bool = False
    # GEDAI with MRCP-derived reference covariance (Prof. Ros / Cohen 2022).
    use_gedai_mrcp: bool = False
    # Prior used to construct GEDAI MRCP reference covariance.
    # One of: grand_avg_erp | class_contrast | trial_cov_mean
    gedai_mrcp_prior: str = "grand_avg_erp"
    # GEDAI noise_multiplier for MRCP pipeline. Prof. Ros: keep low (1.0 or 0.0) for
    # MRCP since only a few signal components carry the slow ERP; default 1.0.
    gedai_mrcp_noise_multiplier: float = 1.0
    # MRCP 0.1–1 Hz retention-guard minimum on the 9 motor channels. When retention
    # falls below this, the pipeline reverts to the paper baseline. Set to 0.0 to
    # disable the guard (useful for honestly testing noise_multiplier=0).
    gedai_mrcp_retention_min: float = 0.50
    # Optional late-window start (seconds, relative to movement onset at 0 s) used
    # when building the MRCP reference covariance. The default focuses the refcov on
    # the strongest late MRCP portion instead of the entire [-2, 0] epoch.
    gedai_mrcp_refcov_tmin_s: float | None = None
    # Optional maximum rank for MRCP refcov priors such as class_contrast. Keeping
    # only the leading movement-dominant eigenmodes can better match SENSAI's small
    # reference-PC budget.
    gedai_mrcp_refcov_rank_max: int | None = None
    # Optional motor-strip channel weighting applied to the MRCP refcov. Values >1
    # emphasize the canonical MRCP motor subset in the spatial target.
    gedai_mrcp_refcov_motor_weight: float | None = None
    # Optional movement-covariance blend for class_contrast. Small values can
    # stabilize the positive contrast subspace on low-trial folds.
    gedai_mrcp_refcov_move_mix: float | None = None
    # Optional leakage-safe fold gate: compare baseline vs GEDAI-MRCP on each
    # outer fold's training data and only keep GEDAI when it wins by at least
    # ``gedai_mrcp_gate_margin``. Useful when GEDAI helps some subjects but
    # hurts others.
    gedai_mrcp_adaptive_baseline_gate: bool = False
    gedai_mrcp_gate_margin: float = 0.0
    gedai_mrcp_gate_inner_splits: int = 2
    # When True: build C_mrcp from ALL subject trials (not just training fold).
    # This is the professor's simple version — more stable estimate with double
    # the trials. Mild spatial-filter leakage, much lower variance.
    gedai_mrcp_refcov_all_trials: bool = False
    # AASD/Transformer leakage-safe gate: per outer fold, pick the better of
    # {baseline, gedai} using inner-CV on training data only. Prevents GEDAI
    # from hurting subjects where the denoiser is detrimental.
    gedai_aasd_adaptive_gate: bool = False
    gedai_aasd_gate_margin: float = 0.0
    gedai_aasd_gate_inner_splits: int = 3
    # AASD per-fold leakage-safe reference covariance: when True, recompute
    # the CSP-style refcov for each outer CV fold using only that fold's
    # training trials. The published claim becomes airtight because no test
    # labels touch the GEDAI spatial filter direction.
    gedai_aasd_perfold_refcov: bool = False
    # Anti-Laplacian spatial filter (Reyes-Jiménez et al., Data in Brief 65, 2026 §4.5).
    # The dataset authors validated MRCP extraction using: CAR → Anti-Laplacian → 0.1–1 Hz.
    # Formula: VL(i) = V(i) + (1/N) * sum_{j ∈ neighbors(i)} V(j)
    # This averages in nearby electrodes, enhancing the spatially broad MRCP signal.
    use_anti_laplacian: bool = False
    anti_laplacian_n_neighbors: int = 4


@dataclass
class SignalIntegrityConfig:
    channels_of_interest: List[str]
    segment_duration_s: float


@dataclass
class ExperimentConfig:
    data_root: Path
    results_root: Path
    subjects: List[int]
    sampling_rate: float
    bandpass: BandpassConfig
    cv: CVConfig
    permutation: PermutationConfig
    memory: MemoryConfig
    backbones: BackboneConfig
    denoising: DenoisingConfig
    signal_integrity: SignalIntegrityConfig
    statistics: StatisticsConfig
    dataset_label: str | None = None  # e.g. "physionet_eegbci"; for cross-dataset report
    # If set, stratified subsample to at most this many trials per subject (smoke / fast runs).
    max_trials: Optional[int] = None
    # Optional cap on number of classes when max_trials_strategy=multiclass_topk.
    max_trials_top_k: Optional[int] = None
    # How to handle max_trials when labels are multi-class:
    # - "binary_for_multiclass": legacy behavior (top-2 classes -> binary)
    # - "multiclass_topk": balanced subset over the most frequent K classes
    max_trials_strategy: str = "binary_for_multiclass"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        denoising_cfg = cfg["denoising"]
        denoising = DenoisingConfig(
            use_baseline=denoising_cfg.get("use_baseline", True),
            use_icalabel=denoising_cfg.get("use_icalabel", False),
            use_gedai=denoising_cfg.get("use_gedai", False),
            use_asr=denoising_cfg.get("use_asr", False),
            use_pylossless=denoising_cfg.get("use_pylossless", False),
            use_gedai_mrcp=denoising_cfg.get("use_gedai_mrcp", False),
            gedai_mrcp_prior=str(
                denoising_cfg.get("gedai_mrcp_prior", "grand_avg_erp")
            ).lower(),
            gedai_mrcp_noise_multiplier=float(
                denoising_cfg.get("gedai_mrcp_noise_multiplier", 1.0)
            ),
            gedai_mrcp_retention_min=float(
                denoising_cfg.get("gedai_mrcp_retention_min", 0.50)
            ),
            gedai_mrcp_refcov_tmin_s=(
                None
                if denoising_cfg.get("gedai_mrcp_refcov_tmin_s") is None
                else float(denoising_cfg.get("gedai_mrcp_refcov_tmin_s"))
            ),
            gedai_mrcp_refcov_rank_max=(
                None
                if denoising_cfg.get("gedai_mrcp_refcov_rank_max") is None
                else int(denoising_cfg.get("gedai_mrcp_refcov_rank_max"))
            ),
            gedai_mrcp_refcov_motor_weight=(
                None
                if denoising_cfg.get("gedai_mrcp_refcov_motor_weight") is None
                else float(denoising_cfg.get("gedai_mrcp_refcov_motor_weight"))
            ),
            gedai_mrcp_refcov_move_mix=(
                None
                if denoising_cfg.get("gedai_mrcp_refcov_move_mix") is None
                else float(denoising_cfg.get("gedai_mrcp_refcov_move_mix"))
            ),
            gedai_mrcp_adaptive_baseline_gate=bool(
                denoising_cfg.get("gedai_mrcp_adaptive_baseline_gate", False)
            ),
            gedai_mrcp_gate_margin=float(
                denoising_cfg.get("gedai_mrcp_gate_margin", 0.0)
            ),
            gedai_mrcp_gate_inner_splits=int(
                denoising_cfg.get("gedai_mrcp_gate_inner_splits", 2)
            ),
            gedai_mrcp_refcov_all_trials=bool(
                denoising_cfg.get("gedai_mrcp_refcov_all_trials", False)
            ),
            use_anti_laplacian=bool(
                denoising_cfg.get("use_anti_laplacian", False)
            ),
            anti_laplacian_n_neighbors=int(
                denoising_cfg.get("anti_laplacian_n_neighbors", 4)
            ),
            gedai_aasd_adaptive_gate=bool(
                denoising_cfg.get("gedai_aasd_adaptive_gate", False)
            ),
            gedai_aasd_gate_margin=float(
                denoising_cfg.get("gedai_aasd_gate_margin", 0.0)
            ),
            gedai_aasd_gate_inner_splits=int(
                denoising_cfg.get("gedai_aasd_gate_inner_splits", 3)
            ),
            gedai_aasd_perfold_refcov=bool(
                denoising_cfg.get("gedai_aasd_perfold_refcov", False)
            ),
        )

        mem = cfg["memory"]
        stat_cfg = cfg.get("statistics") or {}
        statistics = StatisticsConfig(
            subject_chance_method=str(
                stat_cfg.get("subject_chance_method", "binomial")
            ).lower(),
            pipeline_comparison_method=str(
                stat_cfg.get("pipeline_comparison_method", "mann_whitney")
            ).lower(),
        )

        return cls(
            data_root=Path(cfg["data_root"]),
            results_root=Path(cfg["results_root"]),
            subjects=list(cfg["subjects"]),
            sampling_rate=float(cfg["sampling_rate"]),
            bandpass=BandpassConfig(**cfg["bandpass"]),
            cv=CVConfig(**cfg["cv"]),
            permutation=PermutationConfig(**cfg["permutation"]),
            memory=MemoryConfig(
                float_dtype=mem["float_dtype"],
                n_jobs=mem["n_jobs"],
                save_models=mem.get("save_models", True),
                save_denoised_npz=mem.get("save_denoised_npz", False),
                denoised_subdir=str(mem.get("denoised_subdir", "denoised")),
            ),
            backbones=BackboneConfig(**cfg["backbones"]),
            denoising=denoising,
            signal_integrity=SignalIntegrityConfig(**cfg["signal_integrity"]),
            statistics=statistics,
            dataset_label=cfg.get("dataset_label"),
            max_trials=(
                int(cfg["max_trials"])
                if cfg.get("max_trials") is not None
                else None
            ),
            max_trials_top_k=(
                int(cfg["max_trials_top_k"])
                if cfg.get("max_trials_top_k") is not None
                else None
            ),
            max_trials_strategy=str(cfg.get("max_trials_strategy", "binary_for_multiclass")),
        )
