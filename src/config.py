from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

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


@dataclass
class DenoisingConfig:
    use_baseline: bool
    use_icalabel: bool
    use_gedai: bool
    # Optional additional pipelines (e.g. ASR, PyLossless); default False for older configs.
    use_asr: bool = False
    use_pylossless: bool = False


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
        )
