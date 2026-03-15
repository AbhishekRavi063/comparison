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


@dataclass
class BackboneConfig:
    use_csp: bool
    use_tangent_space: bool


@dataclass
class DenoisingConfig:
    use_baseline: bool
    use_icalabel: bool
    use_gedai: bool
    # Optional additional pipelines (e.g. ASR); default False for older configs.
    use_asr: bool = False


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
    dataset_label: str | None = None  # e.g. "physionet_eegbci"; for cross-dataset report

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r") as f:
            cfg = yaml.safe_load(f)

        denoising_cfg = cfg["denoising"]
        denoising = DenoisingConfig(
            use_baseline=denoising_cfg.get("use_baseline", True),
            use_icalabel=denoising_cfg.get("use_icalabel", False),
            use_gedai=denoising_cfg.get("use_gedai", False),
            use_asr=denoising_cfg.get("use_asr", False),
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
                float_dtype=cfg["memory"]["float_dtype"],
                n_jobs=cfg["memory"]["n_jobs"],
                save_models=cfg["memory"].get("save_models", True),
            ),
            backbones=BackboneConfig(**cfg["backbones"]),
            denoising=denoising,
            signal_integrity=SignalIntegrityConfig(**cfg["signal_integrity"]),
            dataset_label=cfg.get("dataset_label"),
        )
