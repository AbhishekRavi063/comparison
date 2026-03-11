from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ExperimentConfig
from src.evaluation.experiment import run_experiment


def _make_synthetic_subject(
    out_path: Path,
    n_trials: int = 40,
    n_channels: int = 8,
    n_times: int = 250,
    sfreq: float = 250.0,
) -> None:
    rng = np.random.RandomState(0)
    t = np.arange(n_times) / sfreq

    X = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    y = np.zeros((n_trials,), dtype=int)

    for i in range(n_trials):
        label = rng.randint(0, 2)
        y[i] = label
        base = rng.randn(n_channels, n_times) * 0.2
        if label == 0:
            signal = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz
        else:
            signal = 0.5 * np.sin(2 * np.pi * 20 * t)  # 20 Hz
        X[i] = base + signal

    ch_names = [f"Ch{idx}" for idx in range(n_channels)]
    np.savez(
        out_path,
        X=X,
        y=y,
        sfreq=sfreq,
        ch_names=np.array(ch_names, dtype=object),
    )


def test_synthetic_baseline_only(tmp_path: Path) -> None:
    """Minimal smoke test on synthetic data using baseline pipelines only."""
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir(parents=True, exist_ok=True)

    _make_synthetic_subject(data_root / "subject_1.npz")

    cfg = ExperimentConfig.from_yaml("config/config.yml")
    cfg.data_root = data_root
    cfg.results_root = results_root
    cfg.subjects = [1]
    # Avoid calling external mne/gedai-based denoisers in the test environment
    cfg.denoising.use_icalabel = False
    cfg.denoising.use_gedai = False

    run_experiment(cfg)

    assert (results_root / "tables" / "subject_level_performance.csv").exists()
    assert (results_root / "stats" / "pipeline_comparisons.csv").exists()

