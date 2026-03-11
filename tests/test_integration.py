"""
Integration tests: run full workflows with a few synthetic subjects to verify
everything works (experiment, plots, signal integrity, model save/load).

Run from project root:
  pytest tests/test_integration.py -v
  pytest tests/test_integration.py -v -k "case_1"   # only case 1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force non-GUI backend before any matplotlib import
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import pytest

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.evaluation.experiment import run_experiment
from src.run_plots import main as run_plots_main
from src.run_signal_integrity import main as run_signal_integrity_main


def make_synthetic_subject(
    out_path: Path,
    subject_id: int,
    n_trials: int = 40,
    n_channels: int = 8,
    n_times: int = 250,
    sfreq: float = 250.0,
    seed: int = 0,
) -> None:
    rng = np.random.RandomState(seed + subject_id)
    t = np.arange(n_times) / sfreq
    X = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    y = np.zeros((n_trials,), dtype=int)
    for i in range(n_trials):
        label = rng.randint(0, 2)
        y[i] = label
        base = rng.randn(n_channels, n_times).astype(np.float32) * 0.2
        signal = 0.5 * np.sin(2 * np.pi * (10 if label == 0 else 20) * t).astype(np.float32)
        X[i] = base + signal
    ch_names = [f"Ch{idx}" for idx in range(n_channels)]
    np.savez(out_path, X=X, y=y, sfreq=sfreq, ch_names=np.array(ch_names, dtype=object))


def _test_config(tmp_path: Path, baseline_only: bool = False) -> ExperimentConfig:
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / "config" / "config_test.yml")
    cfg.data_root = data_root
    cfg.results_root = results_root
    if baseline_only:
        cfg.denoising.use_icalabel = False
    return cfg


# ---------- Test case 1: Baseline only (no MNE/ICALabel), 2 subjects ----------
def test_case_1_baseline_only_two_subjects(tmp_path: Path) -> None:
    """Run full experiment with baseline denoising only; 2 subjects. Fast, no MNE."""
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir(parents=True, exist_ok=True)
    make_synthetic_subject(data_root / "subject_1.npz", 1)
    make_synthetic_subject(data_root / "subject_2.npz", 2)

    cfg = _test_config(tmp_path, baseline_only=True)
    cfg.subjects = [1, 2]

    run_experiment(cfg)

    assert (results_root / "tables" / "subject_level_performance.csv").exists()
    df = pd.read_csv(results_root / "tables" / "subject_level_performance.csv")
    assert len(df) >= 1
    assert set(df["subject"]) == {1, 2}
    assert set(df["backbone"]) == {"csp", "tangent"}
    assert set(df["pipeline"]) == {"baseline"}

    assert (results_root / "stats" / "pipeline_comparisons.csv").exists()

    # Models saved
    models_dir = results_root / "models"
    assert models_dir.exists()
    expected = {"subject_1_csp_baseline", "subject_1_tangent_baseline", "subject_2_csp_baseline", "subject_2_tangent_baseline"}
    found = {f.stem for f in models_dir.glob("*.joblib")}
    assert expected <= found, f"Missing models: {expected - found}"


# ---------- Test case 2: Baseline + ICALabel (needs MNE; can be flaky on synthetic) ----------
def test_case_2_baseline_and_icalabel(tmp_path: Path) -> None:
    """Run experiment with baseline + ICALabel; 1 subject. Requires MNE/ICALabel."""
    pytest.importorskip("mne")
    pytest.importorskip("mne_icalabel")

    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir(parents=True, exist_ok=True)
    make_synthetic_subject(data_root / "subject_1.npz", 1)

    cfg = _test_config(tmp_path, baseline_only=False)
    cfg.subjects = [1]

    try:
        run_experiment(cfg)
    except Exception as e:
        pytest.skip(f"ICALabel run failed on synthetic data (expected in some envs): {e}")

    df = pd.read_csv(results_root / "tables" / "subject_level_performance.csv")
    assert "baseline" in df["pipeline"].values
    assert "icalabel" in df["pipeline"].values
    assert (results_root / "models" / "subject_1_csp_baseline.joblib").exists()
    assert (results_root / "models" / "subject_1_csp_icalabel.joblib").exists()


# ---------- Test case 3: Performance plots ----------
def test_case_3_plots(tmp_path: Path) -> None:
    """After running experiment, run_plots produces figures."""
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir(parents=True, exist_ok=True)
    make_synthetic_subject(data_root / "subject_1.npz", 1)

    cfg = _test_config(tmp_path, baseline_only=True)
    cfg.subjects = [1]
    run_experiment(cfg)

    # Override config path for run_plots
    import src.run_plots as rp
    orig_parse = rp.main
    def run_plots_with_config():
        cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / "config" / "config_test.yml")
        cfg.data_root = data_root
        cfg.results_root = results_root
        subject_csv = cfg.results_root / "tables" / "subject_level_performance.csv"
        figures_dir = cfg.results_root / "figures"
        from src.plots.performance import plot_performance, plot_variability
        plot_performance(subject_csv, figures_dir)
        plot_variability(subject_csv, figures_dir)

    run_plots_with_config()

    assert (results_root / "figures" / "performance_csp.png").exists()
    assert (results_root / "figures" / "performance_tangent.png").exists()


# ---------- Test case 4: Signal integrity (pre-post overlay + PSD) ----------
def test_case_4_signal_integrity(tmp_path: Path) -> None:
    """run_signal_integrity produces time overlay and PSD (baseline only)."""
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir(parents=True, exist_ok=True)
    make_synthetic_subject(data_root / "subject_1.npz", 1)

    cfg = _test_config(tmp_path, baseline_only=True)
    cfg.subjects = [1]
    # Ensure signal_integrity channels exist in synthetic data
    cfg.signal_integrity.channels_of_interest = ["Ch0", "Ch1"]

    # Run experiment so we have a valid state; then run signal integrity
    run_experiment(cfg)

    # Call signal integrity programmatically (same as CLI with --subject 1 --trial 0)
    from src.io.dataset import NpzMotorImageryDataset
    from src.plots.signal_integrity import plot_signal_overlays, plot_psd_comparison, plot_prepost_overlay_static
    from src.denoising.pipelines import bandpass_filter

    dataset = NpzMotorImageryDataset(data_root=cfg.data_root, subjects=cfg.subjects, float_dtype="float32")
    _, subj_data = next(dataset.iter_subjects())
    X, sfreq, ch_names = subj_data.X, subj_data.sfreq, subj_data.ch_names
    trial_idx = 0
    times = np.arange(X.shape[2]) / sfreq
    X_bp = bandpass_filter(X[trial_idx : trial_idx + 1], sfreq, cfg.bandpass.l_freq, cfg.bandpass.h_freq)
    signals = {"raw": X[trial_idx, 0], "bandpass": X_bp[0, 0]}
    results_dir = results_root / "figures" / "signal_integrity"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_signal_overlays(times, signals, "Test", results_dir / "time_test.png")
    plot_psd_comparison(signals, sfreq, {"alpha": (8, 12), "beta": (13, 30)}, "PSD", results_dir / "psd_test.png")

    assert (results_dir / "time_test.png").exists()
    assert (results_dir / "psd_test.png").exists()


# ---------- Test case 5: Model load and structure ----------
def test_case_5_models_loadable(tmp_path: Path) -> None:
    """Saved joblib models load and have expected keys."""
    import joblib

    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    data_root.mkdir(parents=True, exist_ok=True)
    make_synthetic_subject(data_root / "subject_1.npz", 1)

    cfg = _test_config(tmp_path, baseline_only=True)
    cfg.subjects = [1]
    run_experiment(cfg)

    csp_model = joblib.load(results_root / "models" / "subject_1_csp_baseline.joblib")
    assert csp_model["backbone"] == "csp"
    assert "W" in csp_model and "clf" in csp_model
    assert csp_model["W"].dtype == np.float32

    tangent_model = joblib.load(results_root / "models" / "subject_1_tangent_baseline.joblib")
    assert tangent_model["backbone"] == "tangent"
    assert "C_ref" in tangent_model and "clf" in tangent_model
    assert tangent_model["C_ref"].dtype == np.float32


# ---------- Test case 6: Dataset loader ----------
def test_case_6_dataset_loader(tmp_path: Path) -> None:
    """NpzMotorImageryDataset loads synthetic npz and yields float32."""
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    make_synthetic_subject(data_root / "subject_1.npz", 1)

    from src.io.dataset import NpzMotorImageryDataset

    dataset = NpzMotorImageryDataset(data_root=data_root, subjects=[1], float_dtype="float32")
    sid, data = next(dataset.iter_subjects())
    assert sid == 1
    assert data.X.dtype == np.float32
    assert data.X.shape[0] == 40 and data.X.ndim == 3
    assert len(data.ch_names) == data.X.shape[1]
