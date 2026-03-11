"""Create data/synth_a and data/synth_b with synthetic subject_1.npz for a full-test dry run (no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def make_synthetic_subject(out_path: Path, subject_id: int, seed: int = 0) -> None:
    rng = np.random.RandomState(seed + subject_id)
    n_trials, n_channels, n_times = 40, 8, 250
    sfreq = 250.0
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


def main() -> None:
    for name in ["synth_a", "synth_b"]:
        data_root = PROJECT_ROOT / "data" / name
        data_root.mkdir(parents=True, exist_ok=True)
        make_synthetic_subject(data_root / "subject_1.npz", 1, seed=hash(name) % 10000)
        print(f"Created {data_root / 'subject_1.npz'}")


if __name__ == "__main__":
    main()
