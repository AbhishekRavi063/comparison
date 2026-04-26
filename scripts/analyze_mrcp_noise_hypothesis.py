from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backbones.tangent_space import run_tangent_cv_preprocessed
from src.denoising.pipelines import (
    MRCP_MOTOR_CHANNELS,
    _median_bandpower_ratio,
    _resolve_eeg_emg_mrcp_raw_root,
    _run_mrcp_raw_pipeline,
)


def _late_window_mask(sfreq: float, n_times: int, start_s: float = -0.5, end_s: float = 0.0) -> np.ndarray:
    times = np.arange(n_times, dtype=np.float32) / float(sfreq) - 2.0
    return (times >= start_s) & (times < end_s)


def _cohen_d_binary(x0: np.ndarray, x1: np.ndarray) -> float:
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    n0 = len(x0)
    n1 = len(x1)
    if n0 < 2 or n1 < 2:
        return float("nan")
    v0 = np.var(x0, ddof=1)
    v1 = np.var(x1, ddof=1)
    pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def _motor_trial_window_values(X: np.ndarray, motor_idx: np.ndarray, window_mask: np.ndarray) -> np.ndarray:
    return X[:, motor_idx][:, :, window_mask].mean(axis=(1, 2))


def _available_subjects(results_root: Path) -> list[int]:
    perf = pd.read_csv(results_root / "tables" / "subject_level_performance.csv")
    return sorted(int(s) for s in perf["subject"].dropna().unique().tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether baseline benefits from nuisance variance on MRCP.")
    parser.add_argument(
        "--results-root",
        default="/Users/abhishekr/Documents/EEG/comparison/results/eeg_emg_mrcp_full_corrected",
    )
    parser.add_argument(
        "--data-root",
        default="/Users/abhishekr/Documents/EEG/comparison/data/eeg_emg_mrcp/processed",
    )
    parser.add_argument(
        "--raw-root",
        default="/Users/abhishekr/Documents/EEG/comparison/data/EEG and EMG Dataset for Analyzing Movement-Related",
    )
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--out-csv", default="/Users/abhishekr/Documents/EEG/comparison/results/eeg_emg_mrcp_noise_hypothesis.csv")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    raw_root = _resolve_eeg_emg_mrcp_raw_root(Path(args.raw_root))

    subjects = args.subjects or _available_subjects(results_root)
    rows: list[dict[str, float | int]] = []

    for sid in subjects:
        npz_path = data_root / f"subject_{sid}.npz"
        if not npz_path.exists():
            print(f"skip subject {sid}: missing {npz_path}")
            continue

        with np.load(npz_path, allow_pickle=True) as z:
            y = z["y"].astype(int)
            sfreq = float(np.asarray(z["sfreq"]).item())
            ch_names = [str(c) for c in z["ch_names"].tolist()]

        X_gedai, X_base = _run_mrcp_raw_pipeline(
            subject_id=sid,
            raw_root=raw_root,
            l_freq=0.1,
            h_freq=1.0,
            sfreq=sfreq,
            gedai_n_jobs=1,
            run_gedai=True,
        )
        # Ensure labels align with actual extracted trial count.
        n = min(len(y), len(X_base), len(X_gedai))
        y = y[:n]
        X_base = X_base[:n]
        X_gedai = X_gedai[:n]
        X_removed = X_base - X_gedai

        motor_idx = np.array([i for i, c in enumerate(ch_names) if c.upper() in {m.upper() for m in MRCP_MOTOR_CHANNELS}], dtype=int)
        Xb = X_base[:, motor_idx, :]
        Xg = X_gedai[:, motor_idx, :]
        Xr = X_removed[:, motor_idx, :]

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        cv_splits = list(cv.split(Xb, y))
        base_res = run_tangent_cv_preprocessed(Xb, y, cv_splits)
        gedai_res = run_tangent_cv_preprocessed(Xg, y, cv_splits)
        removed_res = run_tangent_cv_preprocessed(Xr, y, cv_splits)

        window_mask = _late_window_mask(sfreq, Xb.shape[-1], start_s=-0.5, end_s=0.0)
        base_vals = _motor_trial_window_values(Xb, np.arange(Xb.shape[1]), window_mask)
        gedai_vals = _motor_trial_window_values(Xg, np.arange(Xg.shape[1]), window_mask)
        rem_vals = _motor_trial_window_values(Xr, np.arange(Xr.shape[1]), window_mask)
        y0 = y == 0
        y1 = y == 1
        base_d = _cohen_d_binary(base_vals[y0], base_vals[y1])
        gedai_d = _cohen_d_binary(gedai_vals[y0], gedai_vals[y1])
        removed_d = _cohen_d_binary(rem_vals[y0], rem_vals[y1])

        base_diff = Xb[y1].mean(axis=0) - Xb[y0].mean(axis=0)
        gedai_diff = Xg[y1].mean(axis=0) - Xg[y0].mean(axis=0)
        removed_diff = Xr[y1].mean(axis=0) - Xr[y0].mean(axis=0)
        base_peak = float(np.min(np.median(base_diff, axis=0)))
        gedai_peak = float(np.min(np.median(gedai_diff, axis=0)))
        removed_peak = float(np.min(np.median(removed_diff, axis=0)))

        retention_motor = float(_median_bandpower_ratio(Xg, Xb, sfreq, 0.1, 1.0))
        removed_ratio = float(_median_bandpower_ratio(Xr, Xb, sfreq, 0.1, 1.0))

        raw_std = float(np.std(X_base[:, :, :], dtype=np.float64))
        motor_std = float(np.std(Xb, dtype=np.float64))

        rows.append(
            {
                "subject": sid,
                "baseline_acc": float(np.mean(base_res.fold_accuracies)),
                "gedai_acc": float(np.mean(gedai_res.fold_accuracies)),
                "removed_acc": float(np.mean(removed_res.fold_accuracies)),
                "delta_acc": float(np.mean(gedai_res.fold_accuracies) - np.mean(base_res.fold_accuracies)),
                "retention_motor_0.1_1.0": retention_motor,
                "removed_ratio_0.1_1.0": removed_ratio,
                "baseline_d_late": base_d,
                "gedai_d_late": gedai_d,
                "removed_d_late": removed_d,
                "baseline_peak_diff": base_peak,
                "gedai_peak_diff": gedai_peak,
                "removed_peak_diff": removed_peak,
                "raw_std": raw_std,
                "motor_std": motor_std,
            }
        )
        print(
            f"subject {sid:02d} | base={rows[-1]['baseline_acc']:.3f} | "
            f"gedai={rows[-1]['gedai_acc']:.3f} | removed={rows[-1]['removed_acc']:.3f} | "
            f"ret={retention_motor:.3f} | d(base/gedai/rem)={base_d:.2f}/{gedai_d:.2f}/{removed_d:.2f}"
        )

    df = pd.DataFrame(rows).sort_values("subject")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}\n")
    print(df.describe().to_string())

    if len(df) >= 3:
        print("\nCorrelations")
        for x, ycol in [
            ("removed_acc", "delta_acc"),
            ("removed_d_late", "delta_acc"),
            ("raw_std", "delta_acc"),
            ("motor_std", "delta_acc"),
            ("retention_motor_0.1_1.0", "delta_acc"),
            ("gedai_d_late", "delta_acc"),
        ]:
            r, p = pearsonr(df[x], df[ycol])
            print(f"{ycol} vs {x}: r={r:.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
