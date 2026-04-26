#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.denoising.pipelines import apply_baseline_eeg_emg_mrcp_via_raw_files


def _load_labels(npz_root: Path, subject_id: int) -> np.ndarray:
    npz_path = npz_root / f"subject_{int(subject_id)}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ labels file: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        y = np.asarray(data["y"]).ravel()
    return y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot subject-wise grand-average MRCP on Cz over [-2, 0] s."
    )
    parser.add_argument(
        "--subject",
        type=int,
        required=True,
        help="Subject ID (e.g., 1..39).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/EEG and EMG Dataset for Analyzing Movement-Related"),
        help="Dataset root containing SUBJECT<NN> or SUBJECTS/SUBJECT<NN> folders.",
    )
    parser.add_argument(
        "--npz-root",
        type=Path,
        default=Path("data/eeg_emg_mrcp/processed"),
        help="Root containing subject_<id>.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/diagnostics"),
        help="Output directory for PNG plots.",
    )
    args = parser.parse_args()

    y = _load_labels(args.npz_root, args.subject)
    X_ref = apply_baseline_eeg_emg_mrcp_via_raw_files(
        subject_id=args.subject,
        raw_root=args.raw_root,
        l_freq=0.1,
        h_freq=1.0,
        sfreq=128.0,
    )
    if X_ref.shape[0] != y.shape[0]:
        raise ValueError(
            f"Trial mismatch for subject {args.subject}: X_ref={X_ref.shape[0]} vs y={y.shape[0]}"
        )

    ch_names = [
        "FC3", "FC1", "FCZ", "C3", "C1", "CZ", "CP3", "CP1", "CPZ",
        "FC4", "FC2", "C4", "C2", "CP4", "CP2", "F3", "F1", "FZ",
        "F2", "F4", "P3", "P1", "PZ", "P2", "P4", "O1", "OZ", "O2",
        "T7", "T8", "P7", "P8",
    ]
    cz_idx = [c.upper() for c in ch_names].index("CZ")
    move = X_ref[y == 1, cz_idx, :]
    rest = X_ref[y == 0, cz_idx, :]
    if move.shape[0] == 0:
        raise ValueError(f"No movement trials found for subject {args.subject}.")

    t = np.linspace(-2.0, 0.0, X_ref.shape[-1], endpoint=False)
    move_ga = np.mean(move, axis=0)
    rest_ga = np.mean(rest, axis=0) if rest.shape[0] else np.zeros_like(move_ga)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"mrcp_grand_average_subject_{args.subject}.png"

    plt.figure(figsize=(9, 4))
    plt.plot(t, move_ga, label=f"Movement (n={move.shape[0]})", linewidth=2.0)
    if rest.shape[0]:
        plt.plot(t, rest_ga, label=f"Rest (n={rest.shape[0]})", linewidth=1.6, alpha=0.8)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.title(f"Subject {args.subject} Cz Grand-Average MRCP (-2 to 0 s)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(out_path)


if __name__ == "__main__":
    main()
