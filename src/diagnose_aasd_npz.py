"""Label / shape diagnostics for AASD ``subject_*.npz`` (professor checklist).

Usage:
  python -m src.diagnose_aasd_npz --data-root data/aasd/npz --subjects 1 2 3 4

Checks class balance, whether per-window labels vary within 60 s trials, and
rough agreement between window-0 label vs majority label per trial.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="AASD NPZ label diagnostics.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/aasd/npz"),
        help="Folder with subject_<id>.npz",
    )
    p.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        required=True,
        help="Subject IDs (integers matching NPZ filenames).",
    )
    args = p.parse_args()

    for sid in args.subjects:
        path = args.data_root / f"subject_{sid}.npz"
        if not path.is_file():
            print(f"[missing] {path}")
            continue
        z = np.load(path, allow_pickle=True)
        X = np.asarray(z["X"])
        y = np.asarray(z["y"]).ravel()
        z.close()

        uni, counts = np.unique(y, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(uni, counts)}
        print(f"\n=== Subject {sid} ===")
        print(f"  X shape: {X.shape}  y shape: {y.shape}")
        print(f"  class counts: {dist}")
        if len(uni) < 2:
            print("  WARNING: single class — classification is meaningless.")

        n = int(y.shape[0])
        if n % 60 == 0 and n >= 60:
            n_tri = n // 60
            yt = y.reshape(n_tri, 60)
            switches = np.sum(yt[:, 1:] != yt[:, :-1], axis=1)
            const_trials = int(np.sum(np.all(yt == yt[:, :1], axis=1)))
            maj = np.array(
                [np.bincount(yt[i].astype(int)).argmax() for i in range(n_tri)],
                dtype=int,
            )
            agree_first = int(np.sum(maj == yt[:, 0]))
            print(
                f"  reshaped as {n_tri} trials × 60 windows: "
                f"constant-label trials={const_trials}/{n_tri}"
            )
            print(
                f"  per-trial window-to-window switches: "
                f"mean={switches.mean():.2f}  min={switches.min()}  max={switches.max()}"
            )
            print(
                f"  majority label == first-window label: {agree_first}/{n_tri} trials"
            )
        else:
            print(
                f"  (skip trial reshape: n_windows={n} not a multiple of 60)"
            )


if __name__ == "__main__":
    main()
