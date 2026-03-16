from __future__ import annotations

"""
Prepare BNCI 2014-001 (BCI Competition IV 2a) for this project.

This script uses MOABB's BNCI2014_001 dataset together with the
LeftRightImagery paradigm (binary left vs right hand MI) and saves each
subject as:

    <output_root>/subject_<ID>.npz

with keys: X, y, sfreq, ch_names.
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _prepare_subject(subject: int, out_root: Path) -> None:
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    out_root.mkdir(parents=True, exist_ok=True)

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery()

    # X: (n_trials, n_channels, n_times), labels: (n_trials,)
    X, labels, meta = paradigm.get_data(
        dataset=dataset,
        subjects=[subject],
    )

    # X is already in (trials, channels, times) and float; cast to float32
    X = X.astype("float32", copy=False)

    # Map labels (typically 'left_hand', 'right_hand') to integers 0,1
    unique_labels = sorted(set(labels))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[lab] for lab in labels], dtype=int)

    # Sampling frequency and channel names can be obtained from the dataset's
    # internal MNE objects; however MOABB exposes them via the paradigm's
    # internal processing. We can recover them through a single example epoch.
    # To avoid deep introspection, we fall back to known dataset constants.
    sfreq = 250.0  # BNCI 2014-001 sampling frequency (Hz)

    # Channel names are available from the dataset; see BNCI2014_001 docs
    # for the 22-channel montage. Hard-code here for robustness.
    ch_names = [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    ]

    out_path = out_root / f"subject_{subject}.npz"
    np.savez(
        out_path,
        X=X,
        y=y,
        sfreq=sfreq,
        ch_names=np.array(ch_names, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and convert BNCI 2014-001 (BCI Competition IV 2a) "
            "left vs right hand motor imagery data to subject_<ID>.npz files."
        )
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        required=True,
        help="Subject IDs to prepare (1–9 for BNCI2014_001).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data",
        help="Output directory where subject_<ID>.npz files will be written.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    for subj in args.subjects:
        _prepare_subject(subject=subj, out_root=out_root)


if __name__ == "__main__":
    main()

