from __future__ import annotations

"""
Prepare PhysioNet EEG Motor Movement/Imagery (EEGBCI) dataset for this project.

This script downloads EEGBCI data via MNE, extracts motor imagery epochs
for a binary hands-vs-feet task, and saves each subject as:

    <output_root>/subject_<ID>.npz

with keys: X, y, sfreq, ch_names.
"""

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np


def _prepare_subject(
    subject: int,
    out_root: Path,
    runs: Iterable[int],
    tmin: float,
    tmax: float,
) -> None:
    from mne.datasets import eegbci
    from mne.io import read_raw_edf, concatenate_raws
    from mne import Epochs, events_from_annotations, pick_types

    out_root.mkdir(parents=True, exist_ok=True)

    # Download & load raw EDF files for this subject and runs
    raw_fnames = eegbci.load_data(subjects=subject, runs=list(runs))
    raws = [read_raw_edf(f, preload=True, verbose="ERROR") for f in raw_fnames]
    raw = concatenate_raws(raws)

    eegbci.standardize(raw)  # set channel names
    try:
        raw.set_montage("standard_1020", verbose="ERROR")
    except Exception:
        # Montage is helpful but not strictly necessary for this benchmark
        pass

    # Events: for runs [6, 10, 14] we follow the MNE example:
    # T1 -> hands, T2 -> feet
    events, event_id_annot = events_from_annotations(raw)
    event_id = dict(hands=2, feet=3)

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
    epochs = Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    X = epochs.get_data().astype("float32")
    y_raw = epochs.events[:, 2]
    classes = np.unique(y_raw)
    # Map to 0..C-1 for consistency
    class_map = {c: i for i, c in enumerate(sorted(classes))}
    y = np.vectorize(class_map.get)(y_raw).astype(int)

    sfreq = float(epochs.info["sfreq"])
    ch_names: List[str] = list(epochs.ch_names)

    out_path = out_root / f"subject_{subject}.npz"
    np.savez(
        out_path,
        X=X,
        y=y,
        sfreq=sfreq,
        ch_names=np.array(ch_names, dtype=object),
    )


def main() -> None:
    # Ensure MNE has a valid download directory (avoid FileNotFoundError)
    import os
    mne_data = os.environ.get("MNE_DATA")
    if mne_data:
        Path(mne_data).mkdir(parents=True, exist_ok=True)
    else:
        project_root = Path(__file__).resolve().parents[2]
        default_mne = project_root / ".mne_home" / "MNE-data"
        default_mne.mkdir(parents=True, exist_ok=True)
        os.environ["MNE_DATA"] = str(default_mne)

    parser = argparse.ArgumentParser(
        description=(
            "Download and convert PhysioNet EEGBCI motor imagery data "
            "to subject_<ID>.npz files for the CSP / tangent benchmark."
        )
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        required=True,
        help="Subject IDs to prepare (1–109 for EEGBCI).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data",
        help="Output directory where subject_<ID>.npz files will be written.",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=-1.0,
        help="Epoch start time relative to cue (seconds).",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=4.0,
        help="Epoch end time relative to cue (seconds).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[6, 10, 14],
        help="EEGBCI run numbers to use (default: [6,10,14] = hands vs feet MI).",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    for subj in args.subjects:
        _prepare_subject(
            subject=subj,
            out_root=out_root,
            runs=args.runs,
            tmin=args.tmin,
            tmax=args.tmax,
        )


if __name__ == "__main__":
    main()

