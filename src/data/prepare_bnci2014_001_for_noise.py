from __future__ import annotations

"""
Prepare BNCI2014_001 as the third dataset (labelled 'gigadb' for the noise comparison).

Already at 250 Hz; no resampling needed. Saves each subject as:

    <out_root>/subject_<ID>.npz

with keys: X (float32), y, sfreq (250.0), ch_names.

Usage:
    python -m src.data.prepare_bnci2014_001_for_noise --subjects 1 2 3 \
        --out-root data/gigadb/processed
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np

TARGET_SFREQ = 250.0


def _prepare_subject(subject: int, out_root: Path) -> dict:
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    out_root.mkdir(parents=True, exist_ok=True)

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery()

    try:
        epochs = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
        X = epochs.get_data().astype("float32", copy=False)
        labels = epochs.events[:, 2]
        ev_id_inv = {v: k for k, v in epochs.event_id.items()}
        labels = np.array([ev_id_inv.get(code, str(code)) for code in labels])
        ch_names = list(epochs.ch_names)
    except Exception:
        X, labels, _meta = paradigm.get_data(dataset=dataset, subjects=[subject])
        X = X.astype("float32", copy=False)
        ch_names = [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "P1", "Pz", "P2", "POz",
        ]

    unique_labels = sorted(set(labels))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[lab] for lab in labels], dtype=int)

    sfreq_out = TARGET_SFREQ

    out_path = out_root / f"subject_{subject}.npz"
    np.savez(
        out_path,
        X=X,
        y=y,
        sfreq=sfreq_out,
        ch_names=np.array(ch_names, dtype=object),
    )

    info_dict = {
        "subject": subject,
        "n_trials": X.shape[0],
        "n_channels": X.shape[1],
        "n_times": X.shape[2],
        "sfreq": sfreq_out,
        "classes": unique_labels,
    }
    del X, y
    gc.collect()
    return info_dict


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_mne = project_root / ".mne_home" / "MNE-data"
    default_mne.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DATA", str(default_mne))

    parser = argparse.ArgumentParser(
        description="Prepare BNCI2014_001 as the 'gigadb' third dataset for noise comparison."
    )
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--out-root", type=str, default="data/gigadb/processed")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    for subj in args.subjects:
        print(f"Preparing BNCI2014_001 (gigadb) subject {subj}...", flush=True)
        info = _prepare_subject(subject=subj, out_root=out_root)
        print(
            f"  ✓ subj={subj} | trials={info['n_trials']} | ch={info['n_channels']} "
            f"| times={info['n_times']} | sfreq={info['sfreq']} Hz",
            flush=True,
        )


if __name__ == "__main__":
    main()
