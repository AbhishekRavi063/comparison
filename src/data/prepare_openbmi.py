from __future__ import annotations

"""
Prepare OpenBMI (Lee2019_MI) motor imagery dataset.

Downloads via MOABB's Lee2019_MI (left vs right hand MI, 54 subjects, 1000 Hz, 62 ch),
resamples to 250 Hz, and saves each subject as:

    <out_root>/subject_<ID>.npz

with keys: X (float32), y, sfreq (250.0), ch_names.

Usage:
    # 3 subjects for quick inspection
    python -m src.data.prepare_openbmi --subjects 1 2 3 --out-root data/openbmi/processed

    # All 54 subjects
    python -m src.data.prepare_openbmi --subjects $(seq 1 54) --out-root data/openbmi/processed
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np

TARGET_SFREQ = 250.0


def _prepare_subject(subject: int, out_root: Path) -> dict:
    """Download, epoch, resample, and save one subject. Returns diagnostic info."""
    from moabb.datasets import Lee2019_MI
    from moabb.paradigms import LeftRightImagery

    out_root.mkdir(parents=True, exist_ok=True)

    dataset = Lee2019_MI()
    paradigm = LeftRightImagery()

    # Use return_epochs=True to get both data and ch_names in one call.
    try:
        epochs = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
        X = epochs.get_data().astype("float32", copy=False)
        labels = epochs.events[:, 2]  # integer codes; convert to str via event_id
        ev_id_inv = {v: k for k, v in epochs.event_id.items()}
        labels = np.array([ev_id_inv.get(code, str(code)) for code in labels])
        sfreq_orig = float(epochs.info["sfreq"])
        ch_names = list(epochs.ch_names)
    except Exception:
        # Fallback: old API without return_epochs
        X, labels, _meta = paradigm.get_data(dataset=dataset, subjects=[subject])
        X = X.astype("float32", copy=False)
        sfreq_orig = float(paradigm.resample) if paradigm.resample else 1000.0
        ch_names = [f"EEG{i+1:02d}" for i in range(X.shape[1])]

    # Map string labels to integers
    unique_labels = sorted(set(labels))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[lab] for lab in labels], dtype=int)

    # Resample from 1000 Hz → 250 Hz using MNE's resample
    if abs(sfreq_orig - TARGET_SFREQ) > 1.0:
        import mne
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq_orig, ch_types="eeg")
        ep_obj = mne.EpochsArray(X, info, verbose="ERROR")
        ep_obj.resample(TARGET_SFREQ, npad="auto", verbose="ERROR")
        X = ep_obj.get_data().astype("float32")
        sfreq_out = TARGET_SFREQ
    else:
        sfreq_out = sfreq_orig

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
    import os

    project_root = Path(__file__).resolve().parents[2]
    default_mne = project_root / ".mne_home" / "MNE-data"
    default_mne.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DATA", str(default_mne))

    parser = argparse.ArgumentParser(
        description="Download and convert OpenBMI (Lee2019_MI) to subject_<ID>.npz."
    )
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--out-root", type=str, default="data/openbmi/processed")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    for subj in args.subjects:
        print(f"Preparing OpenBMI subject {subj}...", flush=True)
        info = _prepare_subject(subject=subj, out_root=out_root)
        print(
            f"  ✓ subj={subj} | trials={info['n_trials']} | ch={info['n_channels']} "
            f"| times={info['n_times']} | sfreq={info['sfreq']} Hz",
            flush=True,
        )


if __name__ == "__main__":
    main()
