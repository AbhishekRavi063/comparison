from __future__ import annotations

"""
Prepare Weibo2014 motor imagery dataset.

Downloads via MOABB's Weibo2014 (left vs right hand MI, 10 subjects,
originally 200 Hz, 60 EEG channels), resamples to 250 Hz, and saves each
subject as:

    <out_root>/subject_<ID>.npz

with keys: X (float32), y, sfreq (250.0), ch_names.

Memory note: processes one subject at a time, calls gc.collect() after each.
This is safe on 16 GB M4 Pro — peak usage ~400 MB per subject.

Usage:
    # All 10 subjects
    python -m src.data.prepare_weibo2014 --subjects 1 2 3 4 5 6 7 8 9 10 \
        --out-root data/weibo2014/processed
"""

import argparse
import gc
import os
import warnings
from pathlib import Path

import numpy as np

TARGET_SFREQ = 250.0


def _prepare_subject(subject: int, out_root: Path) -> dict:
    """Download, epoch, resample, and save one subject. Returns info dict."""
    from moabb.datasets import Weibo2014
    from moabb.paradigms import LeftRightImagery

    out_root.mkdir(parents=True, exist_ok=True)

    dataset = Weibo2014()
    paradigm = LeftRightImagery()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_np, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject])

    X = X_np.astype("float32", copy=False)

    # Get original sfreq and channel names from raw MNE object
    sfreq_orig = TARGET_SFREQ  # default fallback
    ch_names = [f"EEG{i:02d}" for i in range(X.shape[1])]
    try:
        raw_dict = dataset.get_data(subjects=[subject])
        first_raw = next(
            iter(next(iter(next(iter(raw_dict.values())).values())).values())
        )
        sfreq_orig = float(first_raw.info["sfreq"])
        ch_names_raw = [
            c for c in first_raw.info["ch_names"]
            if c not in ("STI 014", "STI014", "Status")
        ]
        ch_names = ch_names_raw[: X.shape[1]]
        del first_raw, raw_dict
    except Exception:
        pass

    # Map string labels → integers
    unique_labels = sorted(set(labels))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[lab] for lab in labels], dtype=int)

    # Resample if needed (200 Hz → 250 Hz)
    if abs(sfreq_orig - TARGET_SFREQ) > 1.0:
        import mne
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq_orig, ch_types="eeg")
        ep_obj = mne.EpochsArray(X, info, verbose="ERROR")
        ep_obj.resample(TARGET_SFREQ, npad="auto", verbose="ERROR")
        X = ep_obj.get_data().astype("float32")
        del ep_obj
    sfreq_out = TARGET_SFREQ

    out_path = out_root / f"subject_{subject}.npz"
    np.savez(
        out_path,
        X=X,
        y=y,
        sfreq=sfreq_out,
        ch_names=np.array(ch_names, dtype=object),
    )

    result = dict(
        subject=subject,
        n_trials=X.shape[0],
        n_channels=X.shape[1],
        n_times=X.shape[2],
        sfreq=sfreq_out,
        sfreq_orig=sfreq_orig,
        classes=unique_labels,
    )
    del X, y
    gc.collect()
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_mne = project_root / ".mne_home" / "MNE-data"
    default_mne.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DATA", str(default_mne))

    parser = argparse.ArgumentParser(
        description="Download and convert Weibo2014 to subject_<ID>.npz (250 Hz, float32)."
    )
    parser.add_argument("--subjects", type=int, nargs="+", required=True,
                        help="Subject IDs (1–10 for Weibo2014).")
    parser.add_argument("--out-root", type=str, default="data/weibo2014/processed")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    print(f"Output root: {out_root}", flush=True)

    for subj in args.subjects:
        npz = out_root / f"subject_{subj}.npz"
        if npz.exists():
            print(f"  subject {subj:2d}: already cached — skip", flush=True)
            continue
        print(f"  subject {subj:2d}: downloading + preparing...", flush=True)
        try:
            info = _prepare_subject(subject=subj, out_root=out_root)
            print(
                f"  subject {subj:2d}: ✓ "
                f"trials={info['n_trials']} ch={info['n_channels']} "
                f"times={info['n_times']} sfreq={info['sfreq']:.0f}Hz "
                f"(orig {info['sfreq_orig']:.0f}Hz)",
                flush=True,
            )
        except Exception as e:
            print(f"  subject {subj:2d}: ✗ {e}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
