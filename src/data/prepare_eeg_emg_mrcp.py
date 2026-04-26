from __future__ import annotations

"""
Prepare the EEG+EMG MRCP dataset from CSV session files into subject_<ID>.npz.

Dataset reference:
  EEG and EMG dataset for analyzing movement-related cortical potentials in hand
  gesture tasks (Mendeley DOI: 10.17632/y23s2xg6x4.1)

Expected folder structure under ``--raw-root``::

    SUBJECT01/
      SUBJECT01_Session_01_EEG.csv
      SUBJECT01_Session_01_EMG.csv
      ...
    SUBJECT02/
      ...

This converter focuses on EEG. It builds a binary dataset:

- class 1 (movement / MRCP): window before execution onset (trigger 7711)
- class 0 (rest): matching window before preparation onset (trigger 771)

Default windows are [-2.0, 0.0] seconds relative to each trigger, which matches
the MRCP-oriented low-frequency setting described in the accompanying article.
"""

import argparse
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

KNOWN_EEG_CHANNELS = [
    "AF3", "AF4", "F3", "F1", "FZ", "F2", "F4",
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C3", "C1", "CZ", "C2", "C4",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P3", "P1", "PZ", "P2", "P4",
    "PO3", "POZ", "PO4", "O1", "O2",
]

TARGET_SFREQ = 128.0


def _subject_dir(raw_root: Path, subject: int) -> Path:
    direct = raw_root / f"SUBJECT{subject:02d}"
    nested = raw_root / "SUBJECTS" / f"SUBJECT{subject:02d}"
    if direct.is_dir():
        return direct
    if nested.is_dir():
        return nested
    return direct


def _session_files(subject_dir: Path) -> List[Path]:
    return sorted(subject_dir.glob("*_EEG.csv"))


def _find_trigger_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        c = str(col).strip().lower()
        if c == "trigger" or "trigger" in c:
            return str(col)
    raise ValueError("Could not find trigger column in EEG CSV.")


def _find_eeg_columns(df: pd.DataFrame, trigger_col: str) -> List[str]:
    cols: List[str] = []
    known = {c.upper() for c in KNOWN_EEG_CHANNELS}
    for col in df.columns:
        name = str(col).strip()
        if name == trigger_col:
            continue
        if name.upper() in known:
            cols.append(name)
    if cols:
        return cols

    # Fallback: keep numeric columns except trigger and obvious time/index columns.
    out: List[str] = []
    for col in df.columns:
        name = str(col).strip()
        low = name.lower()
        if name == trigger_col or "time" in low or low in {"index", "sample"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(name)
    if not out:
        raise ValueError("Could not infer EEG columns from CSV.")

    # The published EEG+EMG MRCP CSVs expose numeric column names and include
    # a few non-EEG bookkeeping/status columns. Empirically, the 32 EEG channels
    # are the middle block after the trigger column, while the leading/trailing
    # numeric columns behave like sample counters or device status.
    if len(out) >= 38 and all(str(c).strip().isdigit() for c in out):
        return out[2:34]

    return out


def _contiguous_trigger_onsets(trigger: np.ndarray, value: int) -> np.ndarray:
    mask = np.asarray(trigger == value, dtype=bool)
    if not np.any(mask):
        return np.empty(0, dtype=int)
    prev = np.concatenate([[False], mask[:-1]])
    return np.flatnonzero(mask & ~prev)


def _extract_epochs(
    X: np.ndarray,
    onsets: Sequence[int],
    *,
    sfreq: float,
    tmin: float,
    tmax: float,
) -> np.ndarray:
    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    n_times = stop_offset - start_offset
    if n_times <= 0:
        raise ValueError(f"Invalid epoch window: tmin={tmin}, tmax={tmax}")

    epochs = []
    for onset in onsets:
        start = int(onset + start_offset)
        stop = int(onset + stop_offset)
        if start < 0 or stop > X.shape[1]:
            continue
        chunk = X[:, start:stop]
        if chunk.shape[1] == n_times:
            epochs.append(chunk.astype(np.float32, copy=False))
    if not epochs:
        return np.empty((0, X.shape[0], n_times), dtype=np.float32)
    return np.stack(epochs).astype(np.float32, copy=False)


def _prepare_subject(
    subject: int,
    raw_root: Path,
    out_root: Path,
    *,
    sfreq: float,
    movement_trigger: int,
    rest_trigger: int,
    movement_tmin: float,
    movement_tmax: float,
    rest_tmin: float,
    rest_tmax: float,
) -> dict:
    subject_dir = _subject_dir(raw_root, subject)
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Missing subject directory: {subject_dir}")

    eeg_files = _session_files(subject_dir)
    if not eeg_files:
        raise FileNotFoundError(f"No EEG CSV files found in {subject_dir}")

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    ch_names_final: List[str] | None = None

    for eeg_path in eeg_files:
        try:
            df = pd.read_csv(eeg_path)
            trigger_col = _find_trigger_column(df)
            eeg_cols = _find_eeg_columns(df, trigger_col)

            inferred_ch_names = eeg_cols
            if len(eeg_cols) == len(KNOWN_EEG_CHANNELS) and all(str(c).strip().isdigit() for c in eeg_cols):
                inferred_ch_names = KNOWN_EEG_CHANNELS.copy()

            if ch_names_final is None:
                ch_names_final = inferred_ch_names
            elif [c.upper() for c in inferred_ch_names] != [c.upper() for c in ch_names_final]:
                raise ValueError(
                    f"Channel mismatch in {eeg_path.name}; expected {ch_names_final}, got {inferred_ch_names}"
                )

            trigger = pd.to_numeric(df[trigger_col], errors="coerce").fillna(0).astype(int).to_numpy()
            X_cont = df[eeg_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32).T

            movement_onsets = _contiguous_trigger_onsets(trigger, movement_trigger)
            rest_onsets = _contiguous_trigger_onsets(trigger, rest_trigger)

            X_move = _extract_epochs(
                X_cont, movement_onsets, sfreq=sfreq, tmin=movement_tmin, tmax=movement_tmax
            )
            X_rest = _extract_epochs(
                X_cont, rest_onsets, sfreq=sfreq, tmin=rest_tmin, tmax=rest_tmax
            )

            n = min(len(X_move), len(X_rest))
            if n == 0:
                warnings.warn(f"No usable epochs in {eeg_path.name}; skipping this file.")
                continue
            X_sess = np.concatenate([X_rest[:n], X_move[:n]], axis=0).astype(np.float32, copy=False)
            y_sess = np.concatenate(
                [np.zeros(n, dtype=int), np.ones(n, dtype=int)],
                axis=0,
            )
            X_all.append(X_sess)
            y_all.append(y_sess)
        except Exception as e:
            warnings.warn(f"Skipping malformed EEG file {eeg_path.name}: {e}")
            continue

    if not X_all or ch_names_final is None:
        raise ValueError(f"No usable epochs extracted for subject {subject}")

    X = np.concatenate(X_all, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_all, axis=0).astype(int, copy=False)

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"subject_{subject}.npz"
    np.savez(
        out_path,
        X=X,
        y=y,
        sfreq=float(sfreq),
        ch_names=np.array(ch_names_final, dtype=object),
    )

    return {
        "subject": subject,
        "n_trials": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "sfreq": float(sfreq),
        "movement_trials": int(np.sum(y == 1)),
        "rest_trials": int(np.sum(y == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EEG+EMG MRCP CSV sessions to subject_<ID>.npz files."
    )
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="data/eeg_emg_mrcp/processed")
    parser.add_argument("--sfreq", type=float, default=TARGET_SFREQ)
    parser.add_argument("--movement-trigger", type=int, default=7711)
    parser.add_argument("--rest-trigger", type=int, default=771)
    parser.add_argument("--movement-tmin", type=float, default=-2.0)
    parser.add_argument("--movement-tmax", type=float, default=0.0)
    parser.add_argument("--rest-tmin", type=float, default=-2.0)
    parser.add_argument("--rest-tmax", type=float, default=0.0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    for subject in args.subjects:
        print(f"Preparing EEG+EMG MRCP subject {subject}...", flush=True)
        info = _prepare_subject(
            subject=subject,
            raw_root=raw_root,
            out_root=out_root,
            sfreq=float(args.sfreq),
            movement_trigger=int(args.movement_trigger),
            rest_trigger=int(args.rest_trigger),
            movement_tmin=float(args.movement_tmin),
            movement_tmax=float(args.movement_tmax),
            rest_tmin=float(args.rest_tmin),
            rest_tmax=float(args.rest_tmax),
        )
        print(
            f"  ✓ subj={info['subject']} | trials={info['n_trials']} "
            f"(rest={info['rest_trials']}, move={info['movement_trials']}) "
            f"| ch={info['n_channels']} | times={info['n_times']} | sfreq={info['sfreq']} Hz",
            flush=True,
        )


if __name__ == "__main__":
    main()
