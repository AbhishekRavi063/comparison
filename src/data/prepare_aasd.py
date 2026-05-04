from __future__ import annotations

"""
Prepare the AASD (Auditory Attention Switching Dataset) for the benchmark pipeline.

Dataset reference:
  An Open Non-Invasive EEG Dataset for Spontaneous Auditory Attention Switch Decoding
  Nature Scientific Data, 2026
  Download: https://zenodo.org/records/17413336
  GitHub:   https://github.com/XXuefeii/AASD-Processing

Expected folder layout (processed MAT files)::

    <raw-root>/
      subject_1.mat   |  S01.mat  |  SUBJECT01.mat   (any of these naming patterns)
      subject_2.mat
      ...

Each MAT file contains one subject's processed EEG.  The converter windows
each 60-second trial into non-overlapping epochs and, when available, assigns
the time-varying attention label from within-trial switch events to each
window.  If dynamic events are unavailable it falls back to one trial-level
label per 60-second trial.

    <out-root>/subject_<ID>.npz

with keys: X (float32, n_windows x n_channels x n_times), y, sfreq, ch_names.

Usage::

    # 2 subjects smoke test
    python -m src.data.prepare_aasd --subjects 1 2 \\
        --raw-root data/aasd/processed --out-root data/aasd/npz

    # All 18 subjects
    python -m src.data.prepare_aasd --subjects $(seq 1 18) \\
        --raw-root data/aasd/processed --out-root data/aasd/npz
"""

import argparse
import gc
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Target sampling rate after optional resampling (native is typically 500 Hz).
# 250 Hz is sufficient for 8-12 Hz alpha-band decoding and halves memory.
TARGET_SFREQ = 250.0

# 1-second window length (in samples at TARGET_SFREQ).
WINDOW_SAMPLES = int(TARGET_SFREQ)

# Standard 64-channel 10-20 names used by the AASD cap layout.
# These are used as fallback when the MAT file does not contain channel names.
FALLBACK_CH_NAMES = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7",  "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",  "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7",  "P5",  "P3",  "P1",  "Pz",  "P2",  "P4",  "P6",  "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1",  "Oz",  "O2",
    "Iz",  "I1",  "I2",
    "TP9", "TP10",
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_subject_file(raw_root: Path, subject: int) -> Path:
    """Try common AASD naming patterns; raise FileNotFoundError if none found.

    Also performs a glob search inside subject sub-folders so that any
    filename (e.g. ``p1_processed.mat``, ``sub-01.mat``) is found
    automatically without hardcoding the exact name.
    """
    candidates = [
        raw_root / f"subject_{subject}.mat",
        raw_root / f"subject{subject:02d}.mat",
        raw_root / f"S{subject:02d}.mat",
        raw_root / f"S{subject}.mat",
        raw_root / f"s{subject:02d}.mat",
        raw_root / f"SUBJECT{subject:02d}.mat",
        raw_root / f"Sub{subject:02d}.mat",
        raw_root / f"{subject:02d}.mat",
        # Common Zenodo layout: Processed EEG/S1/S1.mat
        raw_root / f"S{subject}" / f"S{subject}.mat",
        raw_root / f"S{subject:02d}" / f"S{subject:02d}.mat",
        # CNT fallback
        raw_root / f"subject_{subject}.cnt",
        raw_root / f"S{subject:02d}.cnt",
        raw_root / f"S{subject}.cnt",
        raw_root / f"S{subject}" / f"S{subject}.cnt",
        raw_root / f"S{subject:02d}" / f"S{subject:02d}.cnt",
    ]
    for p in candidates:
        if p.exists():
            return p

    # ---- Glob fallback: find any .mat (or .cnt) inside the subject sub-folder.
    # This handles unknown filenames inside S1/, S01/, s1/, etc.
    subfolder_patterns = [
        raw_root / f"S{subject}",
        raw_root / f"S{subject:02d}",
        raw_root / f"s{subject}",
        raw_root / f"s{subject:02d}",
        raw_root / f"Sub{subject}",
        raw_root / f"Sub{subject:02d}",
        raw_root / f"subject_{subject}",
        raw_root / f"subject{subject:02d}",
    ]
    for folder in subfolder_patterns:
        if folder.is_dir():
            # Prefer .mat; fall back to .cnt
            for ext in ("*.mat", "*.cnt"):
                hits = sorted(folder.glob(ext))
                if len(hits) == 1:
                    return hits[0]
                if len(hits) > 1:
                    # Multiple files — pick the one whose stem most closely matches
                    # the subject number (e.g. "S1", "sub01", "p1", etc.)
                    import re
                    for h in hits:
                        if re.search(rf"\b0*{subject}\b", h.stem, re.IGNORECASE):
                            return h
                    # Nothing matched by number — return first alphabetically
                    return hits[0]

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find AASD file for subject {subject}.\n"
        f"Tried exact paths:\n  {tried}\n"
        f"Also searched for any *.mat/*.cnt in sub-folders: "
        f"{[str(f) for f in subfolder_patterns]}\n"
        f"Place MAT files in {raw_root} or a numbered sub-folder."
    )


def _resolve_raw_root(raw_root: Path) -> Path:
    """Resolve raw root and fall back to common AASD dataset folders."""
    if raw_root.exists():
        return raw_root

    cwd = Path.cwd()
    candidates = [
        cwd / "data" / "Processed EEG",
        cwd / "data" / "processed",
        cwd / "data" / "aasd" / "processed",
        cwd / "Processed EEG",
    ]
    for candidate in candidates:
        if candidate.exists():
            warnings.warn(
                f"Raw root '{raw_root}' does not exist; using '{candidate}' instead."
            )
            return candidate
    return raw_root


# ---------------------------------------------------------------------------
# MAT file loading (handles scipy v5 and h5py v7.3)
# ---------------------------------------------------------------------------

def _load_mat(path: Path) -> dict:
    """Load a MATLAB file; returns dict of top-level variables."""
    try:
        import scipy.io
        data = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        # Remove MATLAB metadata keys
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception as scipy_err:
        # scipy can't handle MATLAB v7.3 (HDF5); fall back to h5py
        try:
            import h5py
            out: dict = {}
            with h5py.File(str(path), "r") as f:
                def _extract(obj):
                    if isinstance(obj, h5py.Dataset):
                        return obj[()]
                    elif isinstance(obj, h5py.Group):
                        return {k: _extract(v) for k, v in obj.items()}
                    return obj
                for k, v in f.items():
                    out[k] = _extract(v)
            return out
        except Exception as h5_err:
            raise RuntimeError(
                f"Failed to load {path}.\n"
                f"  scipy error: {scipy_err}\n"
                f"  h5py error:  {h5_err}\n"
                "Install h5py for MATLAB v7.3 support: pip install h5py"
            )


# ---------------------------------------------------------------------------
# Data extraction from MAT dict
# ---------------------------------------------------------------------------

def _extract_eeg_data(mat: dict, subject: int) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """
    Extract (data, labels, sfreq, ch_names) from a loaded MAT dict.

    Tries multiple common variable/struct layouts used by EEGLAB and custom pipelines.
    Returns:
        data   : float32 array, shape (n_trials, n_channels, n_times)
        labels : int array, shape (n_trials,), values 0 (left) or 1 (right)
        sfreq  : float, sampling rate in Hz
        ch_names: list of str
    """
    keys = set(mat.keys())

    # ---- Strategy 1: EEGLAB EEG struct ----
    if "EEG" in keys:
        return _extract_eeglab_struct(mat["EEG"], subject)
    if "EEG_new" in keys:
        return _extract_eeg_new_struct(mat["EEG_new"], subject)

    # ---- Strategy 2: Simple arrays (data/label/fs or similar) ----
    data_key = _pick_key(keys, ["data", "eeg", "EEG_data", "X", "signal"])
    label_key = _pick_key(keys, ["label", "labels", "y", "condition", "attention"])
    sfreq_key = _pick_key(keys, ["fs", "srate", "sfreq", "Fs", "sampling_rate"])

    if data_key is None:
        raise ValueError(
            f"Subject {subject}: cannot find EEG data array in MAT file.\n"
            f"Available keys: {sorted(keys)}\n"
            "Expected keys like: data, eeg, EEG_data, X, or an EEG struct."
        )

    raw_data = np.asarray(mat[data_key])
    sfreq = float(np.asarray(mat[sfreq_key]).flat[0]) if sfreq_key else 500.0

    # Ensure shape (n_trials, n_channels, n_times)
    data = _coerce_data_shape(raw_data, subject)

    # Labels
    labels = _extract_labels(mat, label_key, data.shape[0], subject)

    # Channel names
    ch_names = _extract_ch_names(mat, data.shape[1])

    return data.astype(np.float32, copy=False), labels, sfreq, ch_names


def _extract_eeg_new_struct(eeg, subject: int) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """Extract from AASD-style EEG_new struct with event matrix labels."""

    raw = getattr(eeg, "data", None)
    if raw is None:
        raise ValueError(f"Subject {subject}: EEG_new struct has no .data field.")

    data = _coerce_data_shape(np.asarray(raw), subject)
    n_trials = int(data.shape[0])

    # AASD files often omit explicit srate; infer from 60s trial duration when possible.
    srate = getattr(eeg, "srate", None)
    if srate is not None:
        sfreq = float(np.asarray(srate).flat[0])
    else:
        sfreq = float(data.shape[2] / 60.0) if data.shape[2] >= 60 else 128.0
        if sfreq <= 0:
            sfreq = 128.0

    ch_names = _fallback_ch_names(int(data.shape[1]))
    labels = _extract_labels_from_eeg_new_event(getattr(eeg, "event", None), n_trials, subject)
    return data.astype(np.float32, copy=False), labels, sfreq, ch_names


def _extract_labels_from_eeg_new_event(event_obj, n_trials: int, subject: int) -> np.ndarray:
    """
    Parse trial labels from AASD EEG_new.event matrix.
    Expected matrix columns: [event_code, latency, urevent, duration, trial_idx].
    We derive one base event code per trial and binarize as odd/even.
    """
    if event_obj is None:
        warnings.warn(
            f"Subject {subject}: EEG_new has no event matrix; defaulting labels to 0."
        )
        return np.zeros(n_trials, dtype=int)

    try:
        ev = np.asarray(event_obj)
        if ev.ndim != 2 or ev.shape[1] < 5:
            raise ValueError("unexpected event matrix shape")

        code_col = ev[:, 0]
        trial_col = ev[:, 4]
        trial_ids = np.array([int(float(str(v))) for v in trial_col], dtype=int)
        code_vals = np.array([int(float(str(v))) for v in code_col], dtype=int)

        trial_codes = []
        for t in range(1, n_trials + 1):
            idx = np.where(trial_ids == t)[0]
            if len(idx) == 0:
                trial_codes.append(11 + ((t - 1) % 60))
                continue
            # Ignore recurrent within-trial markers used in these files.
            non_recurrent = [code_vals[i] for i in idx if code_vals[i] not in (179, 184)]
            trial_codes.append(non_recurrent[0] if non_recurrent else int(code_vals[idx[0]]))

        # AASD event codes are sequential; odd/even splits two attention classes (30/30).
        labels = np.array([0 if (c % 2 == 1) else 1 for c in trial_codes], dtype=int)
        return labels
    except Exception as exc:
        warnings.warn(
            f"Subject {subject}: failed EEG_new event-label parsing ({exc}); defaulting to 0."
        )
        return np.zeros(n_trials, dtype=int)


def _extract_window_labels_from_eeg_new_event(
    event_obj,
    *,
    n_trials: int,
    n_times: int,
    sfreq: float,
    window_s: float,
    subject: int,
) -> Optional[np.ndarray]:
    """Build per-window labels from AASD within-trial switch markers.

    The AASD ``EEG_new.event`` matrix uses recurrent event codes during each
    60-second trial.  In the public files these are commonly 179 and 184, which
    encode the current left/right attention state.  We label every window by the
    latest switch marker at that window midpoint.  If a file does not expose the
    expected event matrix, callers should fall back to trial-level labels.
    """
    if event_obj is None:
        return None
    try:
        ev = np.asarray(event_obj)
        if ev.ndim != 2 or ev.shape[1] < 5:
            return None
        code_vals = np.array([int(float(str(v))) for v in ev[:, 0]], dtype=int)
        lat_vals = np.array([float(str(v)) for v in ev[:, 1]], dtype=float)
        trial_ids = np.array([int(float(str(v))) for v in ev[:, 4]], dtype=int)
        state_code_to_label = {179: 0, 184: 1}
        n_windows = int(n_times // int(round(float(sfreq) * float(window_s))))
        if n_windows <= 0:
            return None
        labels = np.empty((n_trials, n_windows), dtype=int)
        trial_fallback = _extract_labels_from_eeg_new_event(event_obj, n_trials, subject)

        for trial_i in range(n_trials):
            trial_num = trial_i + 1
            rows = np.flatnonzero(trial_ids == trial_num)
            if rows.size == 0:
                labels[trial_i, :] = int(trial_fallback[trial_i])
                continue
            state_rows = [
                r for r in rows.tolist() if int(code_vals[r]) in state_code_to_label
            ]
            if not state_rows:
                labels[trial_i, :] = int(trial_fallback[trial_i])
                continue

            base_rows = [
                r for r in rows.tolist() if int(code_vals[r]) not in state_code_to_label
            ]
            # The trial-start code (11-70 in the public EEG_new files) encodes
            # the attention state before the first recurrent 179/184 switch
            # marker.  Using the first switch marker for the whole trial flips
            # the early windows in many trials.
            initial_label = int(trial_fallback[trial_i])
            if base_rows:
                base_first = min(base_rows, key=lambda r: float(lat_vals[r]))
                initial_label = 0 if int(code_vals[base_first]) % 2 == 1 else 1

            lats = lat_vals[state_rows].astype(float)
            # Latencies may be within-trial samples or continuous/absolute samples.
            # Convert robustly to 0-based samples relative to this trial.
            if np.nanmax(lats) > n_times + 1:
                rel_lats = lats - np.nanmin(lat_vals[rows])
            else:
                rel_lats = lats - 1.0  # MATLAB-style 1-based sample index
            order = np.argsort(rel_lats)
            rel_lats = rel_lats[order]
            state_labels = np.array(
                [state_code_to_label[int(code_vals[state_rows[j]])] for j in order],
                dtype=int,
            )
            labels[trial_i, :] = int(initial_label)
            win_samples = int(round(float(sfreq) * float(window_s)))
            for win_i in range(n_windows):
                midpoint = (win_i + 0.5) * win_samples
                state_idx = int(np.searchsorted(rel_lats, midpoint, side="right") - 1)
                if state_idx >= 0:
                    labels[trial_i, win_i] = int(state_labels[state_idx])
        return labels
    except Exception as exc:
        warnings.warn(
            f"Subject {subject}: failed dynamic EEG_new event-label parsing ({exc}); "
            "falling back to trial-level labels."
        )
        return None


def _extract_valid_windows_from_eeg_new_event(
    event_obj,
    *,
    n_trials: int,
    n_times: int,
    sfreq: float,
    window_s: float,
) -> Optional[np.ndarray]:
    """Mask windows after the first completed attention switch in each trial.

    AASD labels are defined from recorded attentional focus after completed
    switches. Windows before the first recurrent 179/184 state marker are a
    trial-initial state rather than a completed-switch state, so downstream
    analyses can exclude them without losing the 60-window trial layout.
    """
    if event_obj is None:
        return None
    try:
        ev = np.asarray(event_obj)
        if ev.ndim != 2 or ev.shape[1] < 5:
            return None
        code_vals = np.array([int(float(str(v))) for v in ev[:, 0]], dtype=int)
        lat_vals = np.array([float(str(v)) for v in ev[:, 1]], dtype=float)
        trial_ids = np.array([int(float(str(v))) for v in ev[:, 4]], dtype=int)
        n_windows = int(n_times // int(round(float(sfreq) * float(window_s))))
        if n_windows <= 0:
            return None
        mask = np.ones((n_trials, n_windows), dtype=bool)
        win_samples = int(round(float(sfreq) * float(window_s)))

        for trial_i in range(n_trials):
            trial_num = trial_i + 1
            rows = np.flatnonzero(trial_ids == trial_num)
            if rows.size == 0:
                continue
            state_rows = [
                r for r in rows.tolist() if int(code_vals[r]) in (179, 184)
            ]
            if not state_rows:
                continue
            lats = lat_vals[state_rows].astype(float)
            if np.nanmax(lats) > n_times + 1:
                rel_lats = lats - np.nanmin(lat_vals[rows])
            else:
                rel_lats = lats - 1.0
            first_state = float(np.nanmin(rel_lats))
            for win_i in range(n_windows):
                midpoint = (win_i + 0.5) * win_samples
                if midpoint < first_state:
                    mask[trial_i, win_i] = False
        return mask
    except Exception as exc:
        warnings.warn(
            f"Failed AASD valid-window mask extraction ({exc}); all windows marked valid."
        )
        return None


def _extract_eeglab_struct(eeg, subject: int) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """Extract from EEGLAB EEG struct (loaded via scipy struct_as_record=False)."""
    import scipy.io

    def _get(obj, *attrs):
        for a in attrs:
            try:
                obj = getattr(obj, a)
            except AttributeError:
                try:
                    obj = obj[a]
                except (KeyError, TypeError):
                    return None
        return obj

    raw = _get(eeg, "data")
    if raw is None:
        raise ValueError(f"Subject {subject}: EEG struct has no .data field.")

    srate = _get(eeg, "srate")
    sfreq = float(srate) if srate is not None else 500.0

    raw = np.asarray(raw)
    data = _coerce_data_shape(raw, subject)
    n_trials, n_channels, _ = data.shape

    # Channel names from chanlocs
    ch_names: List[str] = []
    chanlocs = _get(eeg, "chanlocs")
    if chanlocs is not None:
        try:
            locs = np.atleast_1d(chanlocs)
            for loc in locs:
                try:
                    name = getattr(loc, "labels", None) or loc.get("labels", None)
                    if name is not None:
                        ch_names.append(str(name).strip())
                except Exception:
                    pass
        except Exception:
            pass
    if len(ch_names) != n_channels:
        ch_names = _fallback_ch_names(n_channels)

    # Labels from epoch struct or event
    epoch = _get(eeg, "epoch")
    labels = None
    if epoch is not None:
        try:
            epochs_arr = np.atleast_1d(epoch)
            raw_lbls = []
            for ep in epochs_arr:
                ev = getattr(ep, "eventtype", None)
                if ev is None:
                    ev = ep.get("eventtype", None) if hasattr(ep, "get") else None
                raw_lbls.append(str(ev) if ev is not None else "")
            labels = _parse_label_strings(raw_lbls, subject)
        except Exception:
            pass

    if labels is None:
        labels = np.zeros(n_trials, dtype=int)
        warnings.warn(
            f"Subject {subject}: could not extract labels from EEGLAB struct. "
            "Defaulting all labels to 0. Check MAT file structure."
        )

    return data.astype(np.float32, copy=False), labels, sfreq, ch_names


def _coerce_data_shape(arr: np.ndarray, subject: int) -> np.ndarray:
    """Ensure EEG array is (n_trials, n_channels, n_times)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        # (channels, times) — single continuous recording; treat as 1 trial
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        # Could be (ch, times, trials) or (trials, ch, times)
        # Heuristic: smallest dim is likely n_trials for AASD (60 trials)
        # Most MATLAB EEGLAB format: (ch, times, trials) → transpose
        s = arr.shape
        if s[0] < s[1] and s[0] < s[2]:
            # (trials, ch, times) — already correct if trials is smallest
            pass
        elif s[2] < s[0] and s[2] < s[1]:
            # (ch, times, trials) — MATLAB default
            arr = arr.transpose(2, 0, 1)
        # else: assume (trials, ch, times)
    else:
        raise ValueError(
            f"Subject {subject}: unexpected EEG data shape {arr.shape}. "
            "Expected 2D (ch × times) or 3D (ch × times × trials) or (trials × ch × times)."
        )
    return arr


def _extract_labels(mat: dict, label_key: Optional[str], n_trials: int, subject: int) -> np.ndarray:
    """Extract binary labels (0=left, 1=right) from MAT dict."""
    if label_key is None:
        warnings.warn(
            f"Subject {subject}: no label array found. Defaulting to 0. "
            "Expected keys: label, labels, y, condition, attention."
        )
        return np.zeros(n_trials, dtype=int)

    raw = np.asarray(mat[label_key]).ravel()
    return _coerce_binary_labels(raw, subject)


def _coerce_binary_labels(raw: np.ndarray, subject: int) -> np.ndarray:
    """Map whatever label encoding is used to 0=left, 1=right."""
    raw = raw.ravel()
    unique = np.unique(raw)

    # Already binary 0/1
    if set(unique).issubset({0, 1}):
        return raw.astype(int)

    # MATLAB 1/2 encoding → 0/1
    if set(unique).issubset({1, 2}):
        return (raw - 1).astype(int)

    # String labels: 'L'/'R', 'left'/'right', '1'/'2'
    try:
        str_labels = [str(v).strip().lower() for v in raw]
        left_kw = {"l", "left", "1", "la", "left_attention"}
        right_kw = {"r", "right", "2", "ra", "right_attention"}
        mapped = []
        for s in str_labels:
            if s in left_kw:
                mapped.append(0)
            elif s in right_kw:
                mapped.append(1)
            else:
                mapped.append(0)
                warnings.warn(f"Subject {subject}: unknown label '{s}', defaulting to 0.")
        return np.array(mapped, dtype=int)
    except Exception:
        pass

    # Generic: map sorted unique values to 0, 1, ...
    warnings.warn(
        f"Subject {subject}: label values {unique} — mapping smallest→0, next→1."
    )
    val_map = {v: i for i, v in enumerate(sorted(unique))}
    return np.array([val_map[v] for v in raw], dtype=int)


def _parse_label_strings(raw_lbls: List[str], subject: int) -> Optional[np.ndarray]:
    try:
        return _coerce_binary_labels(np.array(raw_lbls), subject)
    except Exception:
        return None


def _extract_ch_names(mat: dict, n_channels: int) -> List[str]:
    for key in ["ch_names", "chanlocs", "channel_names", "channels", "chan_names"]:
        if key in mat:
            try:
                raw = mat[key]
                if isinstance(raw, np.ndarray):
                    names = [str(v).strip() for v in raw.ravel().tolist()]
                    if len(names) == n_channels and all(names):
                        return names
            except Exception:
                pass
    return _fallback_ch_names(n_channels)


def _fallback_ch_names(n_channels: int) -> List[str]:
    if n_channels <= len(FALLBACK_CH_NAMES):
        return FALLBACK_CH_NAMES[:n_channels]
    return [f"EEG{i+1:03d}" for i in range(n_channels)]


def _pick_key(keys: set, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in keys:
            return c
    return None


# ---------------------------------------------------------------------------
# CNT fallback loader (raw files via MNE)
# ---------------------------------------------------------------------------

def _load_cnt_subject(path: Path, subject: int) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """Load raw Neuroscan CNT file using MNE, epoch by events, return arrays."""
    try:
        import mne
    except ImportError:
        raise RuntimeError(
            "MNE is required to load CNT files. Install with: pip install mne"
        )

    raw = mne.io.read_raw_cnt(str(path), preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    # Find events
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(events) == 0:
        raise ValueError(
            f"Subject {subject}: no events found in CNT file {path}. "
            "Cannot determine attention labels."
        )

    # Epoch: assume each event starts a 60s trial
    tmax = 59.99
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=0.0, tmax=tmax,
        baseline=None, preload=True, verbose="ERROR"
    )
    data = epochs.get_data().astype(np.float32)  # (n_trials, n_ch, n_times)
    labels = _coerce_binary_labels(epochs.events[:, 2], subject)

    return data, labels, sfreq, ch_names


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _window_trials(
    data: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    window_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Window each trial into non-overlapping windows of length window_s seconds.
    Each window inherits its parent trial's label.

    Parameters
    ----------
    data   : (n_trials, n_channels, n_times)
    labels : (n_trials,) or (n_trials, n_windows_per_trial)
    sfreq  : sampling rate
    window_s : window length in seconds

    Returns
    -------
    X_win : (n_windows, n_channels, win_samples)
    y_win : (n_windows,)
    """
    win_samples = int(round(sfreq * window_s))
    n_trials, n_channels, n_times = data.shape
    n_windows_per_trial = n_times // win_samples

    if n_windows_per_trial == 0:
        raise ValueError(
            f"Trial length {n_times} samples at {sfreq} Hz is shorter than "
            f"window length {win_samples} samples ({window_s}s). "
            "Check sampling rate and trial duration."
        )

    X_parts = []
    y_parts = []
    labels_arr = np.asarray(labels)
    dynamic_labels = labels_arr.ndim == 2
    if dynamic_labels and labels_arr.shape != (n_trials, n_windows_per_trial):
        raise ValueError(
            f"Dynamic labels shape {labels_arr.shape} does not match "
            f"(n_trials={n_trials}, n_windows={n_windows_per_trial})."
        )

    for t in range(n_trials):
        for w in range(n_windows_per_trial):
            start = w * win_samples
            end = start + win_samples
            X_parts.append(data[t, :, start:end])
            y_parts.append(labels_arr[t, w] if dynamic_labels else labels_arr[t])

    X_win = np.stack(X_parts, axis=0).astype(np.float32)
    y_win = np.array(y_parts, dtype=int)
    return X_win, y_win


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _resample(data: np.ndarray, sfreq_in: float, sfreq_out: float, ch_names: List[str]) -> np.ndarray:
    """Resample EEG data from sfreq_in to sfreq_out using MNE."""
    if abs(sfreq_in - sfreq_out) < 1.0:
        return data
    try:
        import mne
        n_trials, n_ch, n_times = data.shape
        info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq_in), ch_types="eeg")
        ep = mne.EpochsArray(data, info, verbose="ERROR")
        ep.resample(float(sfreq_out), npad="auto", verbose="ERROR")
        return ep.get_data().astype(np.float32)
    except ImportError:
        warnings.warn(
            "MNE not installed — skipping resampling. "
            f"Data will remain at {sfreq_in} Hz instead of {sfreq_out} Hz."
        )
        return data


# ---------------------------------------------------------------------------
# Main per-subject converter
# ---------------------------------------------------------------------------

def prepare_subject(
    subject: int,
    raw_root: Path,
    out_root: Path,
    target_sfreq: float = TARGET_SFREQ,
    window_s: float = 1.0,
) -> dict:
    """Convert one AASD subject to NPZ. Returns diagnostic info dict."""
    out_root.mkdir(parents=True, exist_ok=True)
    path = _find_subject_file(raw_root, subject)

    # Load data
    if path.suffix.lower() == ".cnt":
        data, labels, sfreq, ch_names = _load_cnt_subject(path, subject)
        eeg_new_event = None
    else:
        mat = _load_mat(path)
        eeg_new_event = (
            getattr(mat["EEG_new"], "event", None) if "EEG_new" in mat else None
        )
        data, labels, sfreq, ch_names = _extract_eeg_data(mat, subject)
        del mat
        gc.collect()

    n_trials_raw, n_channels, n_times_raw = data.shape

    # Validate: expect 60 trials of ~60s each
    expected_trials = 60
    if n_trials_raw != expected_trials:
        warnings.warn(
            f"Subject {subject}: expected {expected_trials} trials, got {n_trials_raw}. "
            "Proceeding anyway."
        )

    # Validate labels are binary
    unique_y = np.unique(labels)
    if len(unique_y) < 2:
        warnings.warn(
            f"Subject {subject}: only one label class found ({unique_y}). "
            "Check label extraction — this subject will be useless for classification."
        )

    # Resample to target_sfreq
    data = _resample(data, sfreq, target_sfreq, ch_names)
    sfreq_out = target_sfreq if abs(sfreq - target_sfreq) >= 1.0 else sfreq

    # For EEG_new files, prefer time-varying attention labels from recurrent
    # switch markers.  This matches the paper's Transformer target more closely
    # than assigning one label to an entire 60-second trial.
    if eeg_new_event is not None:
        dyn_labels = _extract_window_labels_from_eeg_new_event(
            eeg_new_event,
            n_trials=int(data.shape[0]),
            n_times=int(data.shape[2]),
            sfreq=float(sfreq_out),
            window_s=float(window_s),
            subject=subject,
        )
        if dyn_labels is not None:
            labels = dyn_labels

    # Window into 1s epochs
    X, y = _window_trials(data, labels, sfreq_out, window_s=window_s)
    del data
    gc.collect()
    unique_y = np.unique(y)

    # Save
    out_path = out_root / f"subject_{subject}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        sfreq=np.float32(sfreq_out),
        ch_names=np.array(ch_names, dtype=object),
    )

    n_windows, n_ch, n_win_samples = X.shape
    counts = {int(c): int(np.sum(y == c)) for c in unique_y}
    return {
        "subject": subject,
        "source_file": str(path),
        "n_trials_raw": n_trials_raw,
        "n_windows": n_windows,
        "n_channels": n_ch,
        "win_samples": n_win_samples,
        "sfreq": sfreq_out,
        "class_counts": counts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AASD processed MAT files to 1-second windowed NPZ."
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", required=True,
        help="Subject IDs to convert (e.g. --subjects 1 2 3)."
    )
    parser.add_argument(
        "--raw-root", type=str, default="data/aasd/processed",
        help="Folder containing AASD processed MAT (or CNT) files."
    )
    parser.add_argument(
        "--out-root", type=str, default="data/aasd/npz",
        help="Output folder for subject_N.npz files."
    )
    parser.add_argument(
        "--target-sfreq", type=float, default=TARGET_SFREQ,
        help=f"Target sampling rate after resampling (default: {TARGET_SFREQ} Hz)."
    )
    parser.add_argument(
        "--window-s", type=float, default=1.0,
        help="Window length in seconds (default: 1.0)."
    )
    args = parser.parse_args()

    raw_root = _resolve_raw_root(Path(args.raw_root))
    out_root = Path(args.out_root)

    print(f"AASD converter: {len(args.subjects)} subject(s) | "
          f"sfreq={args.target_sfreq} Hz | window={args.window_s}s")

    for subj in args.subjects:
        print(f"  Subject {subj}...", end=" ", flush=True)
        try:
            info = prepare_subject(
                subject=subj,
                raw_root=raw_root,
                out_root=out_root,
                target_sfreq=args.target_sfreq,
                window_s=args.window_s,
            )
            counts_str = " | ".join(
                f"class{c}={n}" for c, n in sorted(info["class_counts"].items())
            )
            print(
                f"OK — {info['n_windows']} windows "
                f"({info['n_channels']}ch × {info['win_samples']}samples @ {info['sfreq']}Hz) "
                f"| {counts_str}"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")


if __name__ == "__main__":
    main()
