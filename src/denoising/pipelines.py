from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import welch

# ICA+ICALabel: preprocessing and rejection.
ICA_ICALABEL_LOW = 1.0   # Hz — ICLabel expects 1–100 Hz (see MNE/ICLabel docs).
ICA_ICALABEL_HIGH = 100.0
ICA_N_COMPONENTS = 32
ICA_RANDOM_STATE = 97
# Reject non-brain components with probability above this (keep Brain/Other).
ICALABEL_ARTIFACT_THRESHOLD = 0.90
# Artifact label substrings to reject (ICLabel: Brain, Muscle, Eye blink, Heart, Line noise, Channel noise, Other).
ICALABEL_ARTIFACT_LABELS = ("eye", "muscle", "heart", "line noise", "channel noise")
# Optional notch (Hz); None = no notch. Use (50,) or (60,) for line noise.
ICA_NOTCH_HZ = (60.0,)  # set to None to disable
# Over-removal guard: if alpha retention ratio < this, log and revert (no silent fallback).
ICALABEL_RETENTION_HARD_MIN = 0.0
MOTOR_CHANNEL_CANDIDATES = ("C3", "C4", "CZ")
# MRCP-specific motor subset per Reyes-Jiménez et al. (Data in Brief 65, 2026), §4.5/Fig. 5.
MRCP_MOTOR_CHANNELS = ("FC3", "FC1", "FCZ", "C3", "C1", "CZ", "CP3", "CP1", "CPZ")
# Set EEG_ICA_VERBOSE=1 to print rejected components and max signal difference.
# Bands for brain-signal preservation checks (alpha/mu, beta).
ALPHA_BAND = (8.0, 12.0)
BETA_BAND = (13.0, 30.0)

# MRCP GEDAI configuration (FAQ-compliant for sub-1 Hz paradigms).
# Upsample to 250 Hz so an 8-level haar wavelet isolates the MRCP band cleanly:
# approx=(0, 0.49 Hz), detail-1=(0.49, 0.98 Hz) — together they span the 0.1–1 Hz MRCP band
# defined by Reyes-Jiménez et al. (Data in Brief 65, 2026).
MRCP_GEDAI_RESAMPLE_HZ = 250.0
MRCP_GEDAI_HPF_HZ = 0.05   # preserve 0.1–1 Hz MRCP; only remove DC drift.
MRCP_GEDAI_WAVELET_LEVEL = 8
MRCP_GEDAI_NOISE_MULTIPLIER = 4.0   # FAQ "auto" equivalent; band-adaptive mode benefits from milder setting.
MRCP_GEDAI_DURATION_S = 8.0
MRCP_GEDAI_OVERLAP = 0.5
MRCP_RETENTION_MIN = 0.5   # fall back to paper baseline if GEDAI destroys > 50% of 0.1–1 Hz motor power.
_MRCP_PREPARED_CACHE: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = {}


def _butter_bandpass(
    l_freq: float, h_freq: float, sfreq: float, order: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(
    X: np.ndarray, sfreq: float, l_freq: float, h_freq: float, chunk_size: int = 5000
) -> np.ndarray:
    """Bandpass filter trials. Modifies X in-place to save RAM on 16GB machines."""
    X = np.asarray(X, dtype=np.float32)
    b, a = _butter_bandpass(l_freq, h_freq, sfreq)
    
    n_trials = X.shape[0]
    # Process in chunks to avoid float64 upcasting spikes
    for i in range(0, n_trials, chunk_size):
        end = min(i + chunk_size, n_trials)
        # Modify X in-place
        X[i:end] = filtfilt(b, a, X[i:end], axis=-1).astype(np.float32, copy=False)
    
    return X


def _median_bandpower_ratio(
    x_clean: np.ndarray,
    x_ref: np.ndarray,
    sfreq: float,
    lo: float,
    hi: float,
    chunk_size: int = 10000,
) -> float:
    """Median clean/reference band-power ratio across trials and channels."""
    n_trials = x_clean.shape[0]
    ratios: List[np.ndarray] = []
    nperseg = int(min(256, x_clean.shape[-1]))
    
    for i in range(0, n_trials, chunk_size):
        end = min(i + chunk_size, n_trials)
        # Process in chunks: welch creates internal float64 copies
        f, p_clean = welch(x_clean[i:end], fs=sfreq, nperseg=nperseg, axis=-1)
        _, p_ref = welch(x_ref[i:end], fs=sfreq, nperseg=nperseg, axis=-1)
        
        m = (f >= lo) & (f <= hi)
        if not np.any(m):
            continue
            
        try:
            # np.trapezoid (Numpy 2.0+) or np.trapz
            bp_clean = np.trapezoid(p_clean[..., m], f[m], axis=-1)
            bp_ref = np.trapezoid(p_ref[..., m], f[m], axis=-1)
        except AttributeError:
            bp_clean = np.trapz(p_clean[..., m], f[m], axis=-1)
            bp_ref = np.trapz(p_ref[..., m], f[m], axis=-1)
            
        chunk_ratio = np.divide(
            bp_clean,
            np.maximum(bp_ref, 1e-20),
            out=np.ones_like(bp_clean, dtype=np.float32),
            where=bp_ref > 1e-20,
        )
        ratios.append(chunk_ratio.ravel())
        del p_clean, p_ref, bp_clean, bp_ref
        
    if not ratios:
        return 1.0
    return float(np.median(np.concatenate(ratios)))


def _retention_ratios(
    x_clean: np.ndarray,
    x_band_ref: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    ch_names: List[str],
) -> float:
    """Effective retention ratio for the relevant physiological bands.

    For classic MI-style decoding (e.g. 8-30 Hz), we protect full-band, motor-band,
    alpha/mu, and beta power. For low-frequency paradigms such as MRCP
    (e.g. 0.1-1 Hz), alpha/beta checks are not meaningful because the decode band
    does not include them, so we only guard the decode band and motor channels.
    """
    full = _median_bandpower_ratio(x_clean, x_band_ref, sfreq, l_freq, h_freq)
    motor_idx = _select_channel_idx(ch_names, MOTOR_CHANNEL_CANDIDATES)
    motor = full
    if motor_idx.size > 0:
        motor = _median_bandpower_ratio(
            x_clean[:, motor_idx, :],
            x_band_ref[:, motor_idx, :],
            sfreq,
            l_freq,
            h_freq,
        )
    ratios = [full, motor]
    if h_freq >= ALPHA_BAND[0]:
        ratios.append(
            _median_bandpower_ratio(
                x_clean, x_band_ref, sfreq, ALPHA_BAND[0], ALPHA_BAND[1]
            )
        )
    if h_freq >= BETA_BAND[0]:
        ratios.append(
            _median_bandpower_ratio(
                x_clean, x_band_ref, sfreq, BETA_BAND[0], BETA_BAND[1]
            )
        )
    return float(min(ratios))


def _select_channel_idx(ch_names: List[str], wanted: Tuple[str, ...]) -> np.ndarray:
    idx = []
    c_upper = [str(c).upper() for c in ch_names]
    wanted_u = {w.upper() for w in wanted}
    for i, c in enumerate(c_upper):
        if c in wanted_u:
            idx.append(i)
    return np.asarray(idx, dtype=int)


def apply_icalabel(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
) -> np.ndarray:
    """Production ICA + ICALabel: 1–100 Hz preprocess, fit, label, reject artifacts, apply.

    - Bandpass 1–100 Hz before ICA/ICLabel (required by ICLabel).
    - Rejects components labeled eye/muscle/heart/line_noise/channel_noise with prob > threshold.
    - Applies ICA to data and returns actually modified cleaned data (then bandpassed to l_freq–h_freq).
    - Optional over-removal guard: if alpha retention < ICALABEL_RETENTION_HARD_MIN, reverts and logs.
    - Set EEG_ICA_VERBOSE=1 for verification prints (rejected components, max difference).
    """
    try:
        import mne
        from mne.preprocessing import ICA
        from mne_icalabel import label_components
    except Exception as exc:
        raise RuntimeError(
            "ICALabel pipeline requires `mne` and `mne-icalabel` to be installed."
        ) from exc

    _verbose = os.environ.get("EEG_ICA_VERBOSE", "0").strip() == "1"
    X = np.asarray(X, dtype=np.float32)
    n_trials, n_channels, n_times = X.shape
    x_band_ref = bandpass_filter(X.copy(), sfreq, l_freq, h_freq).astype(np.float32, copy=False)

    # ---- 1) Build Epochs, set reference, montage ----
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types="eeg")
    epochs_orig = mne.EpochsArray(X, info, verbose="ERROR")
    epochs_orig.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")

    # Drop channels with no montage position (e.g. Weibo2014's CB1 cerebellum channel).
    # ICALabel needs topomap positions for ALL channels — missing ones crash it.
    # We track dropped indices so we can re-insert zero-padded channels after cleaning.
    no_pos_chs = [ch['ch_name'] for ch in epochs_orig.info['chs'] if np.isnan(ch['loc'][:3]).any()]
    no_pos_idx = [epochs_orig.ch_names.index(c) for c in no_pos_chs]
    if no_pos_chs:
        if len(no_pos_chs) == len(ch_names):
            if _verbose:
                print("  [ICALabel] all channels lack montage pos; proceeding without dropping.")
            no_pos_chs = []
            no_pos_idx = []
        else:
            if _verbose:
                print(f"  [ICALabel] dropping {len(no_pos_chs)} channels with no montage pos: {no_pos_chs}")
            epochs_orig.drop_channels(no_pos_chs)


    epochs_orig.set_eeg_reference("average", projection=False, verbose="ERROR")

    # ---- 2) Optional notch 50/60 Hz ----
    # Notch skipped for epochs to prevent ringing; already bandpassed afterwards.

    try:
        # ---- 3) 1–100 Hz bandpass for ICA/ICLabel (required; removes warning) ----
        nyq = 0.5 * float(sfreq)
        ica_high = min(ICA_ICALABEL_HIGH, nyq - 0.5)
        epochs_for_ica = epochs_orig.copy()
        epochs_for_ica.filter(l_freq=ICA_ICALABEL_LOW, h_freq=ica_high, n_jobs=1, verbose="ERROR")

        # ---- 4) Fit ICA (extended infomax), rank = n_channels ----
        n_comp = min(n_channels - 1, ICA_N_COMPONENTS)
        ica = ICA(
            n_components=n_comp,
            random_state=ICA_RANDOM_STATE,
            max_iter="auto",
            method="infomax",
            fit_params={"extended": True},
        )
        ica.fit(epochs_for_ica, verbose="ERROR")

        # ---- 5) ICALabel: same 1–100 Hz Epochs ----
        labels_dict = label_components(epochs_for_ica, ica, method="iclabel")
        component_labels = labels_dict.get("labels", [])
        probs = labels_dict.get("y_pred_proba", None)

        # ---- 6) Reject non-brain: eye, muscle, heart, line_noise, channel_noise with prob > threshold ----
        exclude_idx = []
        for idx, lab in enumerate(component_labels):
            lab_l = str(lab).lower().strip()
            if lab_l in ("brain", "other"):
                continue
            if not any(art in lab_l for art in ICALABEL_ARTIFACT_LABELS):
                continue
            conf = 1.0
            if probs is not None and idx < len(probs):
                try:
                    conf = float(np.max(np.asarray(probs[idx]).ravel()))
                except Exception:
                    pass
            if conf >= ICALABEL_ARTIFACT_THRESHOLD:
                exclude_idx.append((idx, conf, lab_l))

        exclude_idx.sort(key=lambda t: t[1], reverse=True)
        ica.exclude = [t[0] for t in exclude_idx]

        if _verbose or len(ica.exclude) == 0:
            if len(ica.exclude) == 0:
                import warnings
                warnings.warn(
                    "ICA+ICALabel: no components excluded (ica.exclude is empty). "
                    "Signal will be unchanged. Check labels/threshold."
                )
            if _verbose:
                print("[ICA] Rejected components (idx, prob, label):", ica.exclude)
                for t in exclude_idx:
                    print(f"  IC{t[0]}: prob={t[1]:.3f} label={t[2]!r}")
                print("[ICA] Number rejected:", len(ica.exclude))

        # ---- 7) Apply ICA to original epochs (modifies signal) ----
        epochs_clean = epochs_orig.copy()
        ica.apply(epochs_clean, verbose="ERROR")

        # ---- 8) Verification: max difference ----
        data_orig = epochs_orig.get_data()
        data_clean = epochs_clean.get_data()
        # Use float32 for difference check to save 50% memory on large trial sets
        max_diff = float(np.max(np.abs(data_clean.astype(np.float32) - data_orig.astype(np.float32))))
        if _verbose:
            print("[ICA] Max |clean - orig|:", max_diff)

        # ---- 9) Use clean data array, bandpass to decode band ----
        x_clean = data_clean # use the array we already extracted
        
        # Aggressive cleanup of heavy objects (~5-10GB RAM)
        del epochs_orig, epochs_clean, ica, data_orig, data_clean
        import gc
        gc.collect()

    except Exception as e:
        import warnings
        warnings.warn(f"ICA+ICALabel pipeline failed ({e}); falling back to bandpass baseline.")
        return x_band_ref.astype(np.float32, copy=False)

    # Re-insert dropped channels as zeros to match original input shape
    if no_pos_idx:
        x_full = np.zeros((x_clean.shape[0], len(ch_names), x_clean.shape[2]), dtype=x_clean.dtype)
        kept_idx = [i for i in range(len(ch_names)) if i not in no_pos_idx]
        x_full[:, kept_idx, :] = x_clean
        x_clean = x_full

    x_clean = bandpass_filter(x_clean, sfreq, l_freq, h_freq).astype(np.float32, copy=False)

    # ---- 10) Over-removal guard: log and revert if alpha drops too much (no silent fallback) ----
    # Evaluate only on the channels kept in the ICA cleaning to avoid zero-division
    try:
        eval_idx = [i for i in range(len(ch_names)) if i not in no_pos_idx] if no_pos_idx else slice(None)
        alpha_ratio = _median_bandpower_ratio(
            x_clean[:, eval_idx, :], x_band_ref[:, eval_idx, :], sfreq, ALPHA_BAND[0], ALPHA_BAND[1]
        )
        if alpha_ratio < ICALABEL_RETENTION_HARD_MIN:
            import warnings
            warnings.warn(
                f"ICA+ICALabel: alpha retention ratio {alpha_ratio:.3f} < {ICALABEL_RETENTION_HARD_MIN}. "
                "Reverting to bandpass (no ICA) to avoid over-removal. This is explicitly logged."
            )
            return x_band_ref.astype(np.float32, copy=False)
    except Exception as e:
        import warnings
        warnings.warn(f"ICA+ICALabel post-processing failed ({e}); falling back to bandpass.")
        return x_band_ref.astype(np.float32, copy=False)

    return x_clean.astype(np.float32, copy=False)


# Default GEDAI threads when caller does not pass gedai_n_jobs (16 GB laptops).
_DEFAULT_GEDAI_N_JOBS = 1
# Step-1 broadband: conservative (6.0) to remove only large transient artifacts.
GEDAI_BROADBAND_NOISE_MULTIPLIER = 6.0
# Step-2 spectral: set to 6.0 (per professor's recommendation — equivalent to MATLAB 'auto-' setting).
# Higher value = less aggressive = preserves more brain signal on already-clean PhysioNet data.
GEDAI_SPECTRAL_NOISE_MULTIPLIER = 6.0
GEDAI_SPECTRAL_WAVELET_LEVEL = 5   # number of frequency bands for spectral decomp
GEDAI_SPECTRAL_LOW_CUTOFF = 2      # Hz — exclude lowest freqs (avoid epoch-length artefacts)
# Retention guard.
# If GEDAI removes too much signal in the decode/motor/alpha/beta bands, retry
# with a more conservative setting and eventually fall back to baseline.
GEDAI_RETENTION_MEDIAN_MIN = 0.75
GEDAI_RETENTION_HARD_MIN = 0.60


def _gedai_leadfield_ch_names(gedai_lib_path: str, n_channels: int) -> List[str]:
    """Return first n_channels names from GEDAI leadfield .mat for generic-name fallback (e.g. Cho2017 EEG01..)."""
    import h5py
    from pathlib import Path
    root = Path(gedai_lib_path).resolve()
    mat_path = root / "gedai" / "data" / "fsavLEADFIELD_4_GEDAI.mat"
    if not mat_path.exists():
        return []
    with h5py.File(mat_path, "r") as f:
        leadfield_data = f["leadfield4GEDAI"]
        leadfield_channel_data = leadfield_data["electrodes"]
        names = [
            f[ref[0]][()].tobytes().decode("utf-16le").lower()
            for ref in leadfield_channel_data["Name"]
        ]
    return names[:n_channels] if len(names) >= n_channels else []


def apply_asr(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
) -> np.ndarray:
    """Apply ASR (Artifact Subspace Reconstruction) using asrpy, then bandpass to decode band.

    Operates on concatenated trials by treating them as one continuous Raw.
    Falls back to bandpass-only baseline if ASR is unavailable.
    """
    try:
        import mne
        import asrpy
    except Exception as exc:  # pragma: no cover - environment-dependent
        import warnings
        warnings.warn(
            f"ASR pipeline requires `mne` and `asrpy`; falling back to bandpass-only ({exc})."
        )
        return bandpass_filter(X, sfreq, l_freq, h_freq).astype(np.float32, copy=False)

    X = np.asarray(X, dtype=np.float32)
    n_trials, n_channels, n_times = X.shape

    info = mne.create_info(ch_names=list(ch_names), sfreq=float(sfreq), ch_types="eeg")
    # Concatenate trials along time to approximate continuous data
    X_concat = X.transpose(1, 0, 2).reshape(n_channels, n_trials * n_times)
    raw = mne.io.RawArray(X_concat, info, verbose="ERROR")

    # Standard ASR cutoff; this is the main hyperparameter users may want to tune.
    asr = asrpy.ASR(sfreq=float(sfreq), cutoff=20.0)
    asr.fit(raw)
    raw_clean = asr.transform(raw)

    X_clean_concat = raw_clean.get_data().astype(np.float32, copy=False)
    # Reshape back to (trials, channels, times)
    X_clean = (
        X_clean_concat.reshape(n_channels, n_trials, n_times)
        .transpose(1, 0, 2)
        .astype(np.float32, copy=False)
    )
    return bandpass_filter(X_clean, sfreq, l_freq, h_freq).astype(np.float32, copy=False)


def _run_pylossless_on_continuous_raw(raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
    """Default PyLossless config and default ``RejectionPolicy`` (public pipeline API)."""
    from pylossless import Config, LosslessPipeline

    try:
        from pylossless import RejectionPolicy
    except ImportError:
        from pylossless.config import RejectionPolicy

    cfg = Config().load_default()
    pipeline = LosslessPipeline(config=cfg)
    pipeline.run_with_raw(raw.copy())
    return RejectionPolicy().apply(pipeline, version_mismatch="warning")


def apply_pylossless(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
) -> np.ndarray:
    """PyLossless with trials concatenated into one continuous ``Raw`` (same idea as ASR).

    For Alljoined, prefer ``<data_root>/pylossless_manifest/subject_<id>.txt`` so each EDF
    block is cleaned continuously before epoching (see ``apply_pylossless_alljoined_via_manifest``).
    """
    try:
        import mne
        import pylossless  # noqa: F401 — presence check
    except Exception as exc:  # pragma: no cover
        import warnings
        warnings.warn(
            f"PyLossless requires `mne` and `pylossless` ({exc}); using bandpass-only."
        )
        return bandpass_filter(X, sfreq, l_freq, h_freq).astype(np.float32, copy=False)

    X = np.asarray(X, dtype=np.float32)
    n_trials, n_channels, n_times = X.shape
    info = mne.create_info(ch_names=list(ch_names), sfreq=float(sfreq), ch_types="eeg")
    X_concat = X.transpose(1, 0, 2).reshape(n_channels, n_trials * n_times)
    raw = mne.io.RawArray(X_concat, info, verbose="ERROR")
    try:
        raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    except Exception:
        pass

    try:
        raw_clean = _run_pylossless_on_continuous_raw(raw)
        X_clean_concat = raw_clean.get_data().astype(np.float32, copy=False)
        X_clean = (
            X_clean_concat.reshape(n_channels, n_trials, n_times)
            .transpose(1, 0, 2)
            .astype(np.float32, copy=False)
        )
    except Exception as e:
        import warnings
        warnings.warn(f"PyLossless failed ({e}); falling back to bandpass-only.")
        return bandpass_filter(X, sfreq, l_freq, h_freq).astype(np.float32, copy=False)

    return bandpass_filter(X_clean, sfreq, l_freq, h_freq).astype(np.float32, copy=False)


def _read_alljoined_pylossless_manifest(
    data_root: Path, subject_id: int
) -> Optional[Tuple[str, List[str]]]:
    """Parse ``<data_root>/pylossless_manifest/subject_<id>.txt``: line0=meta parquet, rest=EDFs."""
    manifest = Path(data_root) / "pylossless_manifest" / f"subject_{subject_id}.txt"
    if not manifest.is_file():
        return None
    lines = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    base = manifest.parent

    def resolve_one(s: str) -> str:
        q = Path(s).expanduser()
        return str(q.resolve() if q.is_absolute() else (base / q).resolve())

    return resolve_one(lines[0]), [resolve_one(x) for x in lines[1:]]


def apply_pylossless_alljoined_via_manifest(
    subject_id: int,
    data_root: Path,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    sfreq: float,
    n_trials_expected: int,
) -> Optional[np.ndarray]:
    """PyLossless on each EDF block, then the same pulse-epoching as ``prepare_alljoined``."""
    parsed = _read_alljoined_pylossless_manifest(data_root, subject_id)
    if parsed is None:
        return None
    meta_path, edf_paths = parsed

    from ..data.prepare_alljoined import epoch_alljoined_trials_from_edfs

    def raw_hook(raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
        return _run_pylossless_on_continuous_raw(raw)

    try:
        all_X, _all_y = epoch_alljoined_trials_from_edfs(
            edf_paths,
            meta_path,
            list(ch_names),
            target_sfreq=float(sfreq),
            t_epoch=(0.0, 1.0),
            raw_hook=raw_hook,
            log_prefix="    [PyLossless] ",
        )
    except Exception as e:
        import warnings
        warnings.warn(
            f"PyLossless Alljoined manifest path failed ({e}); falling back to concat PyLossless."
        )
        return None

    if not all_X:
        return None
    X = np.stack(all_X).astype(np.float32, copy=False)
    if X.shape[0] != n_trials_expected:
        import warnings
        warnings.warn(
            f"PyLossless manifest produced {X.shape[0]} trials but NPZ has "
            f"{n_trials_expected}; falling back to concat PyLossless."
        )
        return None
    return bandpass_filter(X, sfreq, l_freq, h_freq).astype(np.float32, copy=False)


def apply_gedai(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    gedai_n_jobs: int | None = None,
) -> np.ndarray:
    """Apply GEDAI denoising using the official two-step spectral pipeline.

    Recommended pipeline (from GEDAI spectral tutorial):
      Step 1 — Broadband GEDAI (noise_multiplier=6.0): conservative pass to
               remove large transient artefacts (eye blinks, muscle bursts).
      Step 2 — Spectral GEDAI (wavelet_level=5): frequency-band-specific
               denoising on the broadband-cleaned signal.

    Uses float32. Pass ``gedai_n_jobs`` from ``ExperimentConfig.memory.n_jobs`` on
    workstations; omit or set 1 for low-RAM machines.
    Electrode montage is set on the MNE info so GEDAI's SENSAI algorithm has
    proper spatial coordinates for the leadfield computation.
    """
    n_jobs = int(gedai_n_jobs) if gedai_n_jobs is not None else _DEFAULT_GEDAI_N_JOBS
    n_jobs = max(1, n_jobs)

    if os.environ.get("PYGEDAI_FORCE_CPU", "").strip() == "1":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import mne
        # Dynamic import to support switching between Bypass and Threshold engines
        gedai_lib_path = os.environ.get("GEDAI_LIBRARY_PATH", "").strip()
        if gedai_lib_path:
            import sys
            from pathlib import Path
            lib_abs_path = str(Path(gedai_lib_path).resolve())
            if lib_abs_path not in sys.path:
                sys.path.insert(0, lib_abs_path)
            # Force reload if it was already imported from elsewhere
            import importlib
            if 'gedai' in sys.modules:
                importlib.reload(sys.modules['gedai'])
        
        from gedai import Gedai
    except ImportError:  # pragma: no cover
        import warnings
        warnings.warn(
            "GEDAI pipeline requires `gedai` (and mne); not found. Using identity."
        )
        return bandpass_filter(X, sfreq, l_freq, h_freq).astype(np.float32)

    X = np.asarray(X, dtype=np.float32)
    x_band_ref = bandpass_filter(X.copy(), sfreq, l_freq, h_freq).astype(np.float32, copy=False)
    n_trials, n_channels, n_times = X.shape

    # Use standard leadfield names when data has generic names
    ch_names_use = list(ch_names)
    gedai_lib_path = os.environ.get("GEDAI_LIBRARY_PATH", "").strip()
    if gedai_lib_path and n_channels == 64:
        generic = all(len(c) >= 4 and c.upper().startswith("EEG") and c[3:].isdigit() for c in ch_names)
        if generic:
            try:
                leadfield_names = _gedai_leadfield_ch_names(gedai_lib_path, n_channels)
                if leadfield_names:
                    ch_names_use = leadfield_names
            except Exception:
                pass

    info = mne.create_info(ch_names=ch_names_use, sfreq=sfreq, ch_types="eeg")
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        info.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception:
        pass

    no_pos_chs = [ch['ch_name'] for ch in info['chs'] if np.isnan(ch['loc'][:3]).any()]
    no_pos_idx = [info.ch_names.index(c) for c in no_pos_chs]
    
    if len(no_pos_chs) == len(ch_names):
        kept_idx = list(range(len(ch_names)))
    else:
        kept_idx = [i for i in range(len(ch_names)) if i not in no_pos_idx]

    # MEMORY OPTIMIZATION: Work on a view if possible to save 2.5GB
    if len(kept_idx) == X.shape[1]:
        X_kept = X
    else:
        X_kept = X[:, kept_idx, :]
    info_kept = mne.pick_info(info, kept_idx)
    n_kept_channels = len(kept_idx)
    
    import gc
    # Only keep X_kept
    X_concat = X_kept.transpose(1, 0, 2).reshape(n_kept_channels, n_trials * n_times).astype(np.float32)
    # We can't delete X yet because it's owned by the caller (preprocess_subject_data)
    # but we can at least let the caller know it can be deleted.
    
    raw = mne.io.RawArray(X_concat, info_kept, verbose="ERROR")
    del X_concat
    gc.collect()

    raw.set_eeg_reference("average", projection=False, verbose=False)

    try:
        # Step 1: Broadband GEDAI
        gedai_broad = Gedai(wavelet_type="haar", wavelet_level=0)
        gedai_broad.fit_raw(raw, noise_multiplier=GEDAI_BROADBAND_NOISE_MULTIPLIER, n_jobs=n_jobs, verbose=False)
        raw_broad_clean = gedai_broad.transform_raw(raw, n_jobs=n_jobs, overlap=0.5, verbose=False)
        
        # Cleanup input raw to free 2.5GB
        del raw
        gc.collect()

        # Step 2: Spectral GEDAI
        gedai_spectral = Gedai(wavelet_type="haar", wavelet_level=GEDAI_SPECTRAL_WAVELET_LEVEL, wavelet_low_cutoff=GEDAI_SPECTRAL_LOW_CUTOFF)
        gedai_spectral.fit_raw(raw_broad_clean, noise_multiplier=GEDAI_SPECTRAL_NOISE_MULTIPLIER, n_jobs=n_jobs, verbose=False)
        raw_clean = gedai_spectral.transform_raw(raw_broad_clean, n_jobs=n_jobs, overlap=0.5, verbose=False)
        
        del raw_broad_clean
        gc.collect()

    except Exception as e:
        import warnings
        warnings.warn(f"GEDAI two-step failed ({e}); falling back to baseline.")
        # Re-apply bandpass on the fly to save memory
        return bandpass_filter(X_kept, sfreq, l_freq, h_freq).astype(np.float32)

    X_clean_kept = raw_clean.get_data().reshape(n_kept_channels, n_trials, n_times).transpose(1, 0, 2).astype(np.float32)
    del raw_clean
    gc.collect()

    if no_pos_idx:
        X_clean = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
        X_clean[:, kept_idx, :] = X_clean_kept
        del X_clean_kept
    else:
        X_clean = X_clean_kept
    
    x_clean = bandpass_filter(X_clean, sfreq, l_freq, h_freq).astype(np.float32, copy=False)

    # Retention guard: GEDAI should not remove excessive decode-band, alpha, beta,
    # or motor-channel power. If it does, retry with a more conservative spectral
    # pass, then fall back to baseline if necessary.
    try:
        effective_ratio = _retention_ratios(
            x_clean=x_clean,
            x_band_ref=x_band_ref,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            ch_names=ch_names_use,
        )
        if effective_ratio < GEDAI_RETENTION_MEDIAN_MIN:
            import warnings
            warnings.warn(
                f"GEDAI retention ratio {effective_ratio:.3f} < {GEDAI_RETENTION_MEDIAN_MIN:.2f}; "
                "retrying with a more conservative spectral pass."
            )

            # Conservative retry: keep broadband pass, but relax the spectral pass.
            raw_retry_in = mne.io.RawArray(
                X_kept.transpose(1, 0, 2).reshape(n_kept_channels, n_trials * n_times).astype(np.float32),
                info_kept,
                verbose="ERROR",
            )
            raw_retry_in.set_eeg_reference("average", projection=False, verbose=False)
            g_broad_retry = Gedai(wavelet_type="haar", wavelet_level=0)
            g_broad_retry.fit_raw(
                raw_retry_in,
                noise_multiplier=max(GEDAI_BROADBAND_NOISE_MULTIPLIER, 8.0),
                n_jobs=n_jobs,
                verbose=False,
            )
            raw_broad_retry = g_broad_retry.transform_raw(
                raw_retry_in, n_jobs=n_jobs, overlap=0.5, verbose=False
            )
            g_spec_retry = Gedai(
                wavelet_type="haar",
                wavelet_level=max(3, GEDAI_SPECTRAL_WAVELET_LEVEL - 1),
                wavelet_low_cutoff=GEDAI_SPECTRAL_LOW_CUTOFF,
            )
            g_spec_retry.fit_raw(
                raw_broad_retry,
                noise_multiplier=max(GEDAI_SPECTRAL_NOISE_MULTIPLIER, 8.0),
                n_jobs=n_jobs,
                verbose=False,
            )
            raw_retry_out = g_spec_retry.transform_raw(
                raw_broad_retry, n_jobs=n_jobs, overlap=0.5, verbose=False
            )
            x_retry_kept = raw_retry_out.get_data().reshape(
                n_kept_channels, n_trials, n_times
            ).transpose(1, 0, 2).astype(np.float32)
            if no_pos_idx:
                x_retry = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
                x_retry[:, kept_idx, :] = x_retry_kept
            else:
                x_retry = x_retry_kept
            x_retry = bandpass_filter(x_retry, sfreq, l_freq, h_freq).astype(np.float32, copy=False)
            retry_ratio = _retention_ratios(
                x_clean=x_retry,
                x_band_ref=x_band_ref,
                sfreq=sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                ch_names=ch_names_use,
            )
            if retry_ratio >= GEDAI_RETENTION_MEDIAN_MIN:
                return x_retry
            if retry_ratio < GEDAI_RETENTION_HARD_MIN:
                warnings.warn(
                    f"GEDAI retry retention ratio {retry_ratio:.3f} < {GEDAI_RETENTION_HARD_MIN:.2f}; "
                    "reverting to bandpass baseline to avoid over-removal."
                )
                return x_band_ref
            return x_retry
    except Exception as e:
        import warnings
        warnings.warn(f"GEDAI retention guard failed ({e}); returning current GEDAI output.")

    return x_clean



def apply_gedai_from_continuous_raw(
    raw: "mne.io.BaseRaw",  # full continuous session, already loaded with montage
    gedai_n_jobs: int | None = None,
    *,
    use_spectral_step: bool = True,
    broadband_noise_multiplier: float = GEDAI_BROADBAND_NOISE_MULTIPLIER,
    spectral_noise_multiplier: float = GEDAI_SPECTRAL_NOISE_MULTIPLIER,
    spectral_wavelet_level: int = GEDAI_SPECTRAL_WAVELET_LEVEL,
    spectral_low_cutoff: float = GEDAI_SPECTRAL_LOW_CUTOFF,
) -> "mne.io.BaseRaw":
    """Apply the two-step spectral GEDAI pipeline to a continuous raw recording.

    This is the **correct** way to use GEDAI:
    - Feed it a multi-minute continuous session (not concatenated short epochs)
    - GEDAI's SENSAI covariance estimation works on long recordings to distinguish
      stable brain rhythms from transient artifacts

    Returns the cleaned MNE Raw object (same structure as input).
    """
    n_jobs = int(gedai_n_jobs) if gedai_n_jobs is not None else _DEFAULT_GEDAI_N_JOBS
    n_jobs = max(1, n_jobs)

    if os.environ.get("PYGEDAI_FORCE_CPU", "").strip() == "1":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        # Dynamic import to support switching between Bypass and Threshold engines
        gedai_lib_path = os.environ.get("GEDAI_LIBRARY_PATH", "").strip()
        if gedai_lib_path:
            import sys
            from pathlib import Path
            lib_abs_path = str(Path(gedai_lib_path).resolve())
            if lib_abs_path not in sys.path:
                sys.path.insert(0, lib_abs_path)
            # Force reload if it was already imported from elsewhere
            import importlib
            if 'gedai' in sys.modules:
                importlib.reload(sys.modules['gedai'])

        from gedai import Gedai
    except ImportError:
        import warnings
        warnings.warn("gedai not installed; returning raw unchanged.")
        return raw

    # GEDAI python implementation requires explicit average referencing.
    raw = raw.copy()
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    try:
        # Step 1: conservative broadband pass (removes large transients)
        g_broad = Gedai(wavelet_type="haar", wavelet_level=0)
        g_broad.fit_raw(
            raw,
            noise_multiplier=broadband_noise_multiplier,
            n_jobs=n_jobs,
            verbose=False,
        )
        raw_broad = g_broad.transform_raw(raw, n_jobs=n_jobs, verbose=False)

        if not use_spectral_step:
            return raw_broad

        # Step 2: frequency-specific spectral denoising
        g_spec = Gedai(
            wavelet_type="haar",
            wavelet_level=spectral_wavelet_level,
            wavelet_low_cutoff=spectral_low_cutoff,
        )
        g_spec.fit_raw(
            raw_broad,
            noise_multiplier=spectral_noise_multiplier,
            n_jobs=n_jobs,
            verbose=False,
        )
        raw_clean = g_spec.transform_raw(raw_broad, n_jobs=n_jobs, verbose=False)
        
        # GEDAI strips annotations when creating new RawArrays; restore them for epoching
        raw_clean.set_annotations(raw.annotations)
        return raw_clean

    except Exception as e:
        import warnings
        warnings.warn(f"GEDAI continuous pipeline failed ({e}); returning raw unchanged.")
        return raw


def load_continuous_raw_physionet(
    subject: int,
    runs: List[int] = None,
) -> "mne.io.BaseRaw":
    """Load the full continuous PhysioNet EDF session for a subject.

    Downloads via MNE if not cached locally. Sets standard_1020 montage.
    Returns concatenated Raw across all requested runs.
    """
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf, concatenate_raws

    if runs is None:
        runs = [6, 10, 14]  # hands vs feet MI runs

    raw_fnames = eegbci.load_data(subjects=subject, runs=runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose="ERROR") for f in raw_fnames]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    try:
        raw.set_montage("standard_1020", verbose="ERROR")
    except Exception:
        pass
    return raw


def _resolve_eeg_emg_mrcp_raw_root(data_root: Path | None) -> Path:
    candidates: List[Path] = []
    if data_root is not None:
        data_root = Path(data_root)
        candidates.extend(
            [
                data_root.parent / "raw_extracted",
                data_root.parent / "raw",
                data_root.parent.parent / "EEG and EMG Dataset for Analyzing Movement-Related",
            ]
        )
    candidates.append(Path("data/EEG and EMG Dataset for Analyzing Movement-Related"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _mrcp_standard_1005_names(ch_names: List[str]) -> List[str]:
    """Return MRCP channel names in MNE standard_1005 casing.

    The prepared NPZs historically store midline names as FZ/FCZ/CZ/..., but
    MNE's standard montages use Fz/FCz/Cz/... . GEDAI itself is case-insensitive
    for leadfield names, while MNE montage lookup is not, so the raw-path GEDAI
    objects should use the standard_1005 casing.
    """
    z_case = {
        "FZ": "Fz",
        "FCZ": "FCz",
        "CZ": "Cz",
        "CPZ": "CPz",
        "PZ": "Pz",
        "POZ": "POz",
    }
    return [z_case.get(str(c).strip().upper(), str(c).strip()) for c in ch_names]


def _align_mrcp_session_channels(
    X_cont: np.ndarray,
    source_ch_names: List[str],
    expected_ch_names: List[str],
) -> Tuple[np.ndarray, List[str], List[str], List[str], List[str]]:
    """Align one raw MRCP CSV session to the canonical 32-channel layout.

    The public EEG+EMG MRCP CSVs use numeric EEG column names after the trigger
    column. ``prepare_eeg_emg_mrcp.py`` maps those positions to
    ``KNOWN_EEG_CHANNELS``; this raw-path GEDAI loader must do the same or GEDAI
    sees names like "2|3|..." and can fall back to partial leadfield matching.
    """
    source_ch_names = [str(c).strip() for c in source_ch_names]
    expected_std = _mrcp_standard_1005_names(expected_ch_names)
    expected_upper = [c.upper() for c in expected_ch_names]
    expected_map = {c.upper(): i for i, c in enumerate(expected_ch_names)}
    source_upper = [c.upper() for c in source_ch_names]

    if X_cont.shape[0] != len(source_ch_names):
        raise ValueError(
            f"MRCP raw channel count mismatch: data has {X_cont.shape[0]} rows but "
            f"{len(source_ch_names)} channel names were provided."
        )

    if len(source_ch_names) == len(expected_ch_names) and all(
        name.isdigit() for name in source_ch_names
    ):
        return (
            X_cont.astype(np.float32, copy=False),
            expected_std,
            expected_std.copy(),
            [],
            [],
        )

    if len(source_ch_names) == len(expected_ch_names) and source_upper == expected_upper:
        return (
            X_cont.astype(np.float32, copy=False),
            expected_std,
            expected_std.copy(),
            [],
            [],
        )

    X_aligned = np.zeros((len(expected_ch_names), X_cont.shape[1]), dtype=np.float32)
    matched: List[str] = []
    unmatched_source: List[str] = []
    seen_dst: set[int] = set()
    for src_i, name in enumerate(source_ch_names):
        dst_i = expected_map.get(name.upper())
        if dst_i is None:
            unmatched_source.append(name)
            continue
        X_aligned[dst_i] = X_cont[src_i]
        seen_dst.add(dst_i)
        matched.append(expected_std[dst_i])

    missing_expected = [
        expected_std[i] for i in range(len(expected_std)) if i not in seen_dst
    ]
    if not matched:
        raise ValueError(
            "MRCP EEG columns could not be mapped to the canonical 32-channel "
            f"layout. Source columns: {source_ch_names}"
        )

    return X_aligned, expected_std, matched, unmatched_source, missing_expected


def _validate_refcov_shape(
    refcov: np.ndarray | None,
    *,
    n_channels: int,
    subject_id: int,
    session_idx: int,
    source: str,
) -> None:
    if refcov is None:
        return
    if refcov.shape != (n_channels, n_channels):
        raise ValueError(
            f"Subject {subject_id} session {session_idx}: {source} refcov shape "
            f"{refcov.shape} does not match GEDAI channel shape "
            f"({n_channels}, {n_channels})."
        )


def _build_distance_refcov(info) -> np.ndarray:
    """Build a simple distance-based Gram-like refCOV from MNE channel positions.

    Falls back to identity when positions are unavailable. This is used so that
    the vanilla ``gedai`` pipeline on MRCP data (Emotiv Flex 32ch) does not try to
    match against the bundled HD-EEG leadfield — the leadfield only covers a small
    subset of Emotiv Flex channel names (the old 9/32 "wrong b dimensions" failure).
    """
    try:
        import sklearn.metrics
        positions = []
        for i in range(info["nchan"]):
            loc = info["chs"][i]["loc"][:3]
            positions.append(np.asarray(loc, dtype=float))
        positions = np.vstack(positions)
        if not np.isfinite(positions).all():
            return np.eye(info["nchan"], dtype=np.float64)
        if not np.any(np.linalg.norm(positions, axis=1) > 0):
            return np.eye(info["nchan"], dtype=np.float64)
        dmat = sklearn.metrics.pairwise_distances(positions, metric="euclidean")
        # Positive-definite, well-conditioned Gram-like refcov: 1 - dist, then PSD-fix.
        S = 1.0 - dmat
        S = 0.5 * (S + S.T)
        n = S.shape[0]
        evals = np.linalg.eigvalsh(S)
        min_e = float(np.min(evals))
        tr = float(np.trace(S))
        eps = max(1e-6 * (tr / n if tr > 0 else 1.0), 1e-10)
        if min_e <= eps:
            S = S + (eps - min_e + eps) * np.eye(n, dtype=np.float64)
        else:
            S = S + eps * np.eye(n, dtype=np.float64)
        return S.astype(np.float64)
    except Exception:
        return np.eye(info["nchan"], dtype=np.float64)


def _apply_anti_laplacian_to_raw(raw: "mne.io.BaseRaw", n_neighbors: int = 4) -> None:
    """Apply Anti-Laplacian spatial filter in-place (Reyes-Jiménez et al. 2026 §4.5).

    Formula: VL(i) = V(i) + (1/N) * sum_{j ∈ neighbors(i)} V(j)

    For each electrode, the nearest ``n_neighbors`` electrodes (by 3-D Euclidean
    distance from the montage) are averaged and added to the channel signal.  This
    enhances spatially broad slow potentials like the MRCP while averaging down
    spatially uncorrelated noise.

    Falls back silently when electrode positions are unavailable.
    """
    data = raw.get_data()          # (n_ch, n_times)  — float64 copy
    n_ch = data.shape[0]
    pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]], dtype=np.float64)
    has_pos = np.array(
        [np.isfinite(p).all() and np.linalg.norm(p) > 1e-6 for p in pos]
    )
    if not np.any(has_pos):
        return  # no montage positions — skip silently

    data_out = data.copy()
    valid_idx = np.where(has_pos)[0]
    for i in range(n_ch):
        if not has_pos[i]:
            continue
        others = valid_idx[valid_idx != i]
        if others.size == 0:
            continue
        dists = np.linalg.norm(pos[others] - pos[i], axis=1)
        k = min(n_neighbors, others.size)
        neighbors = others[np.argsort(dists)[:k]]
        data_out[i] = data[i] + data[neighbors].mean(axis=0)
    raw._data[:] = data_out.astype(raw._data.dtype)


def _run_mrcp_raw_pipeline(
    subject_id: int,
    raw_root: Path,
    l_freq: float,
    h_freq: float,
    sfreq: float,
    gedai_n_jobs: int | None = None,
    *,
    run_gedai: bool = True,
    reference_cov: np.ndarray | None = None,
    noise_multiplier: float | None = None,
    retention_min: float | None = None,
    use_anti_laplacian: bool = False,
    anti_laplacian_n_neighbors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the continuous-raw MRCP pipeline for both baseline and GEDAI.

    Returns ``(X_clean, X_ref)`` where ``X_ref`` is the paper baseline
    (continuous CAR + 0.1-1 Hz FIR + epoching) and ``X_clean`` is GEDAI output
    (or ``X_ref`` when ``run_gedai=False``). Trial order is ``[rest; move]`` per
    session, matching ``prepare_eeg_emg_mrcp.py``.

    MRCP-specific pipeline (Reyes-Jiménez et al., Data in Brief 65, 2026; GEDAI FAQ):

    per file:
      1. load continuous CSV (128 Hz, microvolts) and build MNE Raw
      2. resample 128 → 250 Hz (FAQ recommends >=200 Hz for GEDAI)
      3. 0.05 Hz high-pass (remove DC drift only; preserves full 0.1-1 Hz MRCP band)
      4. average reference (CAR)
      4b. [optional] Anti-Laplacian: VL(i)=V(i)+(1/N)*sum_neighbors (§4.5 of PDF)
      5. keep a "paper baseline" copy: +0.1-1 Hz Butterworth bandpass (§4.5 of PDF)
      6. fit/transform GEDAI:
           wavelet_type="haar", wavelet_level=8 → bands (0,0.49)(0.49,0.98)... Hz
           noise_multiplier=4.0 (FAQ "auto-" equivalent)
           duration=8 s, overlap=0.5
      7. resample 250 → 128 Hz, apply 0.1-1 Hz Butterworth bandpass
      8. epoch on triggers 7711 (move) and 771 (rest), window [-2, 0] s
    finally:
      9. retention guard in 0.1-1 Hz on 9 motor channels (FC3, FC1, FCz, C3, C1, Cz,
         CP3, CP1, CPz); if retention < MRCP_RETENTION_MIN, fall back to paper baseline.
    """
    import mne

    from ..data.prepare_eeg_emg_mrcp import (
        KNOWN_EEG_CHANNELS,
        _contiguous_trigger_onsets,
        _extract_epochs,
    )

    try:
        from gedai import Gedai
    except ImportError:
        import warnings
        warnings.warn("gedai not installed; returning paper-baseline output only.")
        Gedai = None  # type: ignore

    prepared = _get_mrcp_prepared_sessions(
        subject_id=subject_id,
        raw_root=raw_root,
        sfreq=sfreq,
    )
    if not prepared:
        raise FileNotFoundError(
            f"No EEG CSV files found for subject {subject_id} in {raw_root}"
        )

    import logging as _stdlogging
    _mrcp_diag_log = _stdlogging.getLogger("mrcp_diag")

    X_all: List[np.ndarray] = []
    X_ref_all: List[np.ndarray] = []
    expected_ch_names = _mrcp_standard_1005_names(KNOWN_EEG_CHANNELS.copy())
    n_jobs = int(gedai_n_jobs) if gedai_n_jobs is not None else _DEFAULT_GEDAI_N_JOBS
    n_jobs = max(1, n_jobs)
    trigger_sfreq_in = float(sfreq)
    effective_retention_min = (
        float(retention_min) if retention_min is not None else float(MRCP_RETENTION_MIN)
    )

    # Per-session fallback tracking for diagnostics.
    session_fallback_flags: List[bool] = []
    session_fallback_reasons: List[str] = []

    for session_idx, session in enumerate(prepared):
        ch_names = list(session["ch_names"])
        if [c.upper() for c in ch_names] != [c.upper() for c in expected_ch_names]:
            raise ValueError(
                f"Subject {subject_id} session {session_idx}: prepared raw-path "
                "channels are not in the canonical 32-channel MRCP order. "
                f"Got {ch_names}; expected {expected_ch_names}."
            )
        trigger = np.asarray(session["trigger"], dtype=int)
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=float(MRCP_GEDAI_RESAMPLE_HZ),
            ch_types="eeg",
        )
        try:
            montage = mne.channels.make_standard_montage("standard_1005")
            info.set_montage(montage, on_missing="ignore", verbose=False)
        except Exception:
            pass
        montage_missing = [
            ch["ch_name"] for ch in info["chs"] if not np.isfinite(ch["loc"][:3]).all()
        ]

        # Prepared sessions are already at 250 Hz after HPF+CAR.
        raw = mne.io.RawArray(
            np.asarray(session["eeg_v"], dtype=np.float32),
            info,
            verbose="ERROR",
        )

        # Step 4b: Anti-Laplacian spatial filter (Reyes-Jiménez et al. 2026 §4.5).
        # Applied after CAR and before bandpass, matching the paper's validation pipeline.
        if use_anti_laplacian:
            _apply_anti_laplacian_to_raw(raw, n_neighbors=anti_laplacian_n_neighbors)

        # Step 5: paper-baseline copy (CAR + Anti-Laplacian + 0.1-1 Hz per PDF §4.5).
        raw_ref = raw.copy()
        raw_ref.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="fir",
            fir_design="firwin",
            picks="eeg",
            verbose=False,
        )

        # Step 6: GEDAI with MRCP-aware configuration (skipped when run_gedai=False).
        session_used_fallback = False
        session_fallback_reason = ""
        # Decide refcov BEFORE the call so we can log its shape.
        if run_gedai and Gedai is not None:
            if reference_cov is not None:
                refcov_arr = np.asarray(reference_cov, dtype=np.float64)
                refcov_source = "mrcp_ndarray"
            else:
                # Vanilla `gedai` path: avoid the HD-EEG leadfield (only 9/32 Emotiv
                # Flex channels match → 9×9 refcov vs 32×32 signal → broken).
                # Build a distance-based refcov on the current channels so GEDAI
                # sees a correctly-shaped (n_ch × n_ch) reference.
                refcov_arr = _build_distance_refcov(raw.info)
                refcov_source = "distance_montage"
        else:
            refcov_arr = None
            refcov_source = "none"
        _validate_refcov_shape(
            refcov_arr,
            n_channels=int(raw.info["nchan"]),
            subject_id=subject_id,
            session_idx=session_idx,
            source=refcov_source,
        )

        nm = (
            float(noise_multiplier)
            if noise_multiplier is not None
            else float(MRCP_GEDAI_NOISE_MULTIPLIER)
        )

        # Pre-GEDAI structured diagnostic line.
        _mrcp_diag_log.info(
            "[MRCP_DIAG] stage=pre subject_id=%s session_index=%d n_channels_raw=%d "
            "run_gedai=%s refcov_source=%s refcov_shape=%s "
            "refcov_channel_match=%d/%d noise_multiplier=%.4f retention_min=%.3f "
            "source_file=%s source_ch_names=%s gedai_ch_names=%s "
            "canonical_matched=%s canonical_unmatched=%s canonical_missing=%s "
            "montage_missing=%s",
            subject_id,
            session_idx,
            raw.info["nchan"],
            bool(run_gedai and Gedai is not None),
            refcov_source,
            None if refcov_arr is None else tuple(refcov_arr.shape),
            0 if refcov_arr is None else int(refcov_arr.shape[0]),
            int(raw.info["nchan"]),
            nm,
            effective_retention_min,
            session.get("source_file", ""),
            "|".join(session.get("source_ch_names", ch_names)),
            "|".join(ch_names),
            "|".join(session.get("canonical_matched", [])),
            "|".join(session.get("canonical_unmatched", [])),
            "|".join(session.get("canonical_missing", [])),
            "|".join(montage_missing),
        )

        if not run_gedai or Gedai is None:
            raw_clean = raw_ref.copy()
            session_used_fallback = (Gedai is None and run_gedai)
            session_fallback_reason = "gedai_not_installed" if session_used_fallback else ""
        else:
            try:
                gedai = Gedai(
                    wavelet_type="haar",
                    wavelet_level=MRCP_GEDAI_WAVELET_LEVEL,
                    wavelet_low_cutoff=0.1,
                    epoch_size_in_cycles=12,
                    highpass_cutoff=MRCP_GEDAI_HPF_HZ,
                    signal_type="eeg",
                    preliminary_broadband_noise_multiplier=None,
                )
                fit_kwargs = dict(
                    noise_multiplier=nm,
                    n_jobs=n_jobs,
                    verbose=False,
                    reference_cov=refcov_arr,
                )
                gedai.fit_raw(raw, **fit_kwargs)
                raw_clean = gedai.transform_raw(raw, n_jobs=n_jobs, verbose=False)
            except Exception as e:
                # No silent fallback: emit a structured MRCP_FALLBACK line with full context,
                # then fall back to the paper baseline so the pipeline can still produce
                # trial-aligned output (required for CV).
                session_used_fallback = True
                session_fallback_reason = f"gedai_error:{type(e).__name__}:{e}"
                _mrcp_diag_log.error(
                    "[MRCP_FALLBACK] subject_id=%s session_index=%d n_channels=%d "
                    "refcov_source=%s refcov_shape=%s noise_multiplier=%.4f reason=%s "
                    "source_file=%s gedai_ch_names=%s",
                    subject_id,
                    session_idx,
                    raw.info["nchan"],
                    refcov_source,
                    None if refcov_arr is None else tuple(refcov_arr.shape),
                    nm,
                    session_fallback_reason,
                    session.get("source_file", ""),
                    "|".join(ch_names),
                )
                import warnings
                warnings.warn(
                    f"[MRCP_FALLBACK] GEDAI MRCP fit/transform failed for subject "
                    f"{subject_id} session {session_idx}: {session_fallback_reason}; "
                    "using paper baseline for this session."
                )
                raw_clean = raw_ref.copy()

        session_fallback_flags.append(session_used_fallback)
        session_fallback_reasons.append(session_fallback_reason)

        # Step 7: resample back to original sfreq, then bandpass to 0.1-1 Hz.
        if abs(raw_clean.info["sfreq"] - trigger_sfreq_in) > 0.1:
            raw_clean.resample(trigger_sfreq_in, npad="auto", verbose=False)
        raw_clean.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="fir",
            fir_design="firwin",
            picks="eeg",
            verbose=False,
        )
        if abs(raw_ref.info["sfreq"] - trigger_sfreq_in) > 0.1:
            raw_ref.resample(trigger_sfreq_in, npad="auto", verbose=False)

        # Convert back to microvolts (pipeline convention) and epoch against original-sfreq triggers.
        X_clean_cont = raw_clean.get_data().astype(np.float32, copy=False) * 1e6
        X_ref_cont = raw_ref.get_data().astype(np.float32, copy=False) * 1e6
        movement_onsets = _contiguous_trigger_onsets(trigger, 7711)
        rest_onsets = _contiguous_trigger_onsets(trigger, 771)
        X_move = _extract_epochs(
            X_clean_cont, movement_onsets, sfreq=trigger_sfreq_in, tmin=-2.0, tmax=0.0
        )
        X_rest = _extract_epochs(
            X_clean_cont, rest_onsets, sfreq=trigger_sfreq_in, tmin=-2.0, tmax=0.0
        )
        X_move_ref = _extract_epochs(
            X_ref_cont, movement_onsets, sfreq=trigger_sfreq_in, tmin=-2.0, tmax=0.0
        )
        X_rest_ref = _extract_epochs(
            X_ref_cont, rest_onsets, sfreq=trigger_sfreq_in, tmin=-2.0, tmax=0.0
        )

        n = min(len(X_move), len(X_rest))
        if n == 0:
            continue
        X_all.append(
            np.concatenate([X_rest[:n], X_move[:n]], axis=0).astype(np.float32, copy=False)
        )
        X_ref_all.append(
            np.concatenate([X_rest_ref[:n], X_move_ref[:n]], axis=0).astype(np.float32, copy=False)
        )

    if not X_all:
        raise ValueError(f"No usable MRCP epochs extracted for subject {subject_id}")

    X_clean = np.concatenate(X_all, axis=0).astype(np.float32, copy=False)
    X_ref = np.concatenate(X_ref_all, axis=0).astype(np.float32, copy=False)

    # Step 9: MRCP-specific retention guard over the 9 motor channels in 0.1-1 Hz
    # (only meaningful when GEDAI actually ran).
    retention_ratio = float("nan")
    retention_triggered_fallback = False
    if run_gedai:
        try:
            motor_idx = _select_channel_idx(expected_ch_names, MRCP_MOTOR_CHANNELS)
            if motor_idx.size == 0:
                import warnings
                warnings.warn("MRCP motor channels not found; skipping retention guard.")
            else:
                retention_ratio = float(_median_bandpower_ratio(
                    X_clean[:, motor_idx, :],
                    X_ref[:, motor_idx, :],
                    trigger_sfreq_in,
                    l_freq,
                    h_freq,
                ))
                # retention_min == 0 disables the guard (allows honest nm=0 testing).
                if effective_retention_min > 0.0 and (
                    not np.isfinite(retention_ratio)
                    or retention_ratio < effective_retention_min
                ):
                    retention_triggered_fallback = True
                    _mrcp_diag_log.error(
                        "[MRCP_FALLBACK] subject_id=%s stage=retention_guard "
                        "retention_on_motor_subset=%.4f retention_min=%.3f "
                        "reason=retention_below_min; reverting to paper baseline.",
                        subject_id,
                        retention_ratio,
                        effective_retention_min,
                    )
                    import warnings
                    warnings.warn(
                        f"[MRCP_FALLBACK] GEDAI MRCP 0.1-1 Hz retention on motor subset "
                        f"= {retention_ratio:.3f} < {effective_retention_min:.2f}; "
                        "reverting to paper baseline (CAR + 0.1-1 Hz bandpass)."
                    )
                    X_clean = X_ref
        except Exception as e:
            import warnings
            warnings.warn(f"GEDAI MRCP retention guard failed ({e}); returning GEDAI output.")

    # Post-run structured summary.
    n_sessions = len(session_fallback_flags)
    n_fb = int(sum(session_fallback_flags)) + (1 if retention_triggered_fallback else 0)
    used_fallback_overall = n_fb > 0
    reason_summary = ";".join(
        [r for r in session_fallback_reasons if r]
        + (["retention_below_min"] if retention_triggered_fallback else [])
    ) or "none"
    _mrcp_diag_log.info(
        "[MRCP_DIAG] stage=post subject_id=%s run_gedai=%s n_sessions=%d "
        "n_fallback_sessions=%d retention_on_motor_subset=%.4f "
        "retention_guard_enabled=%s retention_min=%.3f used_fallback=%s "
        "fallback_reason=%s",
        subject_id,
        bool(run_gedai),
        n_sessions,
        n_fb,
        retention_ratio,
        bool(effective_retention_min > 0.0),
        effective_retention_min,
        used_fallback_overall,
        reason_summary,
    )

    return X_clean, X_ref


def _get_mrcp_prepared_sessions(
    subject_id: int,
    raw_root: Path,
    sfreq: float,
) -> List[Dict[str, Any]]:
    """Load and cache per-session MRCP continuous EEG after shared preprocessing.

    Cached payload is post-HPF+CAR at 250 Hz (volts), plus original trigger stream
    at input sampling rate for epoch extraction. This avoids re-reading CSVs and
    re-running resample/HPF/CAR for every fold in leakage-safe GEDAI MRCP runs.
    """
    key = (str(Path(raw_root).resolve()), int(subject_id), float(sfreq))
    cached = _MRCP_PREPARED_CACHE.get(key)
    if cached is not None:
        return cached

    import mne
    import pandas as pd
    from ..data.prepare_eeg_emg_mrcp import (
        KNOWN_EEG_CHANNELS,
        _find_eeg_columns,
        _find_trigger_column,
        _session_files,
        _subject_dir,
    )

    subject_dir = _subject_dir(Path(raw_root), int(subject_id))
    eeg_files = _session_files(subject_dir)
    if not eeg_files:
        raise FileNotFoundError(
            f"No EEG CSV files found for subject {subject_id} in {subject_dir}"
        )

    prepared: List[Dict[str, Any]] = []
    expected_ch_names = KNOWN_EEG_CHANNELS.copy()
    for eeg_path in eeg_files:
        try:
            df = pd.read_csv(eeg_path)
            trigger_col = _find_trigger_column(df)
            eeg_cols = _find_eeg_columns(df, trigger_col)
            trigger = (
                pd.to_numeric(df[trigger_col], errors="coerce")
                .fillna(0)
                .astype(int)
                .to_numpy()
            )
            X_cont = (
                df[eeg_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                .T
            )
            source_ch_names = [str(c) for c in eeg_cols]
            (
                X_cont,
                ch_names,
                canonical_matched,
                canonical_unmatched,
                canonical_missing,
            ) = _align_mrcp_session_channels(
                X_cont=X_cont,
                source_ch_names=source_ch_names,
                expected_ch_names=expected_ch_names,
            )
            if canonical_unmatched or canonical_missing:
                import warnings

                warnings.warn(
                    f"MRCP session {eeg_path.name}: mapped "
                    f"{len(canonical_matched)}/{len(expected_ch_names)} source channels "
                    "to the canonical layout; "
                    f"unmatched_source={canonical_unmatched}; "
                    f"missing_canonical={canonical_missing}."
                )

            info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types="eeg")
            try:
                montage = mne.channels.make_standard_montage("standard_1005")
                info.set_montage(montage, on_missing="ignore", verbose=False)
            except Exception:
                pass
            raw = mne.io.RawArray(X_cont * 1e-6, info, verbose="ERROR")
            if MRCP_GEDAI_RESAMPLE_HZ and abs(raw.info["sfreq"] - MRCP_GEDAI_RESAMPLE_HZ) > 0.1:
                raw.resample(MRCP_GEDAI_RESAMPLE_HZ, npad="auto", verbose=False)
            raw.filter(
                l_freq=MRCP_GEDAI_HPF_HZ,
                h_freq=None,
                method="fir",
                fir_design="firwin",
                picks="eeg",
                verbose=False,
            )
            raw.set_eeg_reference("average", projection=False, verbose=False)
            prepared.append(
                {
                    "ch_names": list(ch_names),
                    "source_ch_names": source_ch_names,
                    "source_file": str(eeg_path),
                    "canonical_matched": list(canonical_matched),
                    "canonical_unmatched": list(canonical_unmatched),
                    "canonical_missing": list(canonical_missing),
                    "trigger": trigger.astype(int, copy=False),
                    "eeg_v": raw.get_data().astype(np.float32, copy=False),
                }
            )
        except Exception as e:
            import warnings

            warnings.warn(
                f"Skipping malformed MRCP raw EEG file {eeg_path.name} for subject "
                f"{subject_id}: {type(e).__name__}: {e}"
            )
            continue

    _MRCP_PREPARED_CACHE[key] = prepared
    return prepared


def apply_gedai_eeg_emg_mrcp_via_raw_files(
    subject_id: int,
    raw_root: Path,
    l_freq: float,
    h_freq: float,
    sfreq: float,
    gedai_n_jobs: int | None = None,
    noise_multiplier: float | None = None,
    retention_min: float | None = None,
) -> np.ndarray:
    """GEDAI-cleaned MRCP epochs via the shared continuous-raw pipeline."""
    X_clean, _ = _run_mrcp_raw_pipeline(
        subject_id=subject_id,
        raw_root=raw_root,
        l_freq=l_freq,
        h_freq=h_freq,
        sfreq=sfreq,
        gedai_n_jobs=gedai_n_jobs,
        run_gedai=True,
        noise_multiplier=noise_multiplier,
        retention_min=retention_min,
    )
    return X_clean


def _compute_mrcp_refcov(
    X_ref: np.ndarray,
    y: np.ndarray,
    n_channels_expected: int,
    prior: str = "grand_avg_erp",
    *,
    sfreq: float = 128.0,
    epoch_tmin_s: float = -2.0,
    refcov_tmin_s: float | None = None,
    rank_max: int | None = None,
    motor_weight: float | None = None,
    move_mix: float | None = None,
) -> np.ndarray:
    """Build MRCP reference covariance from paper-baseline epochs.

    Parameters
    ----------
    X_ref:
        (n_trials, n_channels, n_times) paper baseline epochs.
    y:
        (n_trials,) binary labels with movement=1, rest=0.
    n_channels_expected:
        Sanity check against GEDAI input channels.
    prior:
        One of ``grand_avg_erp`` (xDAWN-style), ``class_contrast`` (movement-rest),
        ``trial_cov_mean`` (mean movement trial covariance).
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    y = np.asarray(y).ravel()
    move = X_ref[y == 1]
    rest = X_ref[y == 0]
    if move.shape[0] < 2:
        raise ValueError(
            f"Need >=2 movement trials to build MRCP reference covariance; got {move.shape[0]}"
        )
    if move.shape[1] != n_channels_expected:
        raise ValueError(
            f"X_ref has {move.shape[1]} channels but GEDAI expects {n_channels_expected}."
        )
    if rest.shape[0] < 2 and prior == "class_contrast":
        raise ValueError(
            f"class_contrast prior needs >=2 rest trials; got {rest.shape[0]}"
        )
    prior = str(prior).strip().lower()

    if refcov_tmin_s is not None:
        refcov_tmin_s = float(refcov_tmin_s)
        start_s = max(float(epoch_tmin_s), refcov_tmin_s)
        start_idx = int(round((start_s - float(epoch_tmin_s)) * float(sfreq)))
        start_idx = max(0, min(int(X_ref.shape[-1]) - 1, start_idx))
        move = move[..., start_idx:]
        rest = rest[..., start_idx:]
        if move.shape[-1] < 4:
            raise ValueError(
                f"MRCP refcov window [{start_s:.3f}, 0.000] s is too short after "
                f"indexing (n_times={move.shape[-1]})."
            )

    # Zero-mean and unit-normalize each trial so the resulting channel covariance
    # depends on spatial structure rather than epoch length / sample rate.
    def _normalize_trials(trials: np.ndarray) -> np.ndarray:
        trials = trials - trials.mean(axis=-1, keepdims=True)
        denom = np.linalg.norm(trials, axis=(1, 2), keepdims=True)
        return trials / np.maximum(denom, 1e-10)

    def _lw_cov_from_trial_means(trials: np.ndarray) -> np.ndarray:
        from sklearn.covariance import LedoitWolf

        X_2d = trials.mean(axis=-1)
        return LedoitWolf().fit(X_2d).covariance_.astype(np.float64, copy=False)

    def _mean_trial_cov(trials: np.ndarray) -> np.ndarray:
        return np.mean(
            np.einsum("tci,tdi->tcd", trials, trials) / float(trials.shape[-1]),
            axis=0,
        )

    move = _normalize_trials(move)
    rest = _normalize_trials(rest) if rest.size else rest

    if prior == "grand_avg_erp":
        grand_avg = move.mean(axis=0)  # (n_channels, n_times)
        erp_cov = grand_avg @ grand_avg.T / float(grand_avg.shape[-1])
        stabilizer = _lw_cov_from_trial_means(move)
        alpha = 0.9
        S = alpha * erp_cov + (1.0 - alpha) * stabilizer
    elif prior == "trial_cov_mean":
        S = _lw_cov_from_trial_means(move)
    elif prior == "class_contrast":
        c_move = _mean_trial_cov(move)
        c_rest = _mean_trial_cov(rest)
        S_raw = 0.5 * (c_move - c_rest + (c_move - c_rest).T)
        evals, evecs = np.linalg.eigh(S_raw)
        max_eval = float(np.max(evals))
        if max_eval <= 0.0:
            # No movement-dominant spatial directions survived the contrast;
            # fall back to the movement covariance so GEDAI still sees the
            # dominant late-MRCP spatial structure rather than a flat floor.
            S = c_move
        else:
            keep = evals > (0.01 * max_eval)
            if not np.any(keep):
                keep[np.argmax(evals)] = True
            pos_idx = np.flatnonzero(keep)
            if rank_max is not None and rank_max > 0 and pos_idx.size > rank_max:
                order = np.argsort(evals[pos_idx])[::-1][: int(rank_max)]
                keep = np.zeros_like(keep, dtype=bool)
                keep[pos_idx[order]] = True
            pos_evals = np.where(keep, evals, 0.0)
            S = (evecs * pos_evals) @ evecs.T
        if move_mix is not None and move_mix > 0.0:
            lam = float(np.clip(move_mix, 0.0, 1.0))
            tr_s = float(np.trace(S))
            tr_move = float(np.trace(c_move))
            c_move_scaled = c_move
            if tr_s > 0.0 and tr_move > 0.0:
                c_move_scaled = c_move * (tr_s / tr_move)
            S = (1.0 - lam) * S + lam * c_move_scaled
    else:
        raise ValueError(
            f"Unknown gedai_mrcp prior '{prior}'. "
            "Expected grand_avg_erp | class_contrast | trial_cov_mean."
        )

    if motor_weight is not None and motor_weight > 1.0:
        weights = np.ones(n_channels_expected, dtype=np.float64)
        # Use the canonical MRCP order expected by GEDAI.
        canonical_names = _mrcp_standard_1005_names(
            [
                "AF3", "AF4", "F3", "F1", "FZ", "F2", "F4",
                "FC3", "FC1", "FCZ", "FC2", "FC4",
                "C3", "C1", "CZ", "C2", "C4",
                "CP3", "CP1", "CPZ", "CP2", "CP4",
                "P3", "P1", "PZ", "P2", "P4",
                "PO3", "POZ", "PO4", "O1", "O2",
            ]
        )
        motor_idx = _select_channel_idx(canonical_names, MRCP_MOTOR_CHANNELS)
        if motor_idx.size > 0:
            weights[motor_idx] = float(motor_weight)
            W = np.diag(weights)
            S = W @ S @ W

    S = 0.5 * (S + S.T)
    evals = np.linalg.eigvalsh(S)
    min_eval = float(np.min(evals))
    n = S.shape[0]
    tr = float(np.trace(S))
    eps = max(1e-6 * (tr / n if tr > 0 else 1.0), 1e-10)
    if min_eval <= eps:
        S = S + (eps - min_eval + eps) * np.eye(n, dtype=np.float64)
    else:
        S = S + eps * np.eye(n, dtype=np.float64)
    return S


def apply_gedai_mrcp_with_mrcp_refcov_via_raw_files(
    subject_id: int,
    raw_root: Path,
    l_freq: float,
    h_freq: float,
    sfreq: float,
    y: np.ndarray,
    gedai_n_jobs: int | None = None,
    train_idx: np.ndarray | None = None,
    X_ref_cache: np.ndarray | None = None,
    prior: str = "grand_avg_erp",
    noise_multiplier: float | None = None,
    retention_min: float | None = None,
    refcov_tmin_s: float | None = None,
    refcov_rank_max: int | None = None,
    refcov_motor_weight: float | None = None,
    refcov_move_mix: float | None = None,
    use_anti_laplacian: bool = False,
    anti_laplacian_n_neighbors: int = 4,
) -> np.ndarray:
    """Run GEDAI with an MRCP-derived reference covariance, leakage-safe by fold.

    Parameters
    ----------
    y:
        Binary labels aligned to NPZ trials, where movement must be encoded as ``1``
        and rest/preparation as ``0``.
    train_idx:
        Optional training-fold indices. When provided, ``reference_cov`` is fit only
        on training movement/rest trials (publishable CV-safe behavior).
    X_ref_cache:
        Optional cached paper-baseline epochs from the same subject/raw pipeline.
        If omitted, this function computes them once internally.
    prior:
        ``grand_avg_erp`` | ``class_contrast`` | ``trial_cov_mean``.

    Contract
    --------
    Trial counts must align exactly with NPZ labels. Any mismatch raises
    ``ValueError`` (no silent truncation).
    """
    y = np.asarray(y).ravel()
    if X_ref_cache is None:
        _, X_ref = _run_mrcp_raw_pipeline(
            subject_id=subject_id,
            raw_root=raw_root,
            l_freq=l_freq,
            h_freq=h_freq,
            sfreq=sfreq,
            gedai_n_jobs=None,
            run_gedai=False,
            use_anti_laplacian=use_anti_laplacian,
            anti_laplacian_n_neighbors=anti_laplacian_n_neighbors,
        )
    else:
        X_ref = np.asarray(X_ref_cache, dtype=np.float32)
    if X_ref.shape[0] != y.shape[0]:
        raise ValueError(
            f"Baseline trial count {X_ref.shape[0]} != NPZ y length {y.shape[0]} "
            f"for subject {subject_id}; cannot align for MRCP covariance."
        )
    if train_idx is not None:
        train_idx = np.asarray(train_idx, dtype=int).ravel()
        if train_idx.size == 0:
            raise ValueError("train_idx is empty; cannot fit MRCP reference covariance.")
        X_fit = X_ref[train_idx]
        y_fit = y[train_idx]
    else:
        X_fit = X_ref
        y_fit = y
    C_mrcp = _compute_mrcp_refcov(
        X_ref=X_fit,
        y=y_fit,
        n_channels_expected=X_ref.shape[1],
        prior=prior,
        sfreq=float(sfreq),
        epoch_tmin_s=-2.0,
        refcov_tmin_s=refcov_tmin_s,
        rank_max=refcov_rank_max,
        motor_weight=refcov_motor_weight,
        move_mix=refcov_move_mix,
    )
    import logging as _stdlogging

    _mrcp_diag_log = _stdlogging.getLogger("mrcp_diag")
    evals = np.linalg.eigvalsh(C_mrcp)
    evals_desc = np.sort(evals)[::-1]
    top = ",".join(f"{v:.4e}" for v in evals_desc[:5])
    max_eval = float(evals_desc[0]) if evals_desc.size else float("nan")
    rank_1pct = int(np.sum(evals_desc > (0.01 * max_eval))) if max_eval > 0 else 0
    _mrcp_diag_log.info(
        "[MRCP_REFCOV] subject_id=%s prior=%s refcov_tmin_s=%s n_train=%d "
        "refcov_rank_max=%s refcov_motor_weight=%s refcov_move_mix=%s n_move=%d n_rest=%d "
        "trace=%.4e rank_1pct=%d top_eigs=%s",
        subject_id,
        prior,
        "none" if refcov_tmin_s is None else f"{float(refcov_tmin_s):.3f}",
        int(X_fit.shape[0]),
        "none" if refcov_rank_max is None else str(int(refcov_rank_max)),
        "none" if refcov_motor_weight is None else f"{float(refcov_motor_weight):.3f}",
        "none" if refcov_move_mix is None else f"{float(refcov_move_mix):.3f}",
        int(np.sum(y_fit == 1)),
        int(np.sum(y_fit == 0)),
        float(np.trace(C_mrcp)),
        rank_1pct,
        top,
    )
    X_clean, _ = _run_mrcp_raw_pipeline(
        subject_id=subject_id, raw_root=raw_root,
        l_freq=l_freq, h_freq=h_freq, sfreq=sfreq,
        gedai_n_jobs=gedai_n_jobs, run_gedai=True,
        reference_cov=C_mrcp,
        noise_multiplier=noise_multiplier,
        retention_min=retention_min,
        use_anti_laplacian=use_anti_laplacian,
        anti_laplacian_n_neighbors=anti_laplacian_n_neighbors,
    )
    if X_clean.shape[0] != y.shape[0]:
        raise ValueError(
            f"GEDAI MRCP trial count {X_clean.shape[0]} != NPZ y length {y.shape[0]} "
            f"for subject {subject_id}; trial alignment contract violated."
        )
    return X_clean


def apply_baseline_eeg_emg_mrcp_via_raw_files(
    subject_id: int,
    raw_root: Path,
    l_freq: float,
    h_freq: float,
    sfreq: float,
    use_anti_laplacian: bool = False,
    anti_laplacian_n_neighbors: int = 4,
) -> np.ndarray:
    """Paper-style MRCP baseline (CAR + [Anti-Laplacian] + 0.1-1 Hz) via continuous-raw path.
    Ensures baseline and GEDAI differ only in the GEDAI denoising step.
    """
    _, X_ref = _run_mrcp_raw_pipeline(
        subject_id=subject_id,
        raw_root=raw_root,
        l_freq=l_freq,
        h_freq=h_freq,
        sfreq=sfreq,
        gedai_n_jobs=None,
        run_gedai=False,
        use_anti_laplacian=use_anti_laplacian,
        anti_laplacian_n_neighbors=anti_laplacian_n_neighbors,
    )
    return X_ref


def preprocess_subject_data(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    denoising: str,
    subject_id: int = None,     # used by GEDAI to load continuous raw
    runs: List[int] = None,     # used by GEDAI to select session runs
    tmin: float = -1.0,         # epoch window start
    tmax: float = 4.0,          # epoch window end
    y: np.ndarray = None,       # labels (used to re-epoch after GEDAI)
    train_idx: np.ndarray = None,  # training-fold indices for leakage-safe GEDAI MRCP refcov fit
    dataset_name: str = "",     # dataset name to gate dataset-specific logic
    gedai_n_jobs: int | None = None,
    data_root: Path | None = None,
    mrcp_refcov_prior: str = "grand_avg_erp",
    mrcp_refcov_cache: np.ndarray | None = None,
    mrcp_gedai_noise_multiplier: float | None = None,
    mrcp_gedai_retention_min: float | None = None,
    mrcp_refcov_tmin_s: float | None = None,
    mrcp_refcov_rank_max: int | None = None,
    mrcp_refcov_motor_weight: float | None = None,
    mrcp_refcov_move_mix: float | None = None,
    use_anti_laplacian: bool = False,
    anti_laplacian_n_neighbors: int = 4,
) -> np.ndarray:
    """Preprocess one subject once per denoising pipeline.

    For GEDAI, when `subject_id` is provided and dataset is physionet, loads the continuous PhysioNet
    EDF session, applies GEDAI on the full recording, then re-epochs. This is
    the correct usage of GEDAI (continuous raw, not concatenated short trials).

    Falls back to the old epoch-level GEDAI for non-PhysioNet datasets.

    For PyLossless on Alljoined, set ``data_root`` and add
    ``<data_root>/pylossless_manifest/subject_<id>.txt`` (line 1 = metadata parquet, following lines =
    EDF paths) to run default PyLossless **per EDF** before epoching; otherwise trials are concatenated
    into one continuous ``Raw`` (cf. ASR).
    """
    if denoising not in {"baseline", "icalabel", "gedai", "gedai_mrcp", "asr", "pylossless"}:
        raise ValueError(f"Unknown denoising strategy: {denoising}")

    # ── MRCP-cov GEDAI (Prof. Ros / Cohen 2022): inject C_MRCP as GEDAI reference.
    if denoising == "gedai_mrcp":
        if not ("eeg_emg_mrcp" in (dataset_name or "").lower()) or subject_id is None:
            raise ValueError(
                "gedai_mrcp requires dataset_label 'eeg_emg_mrcp' and a subject_id."
            )
        if y is None:
            raise ValueError("gedai_mrcp requires labels y to build MRCP covariance.")
        raw_root = _resolve_eeg_emg_mrcp_raw_root(data_root)
        X_gm = apply_gedai_mrcp_with_mrcp_refcov_via_raw_files(
            int(subject_id),
            raw_root=raw_root,
            l_freq=l_freq,
            h_freq=h_freq,
            sfreq=float(sfreq),
            y=y,
            gedai_n_jobs=gedai_n_jobs,
            train_idx=train_idx,
            X_ref_cache=mrcp_refcov_cache,
            prior=mrcp_refcov_prior,
            noise_multiplier=mrcp_gedai_noise_multiplier,
            retention_min=mrcp_gedai_retention_min,
            refcov_tmin_s=mrcp_refcov_tmin_s,
            refcov_rank_max=mrcp_refcov_rank_max,
            refcov_motor_weight=mrcp_refcov_motor_weight,
            refcov_move_mix=mrcp_refcov_move_mix,
            use_anti_laplacian=use_anti_laplacian,
            anti_laplacian_n_neighbors=anti_laplacian_n_neighbors,
        )
        if X_gm.shape[0] != X.shape[0]:
            raise ValueError(
                f"gedai_mrcp trial count {X_gm.shape[0]} != NPZ {X.shape[0]}."
            )
        return X_gm.astype(np.float32, copy=False)

    # X_bp = bandpass_filter(X, sfreq, l_freq, h_freq) # REMOVED TO SAVE RAM
    if denoising == "baseline":
        if (
            dataset_name
            and "eeg_emg_mrcp" in dataset_name.lower()
            and subject_id is not None
        ):
            # FAIR MRCP baseline: same continuous-raw path as GEDAI (CAR on continuous,
            # FIR firwin 0.1-1 Hz, 128->250->128 resample round-trip), just without GEDAI.
            # This ensures baseline vs GEDAI differ only in the GEDAI denoising step.
            try:
                raw_root = _resolve_eeg_emg_mrcp_raw_root(data_root)
                X_base = apply_baseline_eeg_emg_mrcp_via_raw_files(
                    int(subject_id),
                    raw_root=raw_root,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    sfreq=float(sfreq),
                    use_anti_laplacian=use_anti_laplacian,
                    anti_laplacian_n_neighbors=anti_laplacian_n_neighbors,
                )
                if X_base.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"MRCP raw-path baseline trial count {X_base.shape[0]} != "
                        f"NPZ trial count {X.shape[0]} for subject {subject_id}. "
                        "Trial alignment with stored y would be broken."
                    )
                return X_base.astype(np.float32, copy=False)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"MRCP raw-path baseline failed for subject {subject_id} ({e}); "
                    f"falling back to per-epoch CAR + bandpass (NOTE: not fair vs GEDAI)."
                )
        if dataset_name and "eeg_emg_mrcp" in dataset_name.lower():
            # Per-epoch CAR + bandpass (fallback when subject_id/raw files unavailable).
            X0 = np.asarray(X, dtype=np.float32).copy()
            X0 -= X0.mean(axis=1, keepdims=True)
            return bandpass_filter(X0, sfreq, l_freq, h_freq).astype(np.float32, copy=False)
        return bandpass_filter(
            np.asarray(X, dtype=np.float32).copy(), sfreq, l_freq, h_freq
        ).astype(np.float32, copy=False)

    if denoising == "asr":
        return apply_asr(X, sfreq, ch_names, l_freq, h_freq).astype(
            np.float32, copy=False
        )

    if denoising == "icalabel":
        # Delete original X immediately if possible or rely on apply_icalabel to be smart
        return apply_icalabel(X, sfreq, ch_names, l_freq, h_freq).astype(
            np.float32, copy=False
        )

    if denoising == "pylossless":
        X = np.asarray(X, dtype=np.float32)
        if (
            subject_id is not None
            and data_root is not None
            and dataset_name
            and "alljoined" in dataset_name.lower()
        ):
            X_m = apply_pylossless_alljoined_via_manifest(
                int(subject_id),
                Path(data_root),
                ch_names,
                l_freq,
                h_freq,
                float(sfreq),
                n_trials_expected=int(X.shape[0]),
            )
            if X_m is not None:
                return X_m.astype(np.float32, copy=False)
        return apply_pylossless(X, sfreq, ch_names, l_freq, h_freq).astype(
            np.float32, copy=False
        )

    # ── GEDAI path ─────────────────────────────────────────────────────────
    if subject_id is not None and "physionet" in dataset_name.lower():
        # Correct path: load continuous raw, denoise, re-epoch
        try:
            import mne
            from mne import Epochs, events_from_annotations
            raw = load_continuous_raw_physionet(subject_id, runs=runs)
            raw_clean = apply_gedai_from_continuous_raw(raw, gedai_n_jobs=gedai_n_jobs)

            # Bandpass the cleaned continuous raw in the decoding band
            raw_clean.filter(
                l_freq=l_freq, h_freq=h_freq,
                method="fir", fir_design="firwin",
                picks="eeg", verbose=False,
            )

            # Re-epoch from cleaned raw using original event structure
            events, _ = events_from_annotations(raw_clean, verbose=False)
            event_id = dict(hands=2, feet=3)
            import mne
            epochs = Epochs(
                raw_clean,
                events=events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                preload=True,
                verbose="ERROR",
            )
            X_gedai = epochs.get_data().astype(np.float32)
            # Match trial count to stored X (some events may not align)
            n = min(X_gedai.shape[0], X.shape[0])
            return X_gedai[:n]
        except Exception as e:
            import warnings
            warnings.warn(
                f"GEDAI continuous path failed for subject {subject_id} ({e}); "
                f"falling back to epoch-level GEDAI."
            )

    if subject_id is not None and "eeg_emg_mrcp" in dataset_name.lower():
        try:
            raw_root = _resolve_eeg_emg_mrcp_raw_root(data_root)
            X_gedai = apply_gedai_eeg_emg_mrcp_via_raw_files(
                int(subject_id),
                raw_root=raw_root,
                l_freq=l_freq,
                h_freq=h_freq,
                sfreq=float(sfreq),
                gedai_n_jobs=gedai_n_jobs,
                noise_multiplier=mrcp_gedai_noise_multiplier,
                retention_min=mrcp_gedai_retention_min,
            )
            if X_gedai.shape[0] != X.shape[0]:
                raise ValueError(
                    f"MRCP raw-path GEDAI trial count {X_gedai.shape[0]} != "
                    f"NPZ trial count {X.shape[0]} for subject {subject_id}. "
                    "Trial alignment with stored y would be broken."
                )
            return X_gedai
        except Exception as e:
            import warnings
            warnings.warn(
                f"GEDAI MRCP raw-file path failed for subject {subject_id} ({e}); "
                f"falling back to epoch-level GEDAI."
            )

    # Fallback: epoch-level GEDAI (less accurate but works without raw EDFs)
    return apply_gedai(
        X, sfreq, ch_names, l_freq, h_freq, gedai_n_jobs=gedai_n_jobs
    ).astype(np.float32, copy=False)
