from __future__ import annotations

import os
from typing import List, Tuple

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
# Set EEG_ICA_VERBOSE=1 to print rejected components and max signal difference.
# Bands for brain-signal preservation checks (alpha/mu, beta).
ALPHA_BAND = (8.0, 12.0)
BETA_BAND = (13.0, 30.0)


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
    """Effective retention ratio = min(full_band, motor_band, alpha_band, beta_band).
    Ensures no over-removal of brain signal in decode band or in alpha/mu/beta specifically.
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
    alpha = _median_bandpower_ratio(
        x_clean, x_band_ref, sfreq, ALPHA_BAND[0], ALPHA_BAND[1]
    )
    beta = _median_bandpower_ratio(
        x_clean, x_band_ref, sfreq, BETA_BAND[0], BETA_BAND[1]
    )
    return min(full, motor, alpha, beta)


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
    x_band_ref = bandpass_filter(X, sfreq, l_freq, h_freq).astype(np.float32, copy=False)

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
# Retention guard — set to 0 (disabled) to avoid passthrough masking results.
GEDAI_RETENTION_MEDIAN_MIN = 0.0
GEDAI_RETENTION_HARD_MIN = 0.0


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
    return x_clean



def apply_gedai_from_continuous_raw(
    raw: "mne.io.BaseRaw",  # full continuous session, already loaded with montage
    gedai_n_jobs: int | None = None,
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
            noise_multiplier=GEDAI_BROADBAND_NOISE_MULTIPLIER,
            n_jobs=n_jobs,
            verbose=False,
        )
        raw_broad = g_broad.transform_raw(raw, n_jobs=n_jobs, verbose=False)

        # Step 2: frequency-specific spectral denoising
        g_spec = Gedai(
            wavelet_type="haar",
            wavelet_level=GEDAI_SPECTRAL_WAVELET_LEVEL,
            wavelet_low_cutoff=GEDAI_SPECTRAL_LOW_CUTOFF,
        )
        g_spec.fit_raw(
            raw_broad,
            noise_multiplier=GEDAI_SPECTRAL_NOISE_MULTIPLIER,
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
    dataset_name: str = "",     # dataset name to gate dataset-specific logic
    gedai_n_jobs: int | None = None,
) -> np.ndarray:
    """Preprocess one subject once per denoising pipeline.

    For GEDAI, when `subject_id` is provided and dataset is physionet, loads the continuous PhysioNet
    EDF session, applies GEDAI on the full recording, then re-epochs. This is
    the correct usage of GEDAI (continuous raw, not concatenated short trials).

    Falls back to the old epoch-level GEDAI for non-PhysioNet datasets.
    """
    if denoising not in {"baseline", "icalabel", "gedai", "asr"}:
        raise ValueError(f"Unknown denoising strategy: {denoising}")

    # X_bp = bandpass_filter(X, sfreq, l_freq, h_freq) # REMOVED TO SAVE RAM
    if denoising == "baseline":
        return bandpass_filter(X, sfreq, l_freq, h_freq).astype(np.float32, copy=False)

    if denoising == "asr":
        return apply_asr(X, sfreq, ch_names, l_freq, h_freq).astype(
            np.float32, copy=False
        )

    if denoising == "icalabel":
        # Delete original X immediately if possible or rely on apply_icalabel to be smart
        return apply_icalabel(X, sfreq, ch_names, l_freq, h_freq).astype(
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

    # Fallback: epoch-level GEDAI (less accurate but works without raw EDFs)
    return apply_gedai(
        X, sfreq, ch_names, l_freq, h_freq, gedai_n_jobs=gedai_n_jobs
    ).astype(np.float32, copy=False)

