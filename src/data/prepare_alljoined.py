from __future__ import annotations

"""
Prepare Alljoined-1.6M (consumer EEG) dataset subset.

Downloads from Hugging Face:
- Raw EDF files (32 channels, 256 Hz)
- Stimulus metadata (Parquet)

Processes each subject by:
1. Identifying sessions and blocks.
2. Aligning EDF annotations with stimulus metadata.
3. Pulse-epoching (100ms images at 256Hz).
4. Saving to data/alljoined/processed/subject_<ID>.npz.

Usage:
    python3 src/data/prepare_alljoined.py --subjects 1 2 --out-root data/alljoined/processed
"""

import argparse
import gc
import os
import random
import re
import time
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

import numpy as np
import pandas as pd
import mne
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Alljoined/Alljoined-1.6M"

T = TypeVar("T")


def _is_transient_download_error(exc: BaseException) -> bool:
    """True for typical network drops (HF, proxies, Wi‑Fi) that often succeed on retry."""
    msg = str(exc).lower()
    name = type(exc).__name__
    if "10054" in msg or "forcibly closed" in msg:
        return True
    if "connection" in name.lower() or "timeout" in name.lower():
        return True
    if "timeout" in msg or "connection reset" in msg or "broken pipe" in msg:
        return True
    if "remotedisconnected" in name or "protocolerror" in name:
        return True
    return False


def _hf_retry(desc: str, fn: Callable[[], T], max_attempts: int = 8) -> T:
    """Retry Hugging Face HTTP calls on transient failures (e.g. WinError 10054)."""
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except BaseException as e:
            last = e
            if attempt >= max_attempts or not _is_transient_download_error(e):
                raise
            wait = min(120.0, 2**attempt + random.uniform(0, 2))
            print(
                f"  [{desc}] failed ({type(e).__name__}), retry {attempt}/{max_attempts} in {wait:.0f}s...",
                flush=True,
            )
            time.sleep(wait)
    assert last is not None
    raise last
TARGET_SFREQ = 250.0  # Consistency with other pipelines
EEG_CHANNELS = [
    'Cz', 'FCz', 'Afz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1', 
    'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'Pz', 'POz', 'Oz', 'O2', 
    'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2', 'F6', 'Fp2'
]

def download_subject_data(subject: int, max_edfs: Optional[int] = None) -> tuple[List[str], str]:
    """Download EDF files and metadata for a subject. Returns local paths."""
    print(f"  Listing files for Subject {subject}...", flush=True)
    all_files = _hf_retry(
        "list_repo_files",
        lambda: list_repo_files(REPO_ID, repo_type="dataset"),
    )
    
    # Raw EDFs
    subj_str = f"sub-{subject:02d}"
    edf_files = [f for f in all_files if subj_str in f and f.endswith(".edf")]
    if max_edfs:
        edf_files = edf_files[:max_edfs]
    
    print(f"  Found {len(edf_files)} EDF files to process. Downloading...", flush=True)
    
    local_edfs = []
    for f in edf_files:
        fname = Path(f).name
        path = _hf_retry(
            f"download {fname}",
            lambda fp=f: hf_hub_download(
                repo_id=REPO_ID, filename=fp, repo_type="dataset"
            ),
        )
        local_edfs.append(path)

    # Metadata
    meta_file = f"preprocessed_eeg/{subj_str}/experiment_metadata_categories.parquet"
    print(f"  Downloading metadata: {meta_file}...", flush=True)
    meta_path = _hf_retry(
        "download metadata parquet",
        lambda: hf_hub_download(
            repo_id=REPO_ID, filename=meta_file, repo_type="dataset"
        ),
    )
    
    return local_edfs, meta_path

def process_subject(subject: int, edf_paths: List[str], meta_path: str, out_root: Path):
    """Epochs data for a single subject and saves to NPZ."""
    df_meta = pd.read_parquet(meta_path)
    # Extract image_id from image_path (e.g. /.../16641.jpg -> 16641)
    df_meta['image_id_val'] = df_meta['image_path'].apply(
        lambda x: int(Path(x).stem) if pd.notnull(x) else -1
    )
    
    all_X = []
    all_y = []
    
    # Fixed-length epochs (samples) so np.stack never fails across sessions/blocks.
    tmin, tmax = 0.0, 1.0
    n_samples = int(round(tmax * TARGET_SFREQ))

    print(f"  Epoching Subject {subject}...", flush=True)

    for edf_path in edf_paths:
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            print(f"    [Skip] Error reading {Path(edf_path).name}: {e}")
            continue
            
        raw.pick_channels([c for c in EEG_CHANNELS if c in raw.ch_names])
        
        # Resample to 250Hz immediately to save memory
        if abs(raw.info['sfreq'] - TARGET_SFREQ) > 0.1:
            raw.resample(TARGET_SFREQ, verbose=False)
        
        # Parse annotations: "partition, image_id, sequence_id, sequence_image_id"
        annots = raw.annotations
        matches_found = 0
        for i, ann in enumerate(annots):
            desc = ann['description']
            if not desc.startswith("stim_"):
                continue
            
            parts = [p.strip() for p in desc.split(",")]
            if len(parts) < 2:
                continue
            
            # The second part is image_id (e.g. 16558)
            try:
                ann_image_id = int(parts[1])
            except ValueError:
                continue
            
            onset = ann['onset']
            
            # Find matching metadata row based on image_id
            match = df_meta[df_meta['image_id_val'] == ann_image_id]
            
            if match.empty:
                continue
                
            label = match.iloc[0]['category_num']
            
            start_samp = int(raw.time_as_index(onset)[0])
            stop_samp = start_samp + n_samples
            if start_samp < 0 or stop_samp > raw.n_times:
                continue

            chunk = raw.get_data(start=start_samp, stop=stop_samp)
            if chunk.shape[1] < n_samples:
                continue
            if chunk.shape[1] > n_samples:
                chunk = chunk[:, :n_samples]

            all_X.append(chunk.astype(np.float32, copy=False))
            all_y.append(label)
            matches_found += 1
            
        print(f"    Processed {Path(edf_path).name}: {matches_found} matches.", flush=True)
        raw.close()
        del raw
        gc.collect()

    if not all_X:
        print(f"  [Error] No trials found for Subject {subject}.")
        return

    X = np.stack(all_X).astype("float32")
    y = np.array(all_y, dtype=int)
    
    out_path = out_root / f"subject_{subject}.npz"
    np.savez(
        out_path,
        X=X,
        y=y,
        sfreq=TARGET_SFREQ,
        ch_names=np.array(EEG_CHANNELS, dtype=object)
    )
    
    print(f"  ✓ Saved {len(all_y)} trials to {out_path}")
    return {
        "subject": subject,
        "n_trials": len(all_y),
        "n_channels": X.shape[1],
        "n_times": X.shape[2],
        "sfreq": TARGET_SFREQ
    }

def _prepare_subject(subject: int, out_root: Path, max_edfs: int | None = 2) -> dict | None:
    """Download and process a subject for the inspection/benchmark suite."""
    edf_paths, meta_path = download_subject_data(subject, max_edfs=max_edfs)
    return process_subject(subject, edf_paths, meta_path, out_root)


def _prepare_subject_worker(payload: tuple) -> tuple[int, str | None]:
    """Picklable worker for ProcessPoolExecutor (Windows spawn). Returns (subject_id, error_or_none)."""
    subject, out_root_str, max_edfs, overwrite = payload
    out_root = Path(out_root_str)
    out_path = out_root / f"subject_{subject}.npz"
    if out_path.exists() and not overwrite:
        return subject, None
    try:
        print(f"Preparing Alljoined Subject {subject}...", flush=True)
        info = _prepare_subject(subject, out_root, max_edfs=max_edfs)
        if info:
            print(f"  ✓ Subject {subject}: {info}", flush=True)
        return subject, None
    except Exception as e:
        return subject, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--out-root", type=str, default="data/alljoined/processed")
    parser.add_argument(
        "--max-edfs",
        type=int,
        default=2,
        help="Cap EDF files per subject (smoke tests). Ignored if --all-edfs.",
    )
    parser.add_argument(
        "--all-edfs",
        action="store_true",
        help="Use every EDF listed on Hugging Face for each subject (full benchmark).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel subject downloads/processes (1 = sequential). Watch RAM and HF rate limits.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files")
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    max_edfs: int | None = None if args.all_edfs else int(args.max_edfs)

    todo: List[int] = []
    for subj in args.subjects:
        out_path = out_root / f"subject_{subj}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping Subject {subj} (already exists). Use --overwrite to re-process.")
            continue
        todo.append(subj)

    if not todo:
        return

    if args.workers <= 1:
        for subj in todo:
            print(f"Preparing Alljoined Subject {subj}...")
            try:
                info = _prepare_subject(subj, out_root, max_edfs=max_edfs)
                if info:
                    print(f"  ✓ {info}")
            except Exception as e:
                print(f"  [Failed] Subject {subj}: {e}")
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    payloads = [
        (s, str(out_root.resolve()), max_edfs, args.overwrite) for s in todo
    ]
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(_prepare_subject_worker, p) for p in payloads]
        for fut in as_completed(futures):
            subj, err = fut.result()
            if err:
                print(f"  [Failed] Subject {subj}: {err}", flush=True)

if __name__ == "__main__":
    main()
