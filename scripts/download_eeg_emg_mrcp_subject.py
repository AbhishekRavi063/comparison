#!/usr/bin/env python3
from __future__ import annotations

"""
Best-effort downloader for the EEG+EMG MRCP Mendeley dataset.

Dataset:
  EEG and EMG dataset for analyzing movement-related cortical potentials in hand
  gesture tasks
  DOI: 10.17632/y23s2xg6x4.1

This helper tries to use the Mendeley dataset API to fetch files for a specific
subject. Public Mendeley datasets sometimes expose download URLs directly, but
some environments require an OAuth bearer token. This script supports both:

  - unauthenticated access if the endpoint allows it
  - authenticated access via --token or MENDELEY_TOKEN

If per-subject files cannot be identified, the script can still download any
matching archives and tell you what it found.
"""

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET_ID = "y23s2xg6x4"
DATASET_VERSION = 1
PUBLIC_ACCEPT = "application/vnd.mendeley-public-dataset.1+json"


def _http_get_json(url: str, token: str | None) -> dict[str, Any]:
    headers = {"Accept": PUBLIC_ACCEPT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_download(url: str, token: str | None, out_path: Path) -> None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req) as resp, out_path.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def _flatten_files(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        files = obj.get("files")
        if isinstance(files, list):
            return [f for f in files if isinstance(f, dict)]
    return []


def _filename_for(file_obj: dict[str, Any]) -> str:
    for key in ("filename", "name", "title"):
        value = file_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    content = file_obj.get("content_details")
    if isinstance(content, dict):
        value = content.get("filename")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown_file"


def _download_url_for(file_obj: dict[str, Any]) -> str | None:
    content = file_obj.get("content_details")
    if isinstance(content, dict):
        url = content.get("download_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
        content_id = content.get("id")
        if isinstance(content_id, str) and content_id.strip():
            return f"https://api.mendeley.com/files/{content_id.strip()}"

    file_id = file_obj.get("id")
    if isinstance(file_id, str) and file_id.strip():
        return f"https://api.mendeley.com/files/{file_id.strip()}"
    return None


def _subject_patterns(subject: int) -> list[re.Pattern[str]]:
    subject_token = f"{subject:02d}"
    return [
        re.compile(rf"\bsubject[_\-\s]?{subject_token}\b", re.IGNORECASE),
        re.compile(rf"\bsubj(?:ect)?[_\-\s]?{subject_token}\b", re.IGNORECASE),
        re.compile(rf"\b0*{subject}\b"),
    ]


def _matches_subject(filename: str, subject: int) -> bool:
    return any(p.search(filename) for p in _subject_patterns(subject))


def _extract_archives(paths: Iterable[Path], extract_root: Path) -> list[Path]:
    extracted: list[Path] = []
    extract_root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.suffix.lower() != ".zip":
            continue
        dest = extract_root / path.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
        extracted.append(dest)
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download subject-specific files from the EEG+EMG MRCP Mendeley dataset."
    )
    parser.add_argument("--subject", type=int, required=True, help="Subject number, e.g. 1")
    parser.add_argument("--dataset-id", type=str, default=DATASET_ID)
    parser.add_argument("--version", type=int, default=DATASET_VERSION)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/eeg_emg_mrcp/raw_downloads",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data/eeg_emg_mrcp/raw_extracted",
        help="Directory to extract any downloaded zip archives",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("MENDELEY_TOKEN", ""),
        help="Optional Mendeley OAuth bearer token. Can also use MENDELEY_TOKEN.",
    )
    parser.add_argument(
        "--download-all-if-no-match",
        action="store_true",
        help="Download all dataset files if no obvious subject-specific file is found.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    extract_dir = Path(args.extract_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query = urlencode({"version": args.version})
    dataset_url = f"https://api.mendeley.com/datasets/{args.dataset_id}?{query}"
    token = args.token or None

    try:
        dataset = _http_get_json(dataset_url, token)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            print(
                "Mendeley rejected the request. This dataset endpoint may require an OAuth token.\n"
                "Retry with:\n"
                "  export MENDELEY_TOKEN='<your token>'\n"
                "or pass --token directly.\n"
                "Dataset page: https://data.mendeley.com/datasets/y23s2xg6x4/1",
                file=sys.stderr,
            )
            return 2
        print(f"HTTP error fetching dataset metadata: {exc}", file=sys.stderr)
        return 2
    except URLError as exc:
        print(f"Network error fetching dataset metadata: {exc}", file=sys.stderr)
        return 2

    files = _flatten_files(dataset)
    if not files:
        print(
            "No file metadata was returned by the dataset API. You may need to download "
            "the archive manually from https://data.mendeley.com/datasets/y23s2xg6x4/1",
            file=sys.stderr,
        )
        return 2

    matches = [f for f in files if _matches_subject(_filename_for(f), args.subject)]
    selected = matches

    if not selected and args.download_all_if_no_match:
        selected = files

    if not selected:
        print("Could not identify subject-specific files from metadata.", file=sys.stderr)
        print("Available filenames:", file=sys.stderr)
        for file_obj in files:
            print(f"  - {_filename_for(file_obj)}", file=sys.stderr)
        print(
            "\nRetry with --download-all-if-no-match, or download manually from:\n"
            "  https://data.mendeley.com/datasets/y23s2xg6x4/1",
            file=sys.stderr,
        )
        return 2

    downloaded: list[Path] = []
    for file_obj in selected:
        filename = _filename_for(file_obj)
        url = _download_url_for(file_obj)
        if not url:
            print(f"Skipping {filename}: no download URL in metadata", file=sys.stderr)
            continue
        out_path = out_dir / filename
        print(f"Downloading {filename} -> {out_path}")
        try:
            _http_download(url, token, out_path)
        except HTTPError as exc:
            print(f"HTTP error downloading {filename}: {exc}", file=sys.stderr)
            continue
        except URLError as exc:
            print(f"Network error downloading {filename}: {exc}", file=sys.stderr)
            continue
        downloaded.append(out_path)

    if not downloaded:
        print("No files were downloaded.", file=sys.stderr)
        return 2

    extracted = _extract_archives(downloaded, extract_dir)

    print("\nDone.")
    print("Downloaded:")
    for path in downloaded:
        print(f"  - {path}")
    if extracted:
        print("Extracted zip archives:")
        for path in extracted:
            print(f"  - {path}")
    else:
        print("No zip archives extracted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
