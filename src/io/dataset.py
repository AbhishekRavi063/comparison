from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class SubjectData:
    """Container for a single subject's data.

    Attributes
    ----------
    X : np.ndarray
        Shape (n_trials, n_channels, n_times), dtype float32/float64.
    y : np.ndarray
        Shape (n_trials,), integer class labels.
    sfreq : float
        Sampling frequency in Hz.
    ch_names : list of str
        Channel names corresponding to the channels dimension.
    """

    X: np.ndarray
    y: np.ndarray
    sfreq: float
    ch_names: List[str]


class NpzMotorImageryDataset:
    """Simple loader for per-subject .npz motor imagery datasets.

    Expected .npz structure per subject::

        X: (n_trials, n_channels, n_times)
        y: (n_trials,)
        sfreq: float
        ch_names: list of str or array of str

    Filenames are assumed to follow the pattern ``subject_<id>.npz`` by default.
    """

    def __init__(
        self,
        data_root: Path,
        subjects: Iterable[int],
        pattern: str = "subject_{id}.npz",
        float_dtype: str = "float32",
    ) -> None:
        self.data_root = Path(data_root)
        self.subjects = list(subjects)
        self.pattern = pattern
        self.float_dtype = np.dtype(float_dtype)

    def _load_subject_file(self, subject_id: int) -> SubjectData:
        fname = self.data_root / self.pattern.format(id=subject_id)
        if not fname.exists():
            raise FileNotFoundError(f"Subject file not found: {fname}")

        with np.load(fname, allow_pickle=True) as data:
            X = data["X"].astype(self.float_dtype, copy=False)
            y = data["y"].astype(int, copy=False)
            sfreq = float(data["sfreq"].item() if np.ndim(data["sfreq"]) else data["sfreq"])
            ch_names_raw = data["ch_names"]

        # Ensure ch_names is a list of Python strings
        if isinstance(ch_names_raw, np.ndarray):
            ch_names = [str(c) for c in ch_names_raw.tolist()]
        else:
            ch_names = [str(c) for c in ch_names_raw]

        return SubjectData(X=X, y=y, sfreq=sfreq, ch_names=ch_names)

    def iter_subjects(self) -> Iterable[Tuple[int, SubjectData]]:
        """Yield (subject_id, SubjectData) pairs. Streams one subject at a time; no list of all data is kept."""
        for sid in self.subjects:
            yield sid, self._load_subject_file(sid)

