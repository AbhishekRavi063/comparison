"""List trial counts and file sizes for data/alljoined/processed/subject_*.npz."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "alljoined" / "processed"
    if not root.is_dir():
        print(f"Missing directory: {root}")
        return
    paths = sorted(root.glob("subject_*.npz"))
    if not paths:
        print(f"No subject_*.npz under {root}")
        return
    for p in paths:
        with np.load(p) as z:
            x = z["X"]
            print(f"{p.name}: trials={x.shape[0]} ch={x.shape[1]} t={x.shape[2]} size_MB={p.stat().st_size / 1024**2:.1f}")


if __name__ == "__main__":
    main()
