"""Band-specific class separability for AASD NPZ (explains mu-only vs wideband decoding).

Compares Cohen's *d* of log band-power between classes after the same CAR + wide
bandpass used for broadband baseline (0.1–45 Hz). If beta (or other bands) shows
larger *d* than mu (7–13 Hz), narrowing the decoder to mu alone removes useful
information — consistent with lower CSP/tangent accuracy under a 7–13 Hz bandpass.

Usage:
  python3 -m src.diagnose_aasd_band_separability \\
    --data-root data/aasd/npz --subjects 1 2 --sfreq 128

Requires: scipy (Welch). Uses the project's ``bandpass_filter`` and CAR helper.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import welch


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = max(1, a.size), max(1, b.size)
    va = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    vb = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if pooled < 1e-20:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _average_reference_trials(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).copy()
    X -= X.mean(axis=1, keepdims=True)
    return X


def _bandpass(
    X: np.ndarray, sfreq: float, l_freq: float, h_freq: float
) -> np.ndarray:
    from .denoising.pipelines import bandpass_filter

    return np.asarray(
        bandpass_filter(
            np.asarray(X, dtype=np.float32).copy(), sfreq, l_freq, h_freq
        ),
        dtype=np.float32,
    )


def _log_band_powers_per_trial(
    X_wb: np.ndarray, sfreq: float, bands: dict[str, Tuple[float, float]]
) -> dict[str, np.ndarray]:
    """X_wb: (n_trials, n_ch, n_times). Return log integrated power per trial per band (mean over ch)."""
    n_trials, n_ch, n_t = X_wb.shape
    nperseg = min(256, n_t)
    if nperseg < 8:
        raise ValueError(f"Need longer epochs for Welch; got n_times={n_t}")

    out: dict[str, np.ndarray] = {k: np.zeros(n_trials, dtype=np.float64) for k in bands}
    f_buf = None

    for i in range(n_trials):
        # Mean PSD across channels → one spectrum per trial
        psd_sum = None
        for c in range(n_ch):
            f, pxx = welch(
                X_wb[i, c], fs=sfreq, nperseg=nperseg, axis=-1, detrend="linear"
            )
            if psd_sum is None:
                psd_sum = pxx.astype(np.float64)
                f_buf = f
            else:
                psd_sum += pxx
        assert f_buf is not None and psd_sum is not None
        psd_mean = psd_sum / max(1, n_ch)
        df = float(f_buf[1] - f_buf[0]) if len(f_buf) > 1 else 1.0
        for name, (lo, hi) in bands.items():
            m = (f_buf >= lo) & (f_buf <= hi)
            if not np.any(m):
                out[name][i] = -np.inf
                continue
            # Trapezoidal integration in native scipy units
            try:
                pow_band = np.trapezoid(psd_mean[m], f_buf[m])
            except AttributeError:
                pow_band = np.trapz(psd_mean[m], f_buf[m])
            out[name][i] = np.log(max(pow_band, 1e-30) + 1e-30)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mu vs beta band separability on AASD NPZ (Cohen's d)."
    )
    ap.add_argument("--data-root", type=Path, default=Path("data/aasd/npz"))
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--sfreq", type=float, default=128.0)
    ap.add_argument(
        "--wide-lf",
        type=float,
        default=0.1,
        help="Wide band low (match Riemannian baseline config)",
    )
    ap.add_argument(
        "--wide-hf",
        type=float,
        default=45.0,
        help="Wide band high (match Riemannian baseline config)",
    )
    ap.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap (stratified subsample) for speed.",
    )
    args = ap.parse_args()

    bands = {
        "mu_7_13": (7.0, 13.0),
        "beta_14_30": (14.0, 30.0),
        "low_0.1_6.9": (0.1, 6.9),
        "hi_13_45": (13.0, 45.0),
    }

    print(
        "Preprocessing: CAR (common average) + wide bandpass "
        f"[{args.wide_lf}, {args.wide_hf}] Hz (same family as experiment).\n"
        "Feature per trial: mean-over-channels log integral of Welch PSD in each band.\n"
        "Cohen's d > 0 means class-0 mean > class-1 mean (sign arbitrary); |d| is effect size.\n"
    )

    for sid in args.subjects:
        path = args.data_root / f"subject_{sid}.npz"
        if not path.is_file():
            print(f"[missing] {path}\n")
            continue
        z = np.load(path, allow_pickle=True)
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"]).astype(int).ravel()
        z.close()

        if len(np.unique(y)) < 2:
            print(f"Subject {sid}: single class, skip.\n")
            continue

        if args.max_trials is not None and X.shape[0] > args.max_trials:
            n = int(args.max_trials)
            rng = np.random.RandomState(42 + sid)
            idx0 = np.flatnonzero(y == np.unique(y)[0])
            idx1 = np.flatnonzero(y == np.unique(y)[1])
            half = n // 2
            take0 = rng.choice(idx0, size=min(half, idx0.size), replace=False)
            take1 = rng.choice(idx1, size=min(n - take0.size, idx1.size), replace=False)
            sel = np.sort(np.concatenate([take0, take1]))
            X = X[sel]
            y = y[sel]

        X_car = _average_reference_trials(X)
        X_wb = _bandpass(X_car, args.sfreq, args.wide_lf, args.wide_hf)

        feats = _log_band_powers_per_trial(X_wb, args.sfreq, bands)
        y0 = y == np.unique(y)[0]
        y1 = y == np.unique(y)[1]

        print(f"=== Subject {sid}  (n={len(y)}) ===")
        for name in bands:
            d = abs(_cohen_d(feats[name][y0], feats[name][y1]))
            print(f"  |Cohen's d| ({name:14s}): {d:.4f}")
        # Which band wins?
        scored = [(abs(_cohen_d(feats[n][y0], feats[n][y1])), n) for n in bands]
        scored.sort(reverse=True)
        print(f"  largest |d|: {scored[0][1]} ({scored[0][0]:.4f})")
        mu_d = abs(_cohen_d(feats["mu_7_13"][y0], feats["mu_7_13"][y1]))
        beta_d = abs(_cohen_d(feats["beta_14_30"][y0], feats["beta_14_30"][y1]))
        if beta_d > mu_d * 1.05:
            print(
                "  → Beta band power separates classes at least as strongly as mu here; "
                "a 7–13 Hz *decoder* discards that variance (expected to hurt vs wideband)."
            )
        print()


if __name__ == "__main__":
    main()
