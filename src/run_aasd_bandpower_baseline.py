"""Fast AASD all-subject spectral baseline.

This is the cheap first-pass check before expensive DL runs. It evaluates a
log-bandpower + logistic-regression classifier with either strict grouped-trial
CV (paper-compatible) or random window CV (diagnostic only).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COARSE_BANDS = {
    "low_0p1_6p9": (0.1, 6.9),
    "mu_7_13": (7.0, 13.0),
    "beta_14_30": (14.0, 30.0),
    "hi_30_45": (30.0, 45.0),
}

FINE_BANDS = {
    "delta_0p1_3p9": (0.1, 3.9),
    "theta_4_6p9": (4.0, 6.9),
    "low_mu_7_10": (7.0, 10.0),
    "high_mu_10_13": (10.0, 13.0),
    "low_beta_14_20": (14.0, 20.0),
    "high_beta_20_30": (20.0, 30.0),
    "gamma_30_45": (30.0, 45.0),
}


def _features(
    X: np.ndarray,
    sfreq: float,
    use_car: bool,
    bands: dict[str, tuple[float, float]],
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if use_car:
        X = X - X.mean(axis=1, keepdims=True)
    freqs, psd = welch(X, fs=sfreq, nperseg=min(128, X.shape[-1]), axis=-1)
    feats = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs <= hi)
        feats.append(np.log(psd[:, :, mask].mean(axis=-1) + 1e-12))
    return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


def _splits(y: np.ndarray, protocol: str, n_splits: int, seed: int):
    y = np.asarray(y).astype(int).ravel()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if protocol == "window":
        return list(cv.split(np.zeros((len(y), 1)), y))
    if len(y) % 60 != 0:
        raise ValueError(f"Grouped AASD CV requires n_windows divisible by 60, got {len(y)}")
    n_trials = len(y) // 60
    y_blocks = y.reshape(n_trials, 60)
    y_trials = np.array([np.bincount(block).argmax() for block in y_blocks], dtype=int)
    out = []
    for tr_trials, te_trials in cv.split(np.zeros((n_trials, 1)), y_trials):
        tr = np.concatenate([np.arange(t * 60, (t + 1) * 60) for t in tr_trials])
        te = np.concatenate([np.arange(t * 60, (t + 1) * 60) for t in te_trials])
        out.append((tr, te))
    return out


def run_subject(
    path: Path,
    subject: int,
    protocol: str,
    use_car: bool,
    n_splits: int,
    seed: int,
    bands: dict[str, tuple[float, float]],
    C: float,
):
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float32)
    y = np.asarray(z["y"]).astype(int).ravel()
    sfreq = float(z["sfreq"]) if "sfreq" in z else 128.0
    z.close()

    F = _features(X, sfreq=sfreq, use_car=use_car, bands=bands)
    rows = []
    correct = 0
    total = 0
    for fold, (tr, te) in enumerate(_splits(y, protocol, n_splits, seed)):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=float(C)),
        )
        clf.fit(F[tr], y[tr])
        prob = clf.predict_proba(F[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        acc = float(accuracy_score(y[te], pred))
        auc = float(roc_auc_score(y[te], prob))
        f1 = float(f1_score(y[te], pred))
        correct += int(np.sum(pred == y[te]))
        total += int(len(te))
        rows.append(
            {
                "subject": subject,
                "fold": fold,
                "accuracy": acc,
                "auc": auc,
                "f1": f1,
                "n_test": int(len(te)),
            }
        )
    mean_acc = float(np.mean([r["accuracy"] for r in rows]))
    mean_auc = float(np.mean([r["auc"] for r in rows]))
    mean_f1 = float(np.mean([r["f1"] for r in rows]))
    for row in rows:
        row.update(
            {
                "protocol": protocol,
                "use_car": bool(use_car),
                "bands_preset": "custom",
                "n_bands": int(len(bands)),
                "C": float(C),
                "mean_accuracy": mean_acc,
                "mean_auc": mean_auc,
                "mean_f1": mean_f1,
                "pooled_correct": correct,
                "pooled_total": total,
                "class1_rate": float(np.mean(y)),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fast AASD log-bandpower LR baseline.")
    ap.add_argument("--data-root", type=Path, default=Path("data/aasd/npz"))
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 19)))
    ap.add_argument("--out", type=Path, default=Path("results/aasd_bandpower_lr_full18sub/tables/subject_level_performance.csv"))
    ap.add_argument("--protocol", choices=["grouped", "window"], default="grouped")
    ap.add_argument("--bands-preset", choices=["coarse", "fine"], default="coarse")
    ap.add_argument("--C", type=float, default=0.3, help="LogisticRegression inverse regularization strength.")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-car", action="store_true", help="Use NPZ data as-is instead of applying common average reference.")
    args = ap.parse_args()

    all_rows = []
    bands = FINE_BANDS if args.bands_preset == "fine" else COARSE_BANDS
    for subject in args.subjects:
        path = args.data_root / f"subject_{subject}.npz"
        if not path.is_file():
            print(f"[missing] {path}")
            continue
        rows = run_subject(
            path,
            subject=subject,
            protocol=args.protocol,
            use_car=not args.no_car,
            n_splits=args.n_splits,
            seed=args.seed,
            bands=bands,
            C=args.C,
        )
        for row in rows:
            row["bands_preset"] = args.bands_preset
        all_rows.extend(rows)
        print(
            f"Subject {subject:02d} | acc={rows[0]['mean_accuracy']:.3f} "
            f"auc={rows[0]['mean_auc']:.3f} f1={rows[0]['mean_f1']:.3f}"
        )

    if not all_rows:
        raise FileNotFoundError(f"No subject files found under {args.data_root}")
    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    subj = df.groupby("subject", as_index=False).agg(
        mean_accuracy=("mean_accuracy", "first"),
        mean_auc=("mean_auc", "first"),
        mean_f1=("mean_f1", "first"),
    )
    print("\nSummary across subjects")
    print(subj[["mean_accuracy", "mean_auc", "mean_f1"]].agg(["mean", "std", "min", "max"]))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
