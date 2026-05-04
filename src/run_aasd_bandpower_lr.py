"""Fast AASD log-bandpower + logistic-regression benchmark.

This is a cheap sanity benchmark for the AASD paper question: it uses
interpretable log-bandpower features and strict grouped-trial CV by default,
so all 60 one-second windows from a 60 s trial stay in the same fold.

Example:
  python3 -m src.run_aasd_bandpower_lr \
    --data-root data/aasd/npz \
    --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
    --out results/aasd_bandpower_lr_full18sub/subject_level_performance.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BANDS = {
    "low_0p1_6p9": (0.1, 6.9),
    "mu_7_13": (7.0, 13.0),
    "beta_14_30": (14.0, 30.0),
    "hi_30_45": (30.0, 45.0),
}


def _load_subject(data_root: Path, subject: int) -> tuple[np.ndarray, np.ndarray, float]:
    path = data_root / f"subject_{subject}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64).ravel()
        sfreq = float(np.asarray(z["sfreq"]).item())
    return X, y, sfreq


def _bandpower_features(X: np.ndarray, sfreq: float) -> np.ndarray:
    freqs, psd = welch(X, fs=sfreq, nperseg=min(128, X.shape[-1]), axis=-1)
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs <= hi)
        feats.append(np.log(psd[:, :, mask].mean(axis=-1) + 1e-12))
    return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


def _cv_splits(y: np.ndarray, n_splits: int, random_state: int, grouped: bool):
    if grouped and y.size >= 120 and y.size % 60 == 0:
        n_trials = y.size // 60
        y_blocks = y.reshape(n_trials, 60)
        y_trials = np.array([np.bincount(block).argmax() for block in y_blocks])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for tr_trials, te_trials in cv.split(np.zeros((n_trials, 1)), y_trials):
            tr = np.concatenate([np.arange(t * 60, (t + 1) * 60) for t in tr_trials])
            te = np.concatenate([np.arange(t * 60, (t + 1) * 60) for t in te_trials])
            yield tr, te
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        yield from cv.split(np.zeros((y.size, 1)), y)


def _run_one_subject(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_splits: int,
    random_state: int,
    grouped: bool,
    c: float,
) -> list[dict]:
    F = _bandpower_features(X, sfreq)
    rows = []
    for fold, (tr, te) in enumerate(_cv_splits(y, n_splits, random_state, grouped)):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=float(c)),
        )
        clf.fit(F[tr], y[tr])
        proba = clf.predict_proba(F[te])[:, 1]
        pred = (proba >= 0.5).astype(np.int64)
        rows.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(y[te], pred)),
                "auc": float(roc_auc_score(y[te], proba)),
                "f1": float(f1_score(y[te], pred)),
                "correct": int(np.sum(pred == y[te])),
                "total": int(te.size),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/aasd/npz"))
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--c", type=float, default=0.3)
    ap.add_argument(
        "--window-cv",
        action="store_true",
        help="Use random window CV instead of strict 60-window trial grouping.",
    )
    args = ap.parse_args()

    grouped = not bool(args.window_cv)
    all_rows = []
    for subject in args.subjects:
        X, y, sfreq = _load_subject(args.data_root, subject)
        fold_rows = _run_one_subject(
            X,
            y,
            sfreq,
            n_splits=args.n_splits,
            random_state=args.random_state,
            grouped=grouped,
            c=args.c,
        )
        mean_acc = float(np.mean([r["accuracy"] for r in fold_rows]))
        mean_auc = float(np.mean([r["auc"] for r in fold_rows]))
        mean_f1 = float(np.mean([r["f1"] for r in fold_rows]))
        print(
            f"Subject {subject:02d} | bandpower_lr | "
            f"acc={mean_acc:.3f} auc={mean_auc:.3f} f1={mean_f1:.3f}"
        )
        for r in fold_rows:
            all_rows.append(
                {
                    "subject": subject,
                    "backbone": "bandpower_lr",
                    "pipeline": "baseline",
                    "protocol": "grouped_trial_cv" if grouped else "window_cv",
                    **r,
                    "mean_accuracy": mean_acc,
                    "mean_auc": mean_auc,
                    "mean_f1": mean_f1,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
