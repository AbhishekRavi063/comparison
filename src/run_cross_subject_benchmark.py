from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .backbones.csp import _compute_csp_filters, _project_csp
from .backbones.tangent_space import (
    _covariance_matrices,
    _riemannian_mean,
    _tangent_space_projection,
)
from .config import ExperimentConfig
from .denoising.pipelines import bandpass_filter, apply_gedai, apply_icalabel
from .evaluation.metrics import paired_permutation_p_value, delong_auc_p_value
from .io.dataset import NpzMotorImageryDataset


PIPELINE_MAP = {
    "A": "tangent_baseline",
    "B": "tangent_icalabel",
    "C": "tangent_gedai",
    "D": "csp_baseline",
}


@dataclass
class FoldData:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    sfreq: float
    ch_names: List[str]


def _load_subjects(
    dataset: NpzMotorImageryDataset,
    subject_ids: Sequence[int],
    float_dtype: str = "float32",
) -> FoldData:
    """Load subjects one-by-one, cast to float32, then concatenate."""
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    sid_list: List[np.ndarray] = []
    sfreq = None
    ch_names = None
    dtype = np.dtype(float_dtype)

    for sid in subject_ids:
        subj = dataset._load_subject_file(int(sid))
        x_i = np.asarray(subj.X, dtype=dtype)
        y_i = np.asarray(subj.y, dtype=int)
        s_i = np.full(shape=(x_i.shape[0],), fill_value=int(sid), dtype=int)
        x_list.append(x_i)
        y_list.append(y_i)
        sid_list.append(s_i)
        if sfreq is None:
            sfreq = subj.sfreq
        if ch_names is None:
            ch_names = subj.ch_names
        del subj, x_i, y_i, s_i
        gc.collect()

    X = np.concatenate(x_list, axis=0).astype(dtype, copy=False)
    y = np.concatenate(y_list, axis=0).astype(int, copy=False)
    subject_ids_per_trial = np.concatenate(sid_list, axis=0).astype(int, copy=False)
    del x_list, y_list, sid_list
    gc.collect()

    return FoldData(
        X=X,
        y=y,
        subject_ids=subject_ids_per_trial,
        sfreq=float(sfreq),
        ch_names=list(ch_names),
    )


def _apply_denoising(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    l_freq: float,
    h_freq: float,
    mode: str,
) -> np.ndarray:
    X_bp = bandpass_filter(X, sfreq, l_freq, h_freq).astype(np.float32, copy=False)
    if mode == "baseline":
        return X_bp
    if mode == "icalabel":
        return apply_icalabel(X_bp, sfreq, ch_names, l_freq, h_freq).astype(
            np.float32, copy=False
        )
    if mode == "gedai":
        return apply_gedai(X_bp, sfreq, ch_names).astype(np.float32, copy=False)
    raise ValueError(f"Unknown denoising mode: {mode}")


def _fit_predict_tangent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cov_train = _covariance_matrices(X_train)
    cov_test = _covariance_matrices(X_test)
    c_ref = _riemannian_mean(cov_train)
    xtr = _tangent_space_projection(cov_train, c_ref)
    xte = _tangent_space_projection(cov_test, c_ref)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(xtr, y_train)
    y_pred = clf.predict(xte)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(xte)
    else:
        dec = clf.decision_function(xte)
        if dec.ndim == 1:
            y_score = np.column_stack([1.0 - dec, dec])
        else:
            y_score = dec
    return y_pred, y_score


def _fit_predict_csp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    w = _compute_csp_filters(X_train, y_train)
    xtr = _project_csp(X_train, w)
    xte = _project_csp(X_test, w)
    clf = LinearDiscriminantAnalysis()
    clf.fit(xtr, y_train)
    y_pred = clf.predict(xte)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(xte)
    else:
        dec = clf.decision_function(xte)
        if dec.ndim == 1:
            y_score = np.column_stack([1.0 - dec, dec])
        else:
            y_score = dec
    return y_pred, y_score


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        return float("nan")
    try:
        if y_score.ndim == 1 or y_score.shape[1] == 1:
            return float(roc_auc_score(y_true, y_score))
        if y_score.shape[1] == 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _paired_cohen_d(scores1: Sequence[float], scores2: Sequence[float]) -> float:
    d = np.asarray(scores1, dtype=float) - np.asarray(scores2, dtype=float)
    sd = np.std(d, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def run_cross_subject_benchmark(
    cfg: ExperimentConfig,
    pipelines: Sequence[str],
    stream_subjects: bool = True,
) -> None:
    if not stream_subjects:
        raise ValueError("This benchmark requires --stream-subjects for memory safety.")

    dataset = NpzMotorImageryDataset(
        data_root=cfg.data_root,
        subjects=cfg.subjects,
        float_dtype=cfg.memory.float_dtype,
    )
    subjects = np.array(cfg.subjects, dtype=int)
    n_splits = min(5, len(subjects))
    if n_splits < 2:
        raise ValueError("Need at least 2 subjects for cross-subject GroupKFold.")

    results_root = Path(cfg.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    # GroupKFold over subject IDs (groups are identical to samples here)
    gkf = GroupKFold(n_splits=n_splits)
    subj_index = np.arange(len(subjects))
    splits = list(gkf.split(subj_index, groups=subjects))
    split_file = results_root / "groupkfold_splits.json"
    split_payload = []
    for fold_id, (tr_idx, te_idx) in enumerate(splits):
        tr = subjects[tr_idx].tolist()
        te = subjects[te_idx].tolist()
        if set(tr).intersection(set(te)):
            raise RuntimeError(f"Fold {fold_id}: train/test subject overlap detected.")
        split_payload.append(
            {"fold_id": int(fold_id), "train_subjects": tr, "test_subjects": te}
        )
    split_file.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    fold_rows: List[Dict] = []
    trial_rows: List[Dict] = []

    for fold_id, (tr_idx, te_idx) in enumerate(splits):
        train_subjects = subjects[tr_idx].tolist()
        test_subjects = subjects[te_idx].tolist()

        train_data = _load_subjects(dataset, train_subjects, cfg.memory.float_dtype)
        test_data = _load_subjects(dataset, test_subjects, cfg.memory.float_dtype)

        # Use subject-independent preprocessing per pipeline; same fold split for all pipelines.
        for p in pipelines:
            pname = PIPELINE_MAP[p]
            if pname == "tangent_baseline":
                xtr = _apply_denoising(
                    train_data.X,
                    train_data.sfreq,
                    train_data.ch_names,
                    cfg.bandpass.l_freq,
                    cfg.bandpass.h_freq,
                    "baseline",
                )
                xte = _apply_denoising(
                    test_data.X,
                    test_data.sfreq,
                    test_data.ch_names,
                    cfg.bandpass.l_freq,
                    cfg.bandpass.h_freq,
                    "baseline",
                )
                y_pred, y_score = _fit_predict_tangent(xtr, train_data.y, xte)
            elif pname == "tangent_icalabel":
                xtr = _apply_denoising(
                    train_data.X,
                    train_data.sfreq,
                    train_data.ch_names,
                    cfg.bandpass.l_freq,
                    cfg.bandpass.h_freq,
                    "icalabel",
                )
                xte = _apply_denoising(
                    test_data.X,
                    test_data.sfreq,
                    test_data.ch_names,
                    cfg.bandpass.l_freq,
                    cfg.bandpass.h_freq,
                    "icalabel",
                )
                y_pred, y_score = _fit_predict_tangent(xtr, train_data.y, xte)
            elif pname == "tangent_gedai":
                xtr = _apply_denoising(
                    train_data.X,
                    train_data.sfreq,
                    train_data.ch_names,
                    cfg.bandpass.l_freq,
                    cfg.bandpass.h_freq,
                    "gedai",
                )
                xte = _apply_denoising(
                    test_data.X,
                    test_data.sfreq,
                    test_data.ch_names,
                    cfg.bandpass.l_freq,
                    cfg.bandpass.h_freq,
                    "gedai",
                )
                y_pred, y_score = _fit_predict_tangent(xtr, train_data.y, xte)
            elif pname == "csp_baseline":
                xtr = _apply_denoising(
                    train_data.X,
                    train_data.sfreq,
                    train_data.ch_names,
                    cfg.bandpass.l_freq,
                    cfg.bandpass.h_freq,
                    "baseline",
                )
                xte = _apply_denoising(
                    test_data.X,
                    test_data.sfreq,
                    test_data.ch_names,
                    cfg.bandpass.l_freq,
                    cfg.bandpass.h_freq,
                    "baseline",
                )
                y_pred, y_score = _fit_predict_csp(xtr, train_data.y, xte)
            else:
                raise ValueError(f"Unsupported pipeline key: {pname}")

            acc = float(accuracy_score(test_data.y, y_pred))
            kappa = float(cohen_kappa_score(test_data.y, y_pred))
            auc_macro = _safe_auc(test_data.y, y_score)
            fold_rows.append(
                {
                    "fold_id": int(fold_id),
                    "pipeline": pname,
                    "accuracy": acc,
                    "auc_macro": auc_macro,
                    "kappa": kappa,
                }
            )

            for i in range(len(test_data.y)):
                trial_rows.append(
                    {
                        "fold_id": int(fold_id),
                        "pipeline": pname,
                        "subject_id": int(test_data.subject_ids[i]),
                        "y_true": int(test_data.y[i]),
                        "y_pred": int(y_pred[i]),
                        "y_score": float(y_score[i, 1] if y_score.ndim > 1 and y_score.shape[1] >= 2 else y_score[i]),
                    }
                )

            del xtr, xte, y_pred, y_score
            gc.collect()

        del train_data, test_data
        gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(results_root / "fold_metrics.csv", index=False)

    trial_df = pd.DataFrame(trial_rows)
    subject_rows: List[Dict] = []
    for (sid, pname), g in trial_df.groupby(["subject_id", "pipeline"]):
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        y_score = g["y_score"].to_numpy()
        subject_rows.append(
            {
                "subject_id": int(sid),
                "pipeline": str(pname),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
                "kappa": float(cohen_kappa_score(y_true, y_pred)),
            }
        )
    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(results_root / "subject_metrics.csv", index=False)

    # Statistical summaries
    stats: Dict[str, object] = {}
    stats["mean_accuracy"] = {
        p: float(v) for p, v in fold_df.groupby("pipeline")["accuracy"].mean().to_dict().items()
    }
    stats["std_accuracy"] = {
        p: float(v) for p, v in fold_df.groupby("pipeline")["accuracy"].std(ddof=1).to_dict().items()
    }
    stats["mean_auc"] = {
        p: float(v) for p, v in fold_df.groupby("pipeline")["auc_macro"].mean().to_dict().items()
    }
    stats["mean_kappa"] = {
        p: float(v) for p, v in fold_df.groupby("pipeline")["kappa"].mean().to_dict().items()
    }

    pvals: Dict[str, float] = {}
    pvals_auc: Dict[str, float] = {}
    effects: Dict[str, float] = {}
    for p1, p2 in [
        ("tangent_gedai", "tangent_baseline"),
        ("tangent_icalabel", "tangent_baseline"),
        ("tangent_gedai", "tangent_icalabel"),
    ]:
        if p1 not in fold_df["pipeline"].values or p2 not in fold_df["pipeline"].values:
            continue
        x1_acc = (
            fold_df[fold_df["pipeline"] == p1]
            .sort_values("fold_id")["accuracy"]
            .to_numpy()
            .astype(float)
        )
        x2_acc = (
            fold_df[fold_df["pipeline"] == p2]
            .sort_values("fold_id")["accuracy"]
            .to_numpy()
            .astype(float)
        )
        if len(x1_acc) != len(x2_acc):
            continue
        key = f"{p1}_vs_{p2}"
        pvals[key] = paired_permutation_p_value(x1_acc, x2_acc, cfg.permutation.n_pipeline_level)
        effects[key] = _paired_cohen_d(x1_acc, x2_acc)
        
        # DeLong AUC
        t1 = trial_df[trial_df["pipeline"] == p1].sort_values(["fold_id", "subject_id"]).reset_index(drop=True)
        t2 = trial_df[trial_df["pipeline"] == p2].sort_values(["fold_id", "subject_id"]).reset_index(drop=True)
        
        if len(t1) == len(t2) and len(t1) > 0 and (t1["y_true"].values == t2["y_true"].values).all():
            pvals_auc[key] = delong_auc_p_value(
                t1["y_true"].values,
                t1["y_score"].values,
                t2["y_score"].values
            )
            
    stats["p_values"] = pvals
    stats["p_values_auc_delong"] = pvals_auc
    stats["effect_sizes"] = effects

    if (
        "tangent_gedai" in subject_df["pipeline"].values
        and "tangent_baseline" in subject_df["pipeline"].values
    ):
        s_g = subject_df[subject_df["pipeline"] == "tangent_gedai"].set_index("subject_id")[
            "accuracy"
        ]
        s_b = subject_df[subject_df["pipeline"] == "tangent_baseline"].set_index("subject_id")[
            "accuracy"
        ]
        common = s_g.index.intersection(s_b.index)
        delta = s_g.loc[common] - s_b.loc[common]
        stats["subjects_improved_percentage"] = float(100.0 * (delta > 0).sum() / len(delta))
    else:
        stats["subjects_improved_percentage"] = float("nan")

    (results_root / "stats_summary.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )


def _parse_pipelines(s: str) -> List[str]:
    vals = [x.strip().upper() for x in s.split(",") if x.strip()]
    for v in vals:
        if v not in PIPELINE_MAP:
            raise ValueError(f"Unsupported pipeline code: {v}. Use A,B,C,D.")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-subject EEG denoising benchmark with GroupKFold streaming.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_real_physionet_full_fast.yml",
    )
    parser.add_argument(
        "--stream-subjects",
        action="store_true",
        help="Required safety flag: stream subjects fold-by-fold.",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="A,B,C,D",
        help="Comma-separated pipeline codes from {A,B,C,D}.",
    )
    args = parser.parse_args()

    # Professor requirement: force GEDAI CPU mode for memory stability.
    os.environ.setdefault("PYGEDAI_FORCE_CPU", "1")

    cfg = ExperimentConfig.from_yaml(args.config)
    pipelines = _parse_pipelines(args.pipelines)
    run_cross_subject_benchmark(cfg, pipelines=pipelines, stream_subjects=args.stream_subjects)


if __name__ == "__main__":
    main()
