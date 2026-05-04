from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm


def _apply_max_trials_smoke(
    cfg: "ExperimentConfig",
    X: np.ndarray,
    y: np.ndarray,
    subject_id: int,
    log: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """When ``max_trials`` is set: cap size for fast smoke runs.

    - **Many-class** data (e.g. Alljoined image categories): take a **balanced binary**
      subset using the two most frequent labels (``n`` trials per class,
      ``n = min(max_trials // 2, counts…)``, each side at least ``cv.n_splits``).
    - **Binary** data: stratified subsample to ``max_trials`` when possible.
    """
    max_trials = cfg.max_trials
    if max_trials is None or max_trials <= 0:
        return X, y

    y = np.asarray(y)
    X = np.asarray(X)
    n = int(X.shape[0])
    n_splits = int(cfg.cv.n_splits)
    rs = int(cfg.cv.random_state) + int(subject_id) * 1_000_003
    rng = np.random.RandomState(rs)

    # AASD-specific smoke sampling:
    # keep full 1s-window blocks per original 60s trial (60 windows/trial) so
    # denoisers like GEDAI still see contiguous trial dynamics.
    if cfg.dataset_label and "aasd" in str(cfg.dataset_label).lower() and n >= 120 and n % 60 == 0:
        n_trials = n // 60
        y_trials = y.reshape(n_trials, 60)[:, 0]
        u, c = np.unique(y_trials, return_counts=True)
        if len(u) == 2:
            # Keep equal trials/class, respecting max_trials budget in windows.
            max_trials_even = max(120, int(max_trials // 120) * 120)
            trials_budget = max_trials_even // 60
            per_cls = max(n_splits, min(trials_budget // 2, int(np.min(c))))
            if per_cls * 2 <= n_trials:
                idx_blocks = []
                for cls in sorted(u.tolist()):
                    t_idx = np.flatnonzero(y_trials == cls)
                    rng.shuffle(t_idx)
                    for t in t_idx[:per_cls]:
                        idx_blocks.append(np.arange(t * 60, (t + 1) * 60))
                idx = np.concatenate(idx_blocks)
                idx.sort()
                log.info(
                    f"Subject {subject_id}: AASD trial-aware smoke subset "
                    f"({2*per_cls} trials, {len(idx)} windows, max_trials={max_trials})."
                )
                return X[idx].copy(), y[idx].copy()

    if len(np.unique(y)) > 2:
        strategy = str(getattr(cfg, "max_trials_strategy", "binary_for_multiclass"))
        uni, counts = np.unique(y, return_counts=True)
        order = np.argsort(-counts)

        if strategy == "binary_for_multiclass":
            c0 = uni[order[0]]
            c1 = uni[order[1]]
            i0 = np.flatnonzero(y == c0)
            i1 = np.flatnonzero(y == c1)
            half_budget = max_trials // 2
            per = min(half_budget, len(i0), len(i1))
            if per < n_splits:
                raise ValueError(
                    f"Subject {subject_id}: max_trials={max_trials} is too small for "
                    f"cv.n_splits={n_splits} on many-class data. After picking the two "
                    f"most frequent classes ({c0}, {c1}), each has at least "
                    f"{per} usable trials; need >= n_splits per class. "
                    f"Increase max_trials (e.g. {2 * n_splits * 20}) or lower cv.n_splits."
                )
            rng.shuffle(i0)
            rng.shuffle(i1)
            idx = np.concatenate([i0[:per], i1[:per]])
            rng.shuffle(idx)
            y_bin = (y[idx] == c1).astype(np.int64)
            log.info(
                f"Subject {subject_id}: smoke — binary subset (most frequent classes "
                f"{c0} vs {c1}), {len(idx)} trials (max_trials={max_trials})."
            )
            return X[idx].copy(), y_bin

        if strategy != "multiclass_topk":
            raise ValueError(
                f"Subject {subject_id}: unknown max_trials_strategy='{strategy}'. "
                "Use 'binary_for_multiclass' or 'multiclass_topk'."
            )

        # Balanced multi-class subset across the most frequent K classes.
        # Choose largest K (optionally capped) such that per-class >= n_splits
        # and total <= max_trials.
        sorted_classes = uni[order]
        sorted_counts = counts[order]
        k_cap = int(getattr(cfg, "max_trials_top_k", 0) or 0)
        k_max = min(len(sorted_classes), k_cap) if k_cap > 0 else len(sorted_classes)
        chosen_k = None
        chosen_per = None
        for k in range(k_max, 1, -1):
            per_class = min(int(max_trials // k), int(np.min(sorted_counts[:k])))
            if per_class >= n_splits:
                chosen_k = k
                chosen_per = per_class
                break

        if chosen_k is None or chosen_per is None:
            raise ValueError(
                f"Subject {subject_id}: max_trials={max_trials} too small for "
                f"cv.n_splits={n_splits} in multiclass_topk mode. Increase max_trials, "
                f"lower cv.n_splits, or reduce max_trials_top_k."
            )

        picked_classes = sorted_classes[:chosen_k]
        idx_parts = []
        for c in picked_classes:
            idx_c = np.flatnonzero(y == c)
            rng.shuffle(idx_c)
            idx_parts.append(idx_c[:chosen_per])
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
        log.info(
            f"Subject {subject_id}: smoke — multiclass_topk subset "
            f"(K={chosen_k}, per_class={chosen_per}, total={len(idx)}, "
            f"max_trials={max_trials})."
        )
        return X[idx].copy(), y[idx].copy()

    if n <= max_trials:
        return X, y

    # AASD trial-aware subsampling: the GEDAI path reconstructs 60s trials from
    # consecutive 1s windows.  Randomly picking individual windows breaks that
    # assumption (groups of 60 random windows ≠ one coherent trial).
    # For AASD we subsample whole 60-window trial blocks so every reconstructed
    # trial is temporally contiguous.
    if cfg.dataset_label and "aasd" in str(cfg.dataset_label).lower():
        WIN_PER_TRIAL_AASD = 60  # 60 s × 250 Hz / 250 samples
        n_full_trials = n // WIN_PER_TRIAL_AASD
        if n_full_trials >= 2 * n_splits:
            # Build per-trial labels (majority vote).
            y_trials = np.array(
                [
                    int(np.bincount(y[t * WIN_PER_TRIAL_AASD:(t + 1) * WIN_PER_TRIAL_AASD].astype(int)).argmax())
                    for t in range(n_full_trials)
                ],
                dtype=np.int64,
            )
            # Stratified trial subsample.
            n_trial_cap = max(2 * n_splits, max_trials // WIN_PER_TRIAL_AASD)
            n_trial_cap = min(n_trial_cap, n_full_trials)
            rng2 = np.random.RandomState(rs + 7)
            cls_vals = np.unique(y_trials)
            per_cls = max(n_splits, n_trial_cap // max(len(cls_vals), 1))
            chosen = []
            for cv in cls_vals:
                idx_c = np.where(y_trials == cv)[0]
                rng2.shuffle(idx_c)
                chosen.append(idx_c[:per_cls])
            trial_idx = np.sort(np.concatenate(chosen))
            # Expand back to window indices (keep temporal order).
            win_idx = np.concatenate(
                [np.arange(t * WIN_PER_TRIAL_AASD, (t + 1) * WIN_PER_TRIAL_AASD) for t in trial_idx]
            )
            log.info(
                f"Subject {subject_id}: AASD smoke — {len(trial_idx)} complete trials "
                f"({len(win_idx)} windows) selected (max_trials={max_trials})."
            )
            return X[win_idx].copy(), y[win_idx].copy()

    train_size = min(max_trials, n)
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=train_size, random_state=rs
        )
        train_idx, _ = next(sss.split(np.zeros((n, 1)), y))
    except ValueError as exc:
        log.warning(
            f"Subject {subject_id}: stratified subsample to {train_size} failed ({exc}); "
            f"using first {train_size} trials."
        )
        return X[:train_size].copy(), y[:train_size].copy()
    log.info(
        f"Subject {subject_id}: using {train_size} of {n} trials (max_trials cap)."
    )
    return X[train_idx].copy(), y[train_idx].copy()


def _log_memory_if_debug(log) -> None:
    """If EEG_MEMORY_DEBUG=1, log current process RSS (MB). Requires psutil."""
    if os.environ.get("EEG_MEMORY_DEBUG", "").strip() != "1":
        return
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 ** 2)
        log.info(f"RAM (MB): {rss_mb:.1f}")
    except Exception:
        pass


def _lazy_transformer_backbone():
    """Load Transformer helpers only when needed (requires ``torch``)."""
    from ..backbones.transformer_eeg import (
        fit_transformer_model_preprocessed,
        run_transformer_cv_preprocessed,
    )

    return run_transformer_cv_preprocessed, fit_transformer_model_preprocessed


def _lazy_eegnet_backbone():
    """Load EEGNet helpers only when needed (requires ``torch``)."""
    from ..backbones.eegnet_eeg import (
        fit_eegnet_model_preprocessed,
        run_eegnet_cv_preprocessed,
    )

    return run_eegnet_cv_preprocessed, fit_eegnet_model_preprocessed


from ..config import ExperimentConfig
from ..io.dataset import NpzMotorImageryDataset
from ..backbones.csp import run_csp_cv_preprocessed, fit_csp_model_preprocessed
from ..backbones.tangent_space import (
    build_tangent_features_for_splits,
    run_tangent_cv_precomputed_features,
    fit_tangent_model_preprocessed,
)
from ..backbones.time_domain_lda import (
    build_time_lda_features_for_splits,
    run_time_lda_cv_precomputed_features,
    fit_time_lda_model_preprocessed,
)
from ..denoising.pipelines import MRCP_MOTOR_CHANNELS, preprocess_subject_data
from ..data.dataset_noise_inspection import (
    plot_denoising_comparison_overlay,
    plot_denoising_psd_comparison,
)
from .metrics import (
    SubjectPerformance,
    binomial_vs_chance_p_value,
    empirical_chance_p_value,
    cohen_d_pooled,
    mann_whitney_pipeline_p_value,
    paired_permutation_p_value,
    wilcoxon_paired_pipeline_p_value,
    compute_band_power,
)

# Parietal channels used for alpha lateralization index.
_LEFT_PARIETAL  = ("P3", "P7", "CP5", "P5")
_RIGHT_PARIETAL = ("P4", "P8", "CP6", "P6")


def _compute_lateralization_index(
    X_proc: np.ndarray,
    y: np.ndarray,
    ch_names: Sequence,
) -> float:
    """Alpha lateralization index (LI) from already bandpass-filtered data.

    LI = mean[(alpha_ipsi - alpha_contra) / (alpha_ipsi + alpha_contra)]
    For auditory attention: y=0 (left attend) → ipsi=right, contra=left.
    LI > 0 confirms expected neural pattern. Higher LI after GEDAI = sharpened signal.
    """
    ch_upper = [str(c).upper() for c in ch_names]
    left_idx  = [ch_upper.index(c) for c in _LEFT_PARIETAL  if c in ch_upper]
    right_idx = [ch_upper.index(c) for c in _RIGHT_PARIETAL if c in ch_upper]
    if not left_idx or not right_idx:
        return 0.0
    power   = np.var(X_proc, axis=-1)
    left_p  = power[:, left_idx].mean(axis=1)
    right_p = power[:, right_idx].mean(axis=1)
    y = np.asarray(y).ravel()
    li_vals = []
    for yi, lp, rp in zip(y, left_p, right_p):
        ipsi, contra = (rp, lp) if int(yi) == 0 else (lp, rp)
        denom = ipsi + contra
        if denom > 1e-12:
            li_vals.append(float((ipsi - contra) / denom))
    return float(np.mean(li_vals)) if li_vals else 0.0


def _build_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    cfg: "ExperimentConfig",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build CV splits, keeping full AASD 60 s trials together when needed."""
    if (
        cfg.dataset_label
        and "aasd" in str(cfg.dataset_label).lower()
        and int(X.shape[0]) >= 120
        and int(X.shape[0]) % 60 == 0
        and bool(getattr(cfg.cv, "aasd_group_trials", True))
        and (
            getattr(cfg.backbones, "use_transformer", False)
            or getattr(cfg.backbones, "use_eegnet", False)
        )
    ):
        n_trials = int(X.shape[0]) // 60
        y_blocks = np.asarray(y).reshape(n_trials, 60)
        y_trials = np.array(
            [int(np.bincount(block.astype(int)).argmax()) for block in y_blocks],
            dtype=np.int64,
        )
        cv = StratifiedKFold(
            n_splits=cfg.cv.n_splits,
            shuffle=cfg.cv.shuffle,
            random_state=cfg.cv.random_state,
        )
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for tr_trials, te_trials in cv.split(np.zeros((n_trials, 1)), y_trials):
            tr_idx = np.concatenate(
                [np.arange(t * 60, (t + 1) * 60, dtype=int) for t in tr_trials]
            )
            te_idx = np.concatenate(
                [np.arange(t * 60, (t + 1) * 60, dtype=int) for t in te_trials]
            )
            splits.append((tr_idx, te_idx))
        return splits
    cv = StratifiedKFold(
        n_splits=cfg.cv.n_splits,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.cv.random_state,
    )
    return list(cv.split(X, y))


def _aasd_window_groups(
    X: np.ndarray,
    cfg: "ExperimentConfig",
) -> np.ndarray | None:
    """Return original-trial group ids for AASD 1-second windows when applicable."""
    if (
        cfg.dataset_label
        and "aasd" in str(cfg.dataset_label).lower()
        and int(X.shape[0]) >= 120
        and int(X.shape[0]) % 60 == 0
        and bool(getattr(cfg.cv, "aasd_group_trials", True))
    ):
        return np.arange(int(X.shape[0]), dtype=int) // 60
    return None


def _is_borderline_p_value(p_value: float, cfg: ExperimentConfig) -> bool:
    low = float(cfg.permutation.borderline_low)
    high = float(cfg.permutation.borderline_high)
    return low <= p_value <= high


def _score_time_lda_inner_cv(
    X_proc: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    target_sfreq: float | None,
    n_splits: int,
    random_state: int,
) -> float:
    y = np.asarray(y).ravel()
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2:
        return float("-inf")
    inner_splits = int(max(2, min(int(n_splits), int(np.min(counts)))))
    cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(cv.split(X_proc, y))
    feats = build_time_lda_features_for_splits(
        X_proc=X_proc,
        cv_splits=splits,
        sfreq=sfreq,
        target_sfreq=target_sfreq,
    )
    res = run_time_lda_cv_precomputed_features(
        y=y,
        cv_splits=splits,
        fold_features=feats,
    )
    return float(np.mean(res.fold_accuracies))


def _score_transformer_inner_cv(
    X_proc: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    target_sfreq: float | None,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
) -> float:
    """Mini-Transformer inner CV used as a same-architecture gate proxy.

    Uses 1 layer, 64 dim, 5 epochs so it's fast but still tracks transformer
    behavior much better than a tangent-space LDA proxy.
    """
    from ..backbones.transformer_eeg import run_transformer_cv_preprocessed

    y = np.asarray(y).ravel()
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2:
        return float("-inf")
    inner_splits = int(max(2, min(int(n_splits), int(np.min(counts)))))

    if groups is not None:
        groups = np.asarray(groups)
        unique_g = np.unique(groups)
        group_labels = np.array(
            [
                int(np.bincount(y[groups == g].astype(int)).argmax())
                for g in unique_g
            ],
            dtype=np.int64,
        )
        if len(np.unique(group_labels)) < 2 or unique_g.size < inner_splits:
            cv = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=random_state
            )
            splits = list(cv.split(X_proc, y))
        else:
            gcv = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=random_state
            )
            splits = []
            for tr_g_idx, te_g_idx in gcv.split(np.zeros((unique_g.size, 1)), group_labels):
                train_g = set(unique_g[tr_g_idx].tolist())
                test_g = set(unique_g[te_g_idx].tolist())
                tr_idx = np.array(
                    [i for i, g in enumerate(groups) if g in train_g], dtype=int
                )
                te_idx = np.array(
                    [i for i, g in enumerate(groups) if g in test_g], dtype=int
                )
                splits.append((tr_idx, te_idx))
    else:
        cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )
        splits = list(cv.split(X_proc, y))

    res = run_transformer_cv_preprocessed(
        X_proc=X_proc,
        y=y,
        cv_splits=splits,
        sfreq=sfreq,
        target_sfreq=target_sfreq,
        epochs=5,
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=1e-3,
        dropout=0.3,
        d_model=64,
        n_heads=4,
        n_layers=1,
        ff_dim=64,
        val_fraction=0.15,
        patience=3,
        device="cpu",
        random_state=random_state,
        groups=groups,
    )
    return float(np.mean(res.fold_accuracies))


def _score_eegnet_inner_cv(
    X_proc: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    target_sfreq: float | None,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
) -> float:
    """Fast EEGNet inner CV for the AASD GEDAI gate (few epochs, small F1)."""
    from ..backbones.eegnet_eeg import run_eegnet_cv_preprocessed

    y = np.asarray(y).ravel()
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2:
        return float("-inf")
    inner_splits = int(max(2, min(int(n_splits), int(np.min(counts)))))

    if groups is not None:
        groups = np.asarray(groups)
        unique_g = np.unique(groups)
        group_labels = np.array(
            [
                int(np.bincount(y[groups == g].astype(int)).argmax())
                for g in unique_g
            ],
            dtype=np.int64,
        )
        if len(np.unique(group_labels)) < 2 or unique_g.size < inner_splits:
            cv = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=random_state
            )
            splits = list(cv.split(X_proc, y))
        else:
            gcv = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=random_state
            )
            splits = []
            for tr_g_idx, te_g_idx in gcv.split(np.zeros((unique_g.size, 1)), group_labels):
                train_g = set(unique_g[tr_g_idx].tolist())
                test_g = set(unique_g[te_g_idx].tolist())
                tr_idx = np.array(
                    [i for i, g in enumerate(groups) if g in train_g], dtype=int
                )
                te_idx = np.array(
                    [i for i, g in enumerate(groups) if g in test_g], dtype=int
                )
                splits.append((tr_idx, te_idx))
    else:
        cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )
        splits = list(cv.split(X_proc, y))

    res = run_eegnet_cv_preprocessed(
        X_proc=X_proc,
        y=y,
        cv_splits=splits,
        sfreq=sfreq,
        target_sfreq=target_sfreq,
        epochs=8,
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=1e-3,
        dropout=0.25,
        F1=4,
        D=2,
        F2=8,
        kernel_length=None,
        val_fraction=0.15,
        patience=2,
        device="cpu",
        random_state=random_state,
        groups=groups,
        paper_exact=True,
    )
    return float(np.mean(res.fold_accuracies))


@dataclass
class ExperimentResult:
    subject_performances: List[SubjectPerformance]
    pipeline_comparisons: pd.DataFrame


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run the complete experiment over all subjects and pipelines.

    Per subject, pipelines are run in order: baseline, ICALabel, then GEDAI.
    Before GEDAI we call gc.collect() so ICA/Raw temporaries from ICALabel are
    freed, reducing peak RAM when GEDAI runs.
    """
    rng = np.random.RandomState(cfg.cv.random_state)
    np.random.set_state(rng.get_state())

    results_root = cfg.results_root
    tables_dir = results_root / "tables"
    stats_dir = results_root / "stats"
    models_dir = results_root / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    if cfg.memory.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    if cfg.memory.save_denoised_npz:
        (results_root / cfg.memory.denoised_subdir).mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("run_full_test")
    npz_pattern = "subject_{id}.npz"
    data_root_p = Path(cfg.data_root)
    subjects_present = [
        s for s in cfg.subjects if (data_root_p / npz_pattern.format(id=s)).is_file()
    ]
    missing_npz = [s for s in cfg.subjects if s not in subjects_present]
    if missing_npz:
        log.warning(
            "Skipping %d subject(s): NPZ not found under %s (expected %s): %s",
            len(missing_npz),
            data_root_p,
            npz_pattern,
            missing_npz,
        )
    if not subjects_present:
        raise FileNotFoundError(
            f"No subject NPZ files found for IDs {cfg.subjects} under {data_root_p} "
            f"(expected names like subject_1.npz). Prepare data or fix data_root."
        )

    dataset = NpzMotorImageryDataset(
        data_root=cfg.data_root,
        subjects=subjects_present,
        float_dtype=cfg.memory.float_dtype,
    )

    subject_performances: List[SubjectPerformance] = []

    pipelines = []
    if cfg.denoising.use_baseline:
        pipelines.append("baseline")
    # ASR (Artifact Subspace Reconstruction) – optional additional denoising pipeline.
    if getattr(cfg.denoising, "use_asr", False):
        pipelines.append("asr")
    if cfg.denoising.use_icalabel:
        pipelines.append("icalabel")
    if cfg.denoising.use_gedai:
        pipelines.append("gedai")
    if getattr(cfg.denoising, "use_gedai_mrcp", False):
        pipelines.append("gedai_mrcp")
    if getattr(cfg.denoising, "use_pylossless", False):
        pipelines.append("pylossless")

    backbones = []
    if cfg.backbones.use_csp:
        backbones.append("csp")
    if cfg.backbones.use_tangent_space:
        backbones.append("tangent")
    if getattr(cfg.backbones, "use_time_lda", False):
        backbones.append("time_lda")
    if getattr(cfg.backbones, "use_transformer", False):
        backbones.append("transformer")
    if getattr(cfg.backbones, "use_eegnet", False):
        backbones.append("eegnet")

    n_subj = len(subjects_present)
    log.info(
        f"Experiment: {n_subj} subjects, pipelines={pipelines}, backbones={backbones}, "
        f"subject_chance={cfg.statistics.subject_chance_method}, "
        f"pipeline_cmp={cfg.statistics.pipeline_comparison_method}, "
        f"null_perm={cfg.permutation.n_subject_level}, pipeline_perm={cfg.permutation.n_pipeline_level}"
    )

    # Subject-level loop (sequential to respect memory constraints)
    for sid, subj_data in tqdm(
        dataset.iter_subjects(), desc="Subjects", total=n_subj
    ):
        log.info(f"Subject {sid}/{n_subj} started (n_trials={subj_data.X.shape[0]})")
        _log_memory_if_debug(log)
        X, y = subj_data.X, subj_data.y
        X, y = _apply_max_trials_smoke(cfg, X, y, sid, log)
        eval_ch_idx = None
        eval_ch_names = list(subj_data.ch_names)
        if cfg.dataset_label and "eeg_emg_mrcp" in cfg.dataset_label.lower():
            wanted = {c.upper() for c in MRCP_MOTOR_CHANNELS}
            idx = [i for i, c in enumerate(subj_data.ch_names) if str(c).upper() in wanted]
            if idx:
                eval_ch_idx = np.asarray(idx, dtype=int)
                eval_ch_names = [str(subj_data.ch_names[i]) for i in eval_ch_idx]
                log.info(
                    f"Subject {sid}: evaluating on MRCP motor subset only "
                    f"({len(eval_ch_names)} channels)."
                )
        n_classes = int(len(np.unique(y)))
        backbones_this = list(backbones)
        if n_classes > 2 and "csp" in backbones_this and not cfg.backbones.allow_csp_multiclass:
            backbones_this = [b for b in backbones_this if b != "csp"]
            log.warning(
                f"Subject {sid}: {n_classes} classes detected. "
                "Skipping CSP (binary-only) unless backbones.allow_csp_multiclass=true."
            )
        cv_splits = _build_cv_splits(X, y, cfg)
        aasd_groups = _aasd_window_groups(X, cfg)

        # Pipeline order: baseline and ICALabel first, then GEDAI.
        mrcp_baseline_cache = None
        aasd_baseline_cache: np.ndarray | None = None
        aasd_gate_decisions: List[str] | None = None
        # When ``gedai_aasd_perfold_refcov`` is enabled, the AASD GEDAI pipeline
        # builds one X_proc per outer fold using only that fold's training
        # trials for the refcov. Each entry is a (n_windows, n_ch, n_times)
        # array with the GEDAI cleaning re-applied for that fold.
        aasd_perfold_X_eval: List[np.ndarray] | None = None
        for pipeline in pipelines:
            if pipeline in ("gedai", "gedai_mrcp", "pylossless") and (
                "baseline" in pipelines or "icalabel" in pipelines
            ):
                gc.collect()
                _log_memory_if_debug(log)
                log.info(
                    f"  (lighter pipelines done for this subject; running {pipeline} next)"
                )

            is_mrcp_gedai = (
                pipeline == "gedai_mrcp"
                and cfg.dataset_label
                and "eeg_emg_mrcp" in cfg.dataset_label.lower()
            )
            is_aasd_perfold_gedai = (
                pipeline == "gedai"
                and cfg.dataset_label
                and "aasd" in cfg.dataset_label.lower()
                and getattr(cfg.denoising, "gedai_aasd_perfold_refcov", False)
            )
            if is_aasd_perfold_gedai:
                # Leakage-safe AASD GEDAI: refit the CSP-style refcov on each
                # outer fold's training trials and re-clean the entire signal
                # with that fold-specific filter. This costs K x apply_gedai
                # per subject but removes the test-label leakage in the spatial
                # filter direction, which is the right thing for a preprint.
                aasd_perfold_X_eval = []
                X_proc = None
                X_eval = None
                for fold_i, (train_idx, _test_idx) in enumerate(cv_splits):
                    X_fold = preprocess_subject_data(
                        X=X,
                        sfreq=subj_data.sfreq,
                        ch_names=subj_data.ch_names,
                        y=y,
                        l_freq=cfg.bandpass.l_freq,
                        h_freq=cfg.bandpass.h_freq,
                        denoising="gedai",
                        subject_id=sid,
                        dataset_name=cfg.dataset_label or "",
                        gedai_n_jobs=cfg.memory.n_jobs,
                        data_root=cfg.data_root,
                        train_idx=train_idx,
                        aasd_refcov_disc_weight=cfg.denoising.gedai_aasd_refcov_disc_weight,
                        aasd_refcov_task_weight=cfg.denoising.gedai_aasd_refcov_task_weight,
                        aasd_refcov_lw_weight=cfg.denoising.gedai_aasd_refcov_lw_weight,
                        use_anti_laplacian=cfg.denoising.use_anti_laplacian,
                        anti_laplacian_n_neighbors=cfg.denoising.anti_laplacian_n_neighbors,
                    )
                    X_fold_eval = (
                        X_fold[:, eval_ch_idx, :] if eval_ch_idx is not None else X_fold
                    )
                    aasd_perfold_X_eval.append(X_fold_eval.astype(np.float32, copy=False))
                    if fold_i == 0:
                        X_proc = X_fold
                        X_eval = X_fold_eval
                    del X_fold
                gc.collect()
            elif is_mrcp_gedai:
                # Leakage-safe mode: fit C_mrcp on each fold's training trials only,
                # then run GEDAI with that fold-specific reference covariance.
                X_proc = None
                X_eval = None
                fold_X_eval = []
                baseline_fold_X_eval = None
                for fold_i, (train_idx, test_idx) in enumerate(cv_splits):
                    X_fold = preprocess_subject_data(
                        X=X,
                        sfreq=subj_data.sfreq,
                        ch_names=subj_data.ch_names,
                        l_freq=cfg.bandpass.l_freq,
                        h_freq=cfg.bandpass.h_freq,
                        denoising=pipeline,
                        subject_id=sid,
                        dataset_name=cfg.dataset_label or "",
                        gedai_n_jobs=cfg.memory.n_jobs,
                        data_root=cfg.data_root,
                        y=y,
                        train_idx=(
                            None
                            if cfg.denoising.gedai_mrcp_refcov_all_trials
                            else train_idx
                        ),
                        mrcp_refcov_prior=cfg.denoising.gedai_mrcp_prior,
                        mrcp_refcov_cache=mrcp_baseline_cache,
                        mrcp_gedai_noise_multiplier=cfg.denoising.gedai_mrcp_noise_multiplier,
                        mrcp_gedai_retention_min=cfg.denoising.gedai_mrcp_retention_min,
                        mrcp_refcov_tmin_s=cfg.denoising.gedai_mrcp_refcov_tmin_s,
                        mrcp_refcov_rank_max=cfg.denoising.gedai_mrcp_refcov_rank_max,
                        mrcp_refcov_motor_weight=cfg.denoising.gedai_mrcp_refcov_motor_weight,
                        mrcp_refcov_move_mix=cfg.denoising.gedai_mrcp_refcov_move_mix,
                        use_anti_laplacian=cfg.denoising.use_anti_laplacian,
                        anti_laplacian_n_neighbors=cfg.denoising.anti_laplacian_n_neighbors,
                    )
                    X_fold_eval = (
                        X_fold[:, eval_ch_idx, :] if eval_ch_idx is not None else X_fold
                    )
                    fold_X_eval.append((X_fold_eval[train_idx], X_fold_eval[test_idx]))
                    if fold_i == 0:
                        X_proc = X_fold
                        X_eval = X_fold_eval
                    del X_fold, X_fold_eval
                if cfg.denoising.gedai_mrcp_adaptive_baseline_gate:
                    if mrcp_baseline_cache is None:
                        X_base = preprocess_subject_data(
                            X=X,
                            sfreq=subj_data.sfreq,
                            ch_names=subj_data.ch_names,
                            l_freq=cfg.bandpass.l_freq,
                            h_freq=cfg.bandpass.h_freq,
                            denoising="baseline",
                            subject_id=sid,
                            dataset_name=cfg.dataset_label or "",
                            gedai_n_jobs=cfg.memory.n_jobs,
                            data_root=cfg.data_root,
                            use_anti_laplacian=cfg.denoising.use_anti_laplacian,
                            anti_laplacian_n_neighbors=cfg.denoising.anti_laplacian_n_neighbors,
                        )
                    else:
                        X_base = mrcp_baseline_cache
                    X_base_eval = (
                        X_base[:, eval_ch_idx, :] if eval_ch_idx is not None else X_base
                    )
                    baseline_fold_X_eval = [
                        (X_base_eval[train_idx], X_base_eval[test_idx])
                        for train_idx, test_idx in cv_splits
                    ]
            else:
                X_proc = preprocess_subject_data(
                    X=X,
                    sfreq=subj_data.sfreq,
                    ch_names=subj_data.ch_names,
                    y=y,
                    l_freq=cfg.bandpass.l_freq,
                    h_freq=cfg.bandpass.h_freq,
                    denoising=pipeline,
                    # GEDAI (PhysioNet): continuous EDF; PyLossless (Alljoined): optional manifest.
                    # MRCP baseline also needs subject_id so it can go through the same
                    # continuous-raw path as GEDAI for a fair comparison.
                    subject_id=(
                        sid
                        if (
                            pipeline in ("gedai", "pylossless")
                            or (
                                pipeline == "baseline"
                                and cfg.dataset_label
                                and "eeg_emg_mrcp" in cfg.dataset_label.lower()
                            )
                        )
                        else None
                    ),
                    dataset_name=cfg.dataset_label or "",
                    gedai_n_jobs=cfg.memory.n_jobs,
                    data_root=cfg.data_root,
                    mrcp_refcov_prior=cfg.denoising.gedai_mrcp_prior,
                    mrcp_gedai_noise_multiplier=cfg.denoising.gedai_mrcp_noise_multiplier,
                    mrcp_gedai_retention_min=cfg.denoising.gedai_mrcp_retention_min,
                    mrcp_refcov_tmin_s=cfg.denoising.gedai_mrcp_refcov_tmin_s,
                    mrcp_refcov_rank_max=cfg.denoising.gedai_mrcp_refcov_rank_max,
                    mrcp_refcov_motor_weight=cfg.denoising.gedai_mrcp_refcov_motor_weight,
                    mrcp_refcov_move_mix=cfg.denoising.gedai_mrcp_refcov_move_mix,
                    aasd_refcov_disc_weight=cfg.denoising.gedai_aasd_refcov_disc_weight,
                    aasd_refcov_task_weight=cfg.denoising.gedai_aasd_refcov_task_weight,
                    aasd_refcov_lw_weight=cfg.denoising.gedai_aasd_refcov_lw_weight,
                    use_anti_laplacian=cfg.denoising.use_anti_laplacian,
                    anti_laplacian_n_neighbors=cfg.denoising.anti_laplacian_n_neighbors,
                )
                X_eval = X_proc[:, eval_ch_idx, :] if eval_ch_idx is not None else X_proc
                if (
                    pipeline == "baseline"
                    and cfg.dataset_label
                    and "eeg_emg_mrcp" in cfg.dataset_label.lower()
                ):
                    mrcp_baseline_cache = X_proc.copy()
                if (
                    pipeline == "baseline"
                    and cfg.dataset_label
                    and "aasd" in cfg.dataset_label.lower()
                    and (
                        getattr(cfg.denoising, "gedai_aasd_adaptive_gate", False)
                        or getattr(cfg.denoising, "gedai_aasd_perfold_refcov", False)
                    )
                ):
                    aasd_baseline_cache = X_proc.copy()

            if cfg.memory.save_denoised_npz:
                ddir = results_root / cfg.memory.denoised_subdir
                ch_arr = np.asarray(list(subj_data.ch_names), dtype=object)
                den_path = ddir / f"subject_{sid}_{pipeline}.npz"
                np.savez_compressed(
                    den_path,
                    X=np.asarray(X_proc, dtype=np.float32),
                    y=y,
                    sfreq=np.float32(subj_data.sfreq),
                    ch_names=ch_arr,
                )
                log.info(f"  Saved denoised data {den_path}")

            # --- Signal Preservation Calculation ---
            # We use the config's channels_of_interest (first one as primary)
            # Baseline power is calculated from X_bp (the first pipeline which is usually 'baseline')
            coi = cfg.signal_integrity.channels_of_interest[0]
            try:
                ch_idx = [c.upper() for c in subj_data.ch_names].index(coi.upper())
            except ValueError:
                ch_idx = 0
            
            alpha_band = (8.0, 12.0)
            beta_band = (13.0, 30.0)
            
            # Record power for this pipeline
            signal_ref = X_proc if X_proc is not None else fold_X_eval[0][0]
            curr_alpha = compute_band_power(signal_ref, subj_data.sfreq, *alpha_band, ch_idx=ch_idx)
            curr_beta = compute_band_power(signal_ref, subj_data.sfreq, *beta_band, ch_idx=ch_idx)
            
            if pipeline == "baseline":
                ref_alpha = curr_alpha if curr_alpha > 0 else 1.0
                ref_beta = curr_beta if curr_beta > 0 else 1.0
                alpha_ratio = 1.0
                beta_ratio = 1.0
            else:
                alpha_ratio = curr_alpha / ref_alpha
                beta_ratio = curr_beta / ref_beta

            # --- Alpha Lateralization Index ---
            # Skipped for AASD: this dataset's MAT files don't ship channel
            # labels, so our ch_names are fabricated 10-20 names. LI computed
            # from those would be physiologically meaningless; report 0.0.
            li = 0.0
            aasd_dataset = (
                cfg.dataset_label and "aasd" in str(cfg.dataset_label).lower()
            )
            if aasd_dataset:
                li = 0.0
            elif X_proc is not None:
                try:
                    li = _compute_lateralization_index(X_proc, y, subj_data.ch_names)
                except Exception:
                    li = 0.0

            # --- Automated Plots (Subject 1 only or if configured) ---
            if sid == cfg.subjects[0]:
                plots_dir = results_root / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # PSD Comparison
                psd_out = plots_dir / f"sub{sid}_{pipeline}_psd_comparison.png"
                # We need X_bp as the reference "In"
                # But X_proc is the current pipeline's "Out"
                # To make this clean, we'd need to keep X_baseline around or re-run bandpass.
                # However, X_proc for 'baseline' IS the bandpass.
                if pipeline == "baseline":
                    # Store baseline for comparison with future pipelines
                    X_baseline_ref = X_proc.copy()
                else:
                    # Plot current vs baseline
                    plot_denoising_psd_comparison(
                        X_in=X_baseline_ref,
                        X_out=X_proc,
                        sfreq=subj_data.sfreq,
                        out_path=psd_out,
                        subject_id=sid,
                        pipeline_name=pipeline
                    )
                    
                    # Time Domain Overlay
                    overlay_out = plots_dir / f"sub{sid}_{pipeline}_overlay_comparison.png"
                    plot_denoising_comparison_overlay(
                        X_in=X_baseline_ref,
                        X_out=X_proc,
                        sfreq=subj_data.sfreq,
                        ch_names=subj_data.ch_names,
                        out_path=overlay_out,
                        subject_id=sid,
                        pipeline_name=pipeline,
                        n_channels=5,
                        trial_idx=0
                    )

            tangent_fold_features = None
            time_lda_fold_features = None
            use_chance_binom = cfg.statistics.subject_chance_method == "binomial"
            n_classes = int(len(np.unique(y)))

            for backbone in backbones_this:
                fold_auc_vals: List[float] = []
                if backbone == "csp":
                    if is_mrcp_gedai:
                        fold_acc = []
                        pooled_c = 0
                        pooled_t = 0
                        for (train_idx, test_idx), (X_train_fold, X_test_fold) in zip(cv_splits, fold_X_eval):
                            y_train = y[train_idx]
                            y_test = y[test_idx]
                            X_both = np.concatenate([X_train_fold, X_test_fold], axis=0)
                            y_both = np.concatenate([y_train, y_test], axis=0)
                            tr = np.arange(len(y_train))
                            te = np.arange(len(y_train), len(y_train) + len(y_test))
                            cres = run_csp_cv_preprocessed(
                                X_proc=X_both, y=y_both, cv_splits=[(tr, te)]
                            )
                            fold_acc.extend(cres.fold_accuracies)
                            fold_auc_vals.extend(getattr(cres, "fold_aucs", []))
                            pooled_c += int(cres.pooled_test_correct)
                            pooled_t += int(cres.pooled_test_total)
                            del cres, X_both, y_both
                        res = None
                    else:
                        res = run_csp_cv_preprocessed(X_proc=X_eval, y=y, cv_splits=cv_splits)
                        fold_acc = list(res.fold_accuracies)
                        fold_auc_vals = list(getattr(res, "fold_aucs", []))
                        pooled_c = int(res.pooled_test_correct)
                        pooled_t = int(res.pooled_test_total)
                elif backbone == "tangent":
                    if tangent_fold_features is None:
                        if is_mrcp_gedai:
                            tangent_fold_features = []
                            for X_train_fold, X_test_fold in fold_X_eval:
                                X_both = np.concatenate([X_train_fold, X_test_fold], axis=0)
                                tr = np.arange(X_train_fold.shape[0])
                                te = np.arange(X_train_fold.shape[0], X_both.shape[0])
                                ff = build_tangent_features_for_splits(X_both, [(tr, te)])
                                tangent_fold_features.append(ff[0])
                                del ff, X_both
                        else:
                            tangent_fold_features = build_tangent_features_for_splits(
                                X_proc=X_eval,
                                cv_splits=cv_splits,
                            )
                    res = run_tangent_cv_precomputed_features(
                        y=y,
                        cv_splits=cv_splits,
                        fold_features=tangent_fold_features,
                    )
                    fold_acc = list(res.fold_accuracies)
                    fold_auc_vals = list(getattr(res, "fold_aucs", []))
                    mean_acc = float(np.mean(fold_acc))
                    std_acc = float(np.std(fold_acc, ddof=1))
                    pooled_c = int(res.pooled_test_correct)
                    pooled_t = int(res.pooled_test_total)
                elif backbone == "time_lda":
                    if time_lda_fold_features is None:
                        if is_mrcp_gedai:
                            active_fold_inputs = fold_X_eval
                            if (
                                cfg.denoising.gedai_mrcp_adaptive_baseline_gate
                                and baseline_fold_X_eval is not None
                            ):
                                active_fold_inputs = []
                                gate_margin = float(cfg.denoising.gedai_mrcp_gate_margin)
                                inner_splits = int(cfg.denoising.gedai_mrcp_gate_inner_splits)
                                for fold_i, (
                                    (train_idx, _test_idx),
                                    (Xg_train, Xg_test),
                                    (Xb_train, Xb_test),
                                ) in enumerate(
                                    zip(cv_splits, fold_X_eval, baseline_fold_X_eval)
                                ):
                                    y_train = y[train_idx]
                                    gate_rs = int(cfg.cv.random_state) + int(sid) * 1009 + fold_i
                                    base_score = _score_time_lda_inner_cv(
                                        X_proc=Xb_train,
                                        y=y_train,
                                        sfreq=subj_data.sfreq,
                                        target_sfreq=getattr(
                                            cfg.backbones, "time_lda_target_sfreq", None
                                        ),
                                        n_splits=inner_splits,
                                        random_state=gate_rs,
                                    )
                                    gedai_score = _score_time_lda_inner_cv(
                                        X_proc=Xg_train,
                                        y=y_train,
                                        sfreq=subj_data.sfreq,
                                        target_sfreq=getattr(
                                            cfg.backbones, "time_lda_target_sfreq", None
                                        ),
                                        n_splits=inner_splits,
                                        random_state=gate_rs,
                                    )
                                    use_gedai_fold = gedai_score >= (base_score + gate_margin)
                                    active_fold_inputs.append(
                                        (Xg_train, Xg_test) if use_gedai_fold else (Xb_train, Xb_test)
                                    )
                                    log.info(
                                        "  Subject %s | time_lda | gedai_mrcp gate | fold=%d "
                                        "baseline_train_cv=%.3f gedai_train_cv=%.3f margin=%.3f "
                                        "choice=%s",
                                        sid,
                                        fold_i,
                                        base_score,
                                        gedai_score,
                                        gate_margin,
                                        "gedai_mrcp" if use_gedai_fold else "baseline",
                                    )
                            time_lda_fold_features = []
                            for X_train_fold, X_test_fold in active_fold_inputs:
                                X_both = np.concatenate([X_train_fold, X_test_fold], axis=0)
                                tr = np.arange(X_train_fold.shape[0])
                                te = np.arange(X_train_fold.shape[0], X_both.shape[0])
                                ff = build_time_lda_features_for_splits(
                                    X_both,
                                    [(tr, te)],
                                    sfreq=subj_data.sfreq,
                                    target_sfreq=getattr(cfg.backbones, "time_lda_target_sfreq", None),
                                )
                                time_lda_fold_features.append(ff[0])
                                del ff, X_both
                        else:
                            time_lda_fold_features = build_time_lda_features_for_splits(
                                X_proc=X_eval,
                                cv_splits=cv_splits,
                                sfreq=subj_data.sfreq,
                                target_sfreq=getattr(cfg.backbones, "time_lda_target_sfreq", None),
                            )
                    res = run_time_lda_cv_precomputed_features(
                        y=y,
                        cv_splits=cv_splits,
                        fold_features=time_lda_fold_features,
                    )
                    fold_acc = list(res.fold_accuracies)
                    fold_auc_vals = list(getattr(res, "fold_aucs", []))
                    mean_acc = float(np.mean(fold_acc))
                    std_acc = float(np.std(fold_acc, ddof=1))
                    pooled_c = int(res.pooled_test_correct)
                    pooled_t = int(res.pooled_test_total)
                elif backbone in ("transformer", "eegnet"):
                    is_eegnet = backbone == "eegnet"
                    if is_eegnet:
                        run_dl_cv, fit_dl_model = _lazy_eegnet_backbone()
                        score_inner_cv = _score_eegnet_inner_cv
                    else:
                        run_dl_cv, fit_dl_model = _lazy_transformer_backbone()
                        score_inner_cv = _score_transformer_inner_cv
                    tgt_sfreq = (
                        getattr(cfg.backbones, "eegnet_target_sfreq", None)
                        if is_eegnet
                        else getattr(cfg.backbones, "transformer_target_sfreq", None)
                    )
                    aasd_gate_enabled = bool(
                        getattr(cfg.denoising, "gedai_aasd_adaptive_gate", False)
                    )
                    aasd_perfold_enabled = bool(
                        getattr(cfg.denoising, "gedai_aasd_perfold_refcov", False)
                    )
                    aasd_gate_active = (
                        pipeline == "gedai"
                        and cfg.dataset_label
                        and "aasd" in cfg.dataset_label.lower()
                        and (aasd_gate_enabled or aasd_perfold_enabled)
                        and aasd_baseline_cache is not None
                    )
                    if aasd_gate_active:
                        gate_margin = float(getattr(cfg.denoising, "gedai_aasd_gate_margin", 0.0))
                        inner_splits = int(getattr(cfg.denoising, "gedai_aasd_gate_inner_splits", 3))
                        X_base_eval = (
                            aasd_baseline_cache[:, eval_ch_idx, :]
                            if eval_ch_idx is not None
                            else aasd_baseline_cache
                        )
                        aasd_gate_decisions = []
                        fold_acc = []
                        fold_auc_vals = []
                        pooled_c = 0
                        pooled_t = 0
                        for fold_i, (train_idx, test_idx) in enumerate(cv_splits):
                            y_train = y[train_idx]
                            gate_rs = int(cfg.cv.random_state) + int(sid) * 1009 + fold_i
                            train_groups = (
                                np.asarray(aasd_groups)[train_idx]
                                if aasd_groups is not None
                                else None
                            )
                            X_gedai_fold = (
                                aasd_perfold_X_eval[fold_i]
                                if aasd_perfold_X_eval is not None
                                else X_eval
                            )
                            if aasd_gate_enabled:
                                base_score = score_inner_cv(
                                    X_proc=X_base_eval[train_idx],
                                    y=y_train,
                                    sfreq=subj_data.sfreq,
                                    target_sfreq=tgt_sfreq,
                                    n_splits=inner_splits,
                                    random_state=gate_rs,
                                    groups=train_groups,
                                )
                                gedai_score = score_inner_cv(
                                    X_proc=X_gedai_fold[train_idx],
                                    y=y_train,
                                    sfreq=subj_data.sfreq,
                                    target_sfreq=tgt_sfreq,
                                    n_splits=inner_splits,
                                    random_state=gate_rs,
                                    groups=train_groups,
                                )
                                use_gedai_fold = gedai_score >= (base_score + gate_margin)
                                choice = "gedai" if use_gedai_fold else "baseline"
                                log.info(
                                    "  Subject %s | %s | aasd gate | fold=%d "
                                    "baseline_train_cv=%.3f gedai_train_cv=%.3f margin=%.3f "
                                    "choice=%s",
                                    sid,
                                    backbone,
                                    fold_i,
                                    base_score,
                                    gedai_score,
                                    gate_margin,
                                    choice,
                                )
                            else:
                                use_gedai_fold = True
                                choice = "gedai"
                                log.info(
                                    "  Subject %s | %s | aasd perfold-refcov | "
                                    "fold=%d (gate disabled, using GEDAI)",
                                    sid,
                                    backbone,
                                    fold_i,
                                )
                            aasd_gate_decisions.append(choice)
                            src = X_gedai_fold if use_gedai_fold else X_base_eval
                            X_train_f = src[train_idx]
                            X_test_f = src[test_idx]
                            X_both = np.concatenate([X_train_f, X_test_f], axis=0)
                            y_both = np.concatenate([y_train, y[test_idx]], axis=0)
                            tr = np.arange(len(y_train))
                            te = np.arange(len(y_train), len(y_both))
                            sub_groups = None
                            if aasd_groups is not None:
                                g_tr = np.asarray(aasd_groups)[train_idx]
                                g_te = np.asarray(aasd_groups)[test_idx]
                                sub_groups = np.concatenate([g_tr, g_te], axis=0)
                            rs_fold = int(cfg.cv.random_state) + int(sid) * 1009 + fold_i
                            if is_eegnet:
                                sub_res = run_dl_cv(
                                    X_proc=X_both,
                                    y=y_both,
                                    cv_splits=[(tr, te)],
                                    sfreq=subj_data.sfreq,
                                    target_sfreq=tgt_sfreq,
                                    epochs=int(getattr(cfg.backbones, "eegnet_epochs", 100)),
                                    batch_size=int(getattr(cfg.backbones, "eegnet_batch_size", 32)),
                                    learning_rate=float(
                                        getattr(cfg.backbones, "eegnet_learning_rate", 1e-3)
                                    ),
                                    weight_decay=float(
                                        getattr(cfg.backbones, "eegnet_weight_decay", 1e-4)
                                    ),
                                    dropout=float(getattr(cfg.backbones, "eegnet_dropout", 0.25)),
                                    F1=int(getattr(cfg.backbones, "eegnet_F1", 8)),
                                    D=int(getattr(cfg.backbones, "eegnet_D", 2)),
                                    F2=int(getattr(cfg.backbones, "eegnet_F2", 16)),
                                    kernel_length=getattr(
                                        cfg.backbones, "eegnet_kernel_length", None
                                    ),
                                    val_fraction=float(
                                        getattr(cfg.backbones, "eegnet_val_fraction", 0.125)
                                    ),
                                    patience=int(getattr(cfg.backbones, "eegnet_patience", 5)),
                                    device=str(getattr(cfg.backbones, "eegnet_device", "cpu")),
                                    random_state=rs_fold,
                                    groups=sub_groups,
                                    paper_exact=bool(
                                        getattr(cfg.backbones, "eegnet_paper_exact", False)
                                    ),
                                )
                            else:
                                sub_res = run_dl_cv(
                                    X_proc=X_both,
                                    y=y_both,
                                    cv_splits=[(tr, te)],
                                    sfreq=subj_data.sfreq,
                                    target_sfreq=tgt_sfreq,
                                    epochs=int(getattr(cfg.backbones, "transformer_epochs", 50)),
                                    batch_size=int(
                                        getattr(cfg.backbones, "transformer_batch_size", 16)
                                    ),
                                    learning_rate=float(
                                        getattr(cfg.backbones, "transformer_learning_rate", 1e-4)
                                    ),
                                    weight_decay=float(
                                        getattr(cfg.backbones, "transformer_weight_decay", 1e-4)
                                    ),
                                    dropout=float(getattr(cfg.backbones, "transformer_dropout", 0.3)),
                                    d_model=int(getattr(cfg.backbones, "transformer_d_model", 256)),
                                    n_heads=int(getattr(cfg.backbones, "transformer_n_heads", 8)),
                                    n_layers=int(getattr(cfg.backbones, "transformer_n_layers", 4)),
                                    ff_dim=int(getattr(cfg.backbones, "transformer_ff_dim", 256)),
                                    val_fraction=float(
                                        getattr(cfg.backbones, "transformer_val_fraction", 0.125)
                                    ),
                                    patience=int(getattr(cfg.backbones, "transformer_patience", 5)),
                                    device=str(
                                        getattr(cfg.backbones, "transformer_device", "cpu")
                                    ),
                                    random_state=rs_fold,
                                    groups=sub_groups,
                                    paper_exact=bool(
                                        getattr(cfg.backbones, "transformer_paper_exact", False)
                                    ),
                                )
                            fold_acc.extend(sub_res.fold_accuracies)
                            fold_auc_vals.extend(getattr(sub_res, "fold_aucs", []))
                            pooled_c += int(sub_res.pooled_test_correct)
                            pooled_t += int(sub_res.pooled_test_total)
                        res = None
                        mean_acc = float(np.mean(fold_acc))
                        std_acc = float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0
                    else:
                        rs_outer = int(cfg.cv.random_state) + int(sid) * 1009
                        if is_eegnet:
                            res = run_dl_cv(
                                X_proc=X_eval,
                                y=y,
                                cv_splits=cv_splits,
                                sfreq=subj_data.sfreq,
                                target_sfreq=tgt_sfreq,
                                epochs=int(getattr(cfg.backbones, "eegnet_epochs", 100)),
                                batch_size=int(getattr(cfg.backbones, "eegnet_batch_size", 32)),
                                learning_rate=float(
                                    getattr(cfg.backbones, "eegnet_learning_rate", 1e-3)
                                ),
                                weight_decay=float(
                                    getattr(cfg.backbones, "eegnet_weight_decay", 1e-4)
                                ),
                                dropout=float(getattr(cfg.backbones, "eegnet_dropout", 0.25)),
                                F1=int(getattr(cfg.backbones, "eegnet_F1", 8)),
                                D=int(getattr(cfg.backbones, "eegnet_D", 2)),
                                F2=int(getattr(cfg.backbones, "eegnet_F2", 16)),
                                kernel_length=getattr(cfg.backbones, "eegnet_kernel_length", None),
                                val_fraction=float(
                                    getattr(cfg.backbones, "eegnet_val_fraction", 0.125)
                                ),
                                patience=int(getattr(cfg.backbones, "eegnet_patience", 5)),
                                device=str(getattr(cfg.backbones, "eegnet_device", "cpu")),
                                random_state=rs_outer,
                                groups=aasd_groups,
                                paper_exact=bool(
                                    getattr(cfg.backbones, "eegnet_paper_exact", False)
                                ),
                            )
                        else:
                            res = run_dl_cv(
                                X_proc=X_eval,
                                y=y,
                                cv_splits=cv_splits,
                                sfreq=subj_data.sfreq,
                                target_sfreq=tgt_sfreq,
                                epochs=int(getattr(cfg.backbones, "transformer_epochs", 50)),
                                batch_size=int(getattr(cfg.backbones, "transformer_batch_size", 16)),
                                learning_rate=float(
                                    getattr(cfg.backbones, "transformer_learning_rate", 1e-4)
                                ),
                                weight_decay=float(
                                    getattr(cfg.backbones, "transformer_weight_decay", 1e-4)
                                ),
                                dropout=float(getattr(cfg.backbones, "transformer_dropout", 0.3)),
                                d_model=int(getattr(cfg.backbones, "transformer_d_model", 256)),
                                n_heads=int(getattr(cfg.backbones, "transformer_n_heads", 8)),
                                n_layers=int(getattr(cfg.backbones, "transformer_n_layers", 4)),
                                ff_dim=int(getattr(cfg.backbones, "transformer_ff_dim", 256)),
                                val_fraction=float(
                                    getattr(cfg.backbones, "transformer_val_fraction", 0.125)
                                ),
                                patience=int(getattr(cfg.backbones, "transformer_patience", 5)),
                                device=str(getattr(cfg.backbones, "transformer_device", "cpu")),
                                random_state=rs_outer,
                                groups=aasd_groups,
                                paper_exact=bool(
                                    getattr(cfg.backbones, "transformer_paper_exact", False)
                                ),
                            )
                        fold_acc = list(res.fold_accuracies)
                        fold_auc_vals = list(getattr(res, "fold_aucs", []))
                        mean_acc = float(np.mean(fold_acc))
                        std_acc = float(np.std(fold_acc, ddof=1))
                        pooled_c = int(res.pooled_test_correct)
                        pooled_t = int(res.pooled_test_total)
                else:
                    raise ValueError(f"Unsupported backbone: {backbone}")

                if backbone == "csp":
                    mean_acc = float(np.mean(fold_acc))
                    std_acc = float(np.std(fold_acc, ddof=1))

                if use_chance_binom:
                    p_emp = binomial_vs_chance_p_value(pooled_c, pooled_t, n_classes)
                    p_method = "binomial"
                else:
                    p_method = "permutation"
                    null_acc: List[float] = []
                    base_n = int(cfg.permutation.n_subject_level)
                    for _ in tqdm(
                        range(base_n),
                        desc=f"  null s{sid} {backbone} {pipeline}",
                        leave=False,
                        total=base_n,
                    ):
                        y_shuffled = np.random.permutation(y)
                        if backbone == "csp":
                            if is_mrcp_gedai:
                                n_fold = []
                                n_pc = 0
                                n_pt = 0
                                for (train_idx, test_idx), (X_train_fold, X_test_fold) in zip(cv_splits, fold_X_eval):
                                    y_train = y_shuffled[train_idx]
                                    y_test = y_shuffled[test_idx]
                                    X_both = np.concatenate([X_train_fold, X_test_fold], axis=0)
                                    y_both = np.concatenate([y_train, y_test], axis=0)
                                    tr = np.arange(len(y_train))
                                    te = np.arange(len(y_train), len(y_train) + len(y_test))
                                    cres = run_csp_cv_preprocessed(X_proc=X_both, y=y_both, cv_splits=[(tr, te)])
                                    n_fold.extend(cres.fold_accuracies)
                                    n_pc += int(cres.pooled_test_correct)
                                    n_pt += int(cres.pooled_test_total)
                                    del cres, X_both, y_both
                                class _Tmp:
                                    pass
                                nres = _Tmp()
                                nres.fold_accuracies = n_fold
                                nres.pooled_test_correct = n_pc
                                nres.pooled_test_total = n_pt
                            else:
                                nres = run_csp_cv_preprocessed(
                                    X_proc=X_eval,
                                    y=y_shuffled,
                                    cv_splits=cv_splits,
                                )
                        elif backbone == "tangent":
                            nres = run_tangent_cv_precomputed_features(
                                y=y_shuffled,
                                cv_splits=cv_splits,
                                fold_features=tangent_fold_features,
                            )
                        elif backbone == "time_lda":
                            nres = run_time_lda_cv_precomputed_features(
                                y=y_shuffled,
                                cv_splits=cv_splits,
                                fold_features=time_lda_fold_features,
                            )
                        elif backbone == "eegnet":
                            from ..backbones.eegnet_eeg import run_eegnet_cv_preprocessed

                            nres = run_eegnet_cv_preprocessed(
                                X_proc=X_eval,
                                y=y_shuffled,
                                cv_splits=cv_splits,
                                sfreq=subj_data.sfreq,
                                target_sfreq=getattr(cfg.backbones, "eegnet_target_sfreq", None),
                                epochs=int(getattr(cfg.backbones, "eegnet_epochs", 100)),
                                batch_size=int(getattr(cfg.backbones, "eegnet_batch_size", 32)),
                                learning_rate=float(getattr(cfg.backbones, "eegnet_learning_rate", 1e-3)),
                                weight_decay=float(getattr(cfg.backbones, "eegnet_weight_decay", 1e-4)),
                                dropout=float(getattr(cfg.backbones, "eegnet_dropout", 0.25)),
                                F1=int(getattr(cfg.backbones, "eegnet_F1", 8)),
                                D=int(getattr(cfg.backbones, "eegnet_D", 2)),
                                F2=int(getattr(cfg.backbones, "eegnet_F2", 16)),
                                kernel_length=getattr(cfg.backbones, "eegnet_kernel_length", None),
                                val_fraction=float(getattr(cfg.backbones, "eegnet_val_fraction", 0.125)),
                                patience=int(getattr(cfg.backbones, "eegnet_patience", 5)),
                                device=str(getattr(cfg.backbones, "eegnet_device", "cpu")),
                                random_state=int(cfg.cv.random_state) + int(sid) * 1009,
                                groups=aasd_groups,
                                paper_exact=bool(getattr(cfg.backbones, "eegnet_paper_exact", False)),
                            )
                        elif backbone == "transformer":
                            (
                                run_transformer_cv_preprocessed,
                                _fit_t,
                            ) = _lazy_transformer_backbone()
                            nres = run_transformer_cv_preprocessed(
                                X_proc=X_eval,
                                y=y_shuffled,
                                cv_splits=cv_splits,
                                sfreq=subj_data.sfreq,
                                target_sfreq=getattr(cfg.backbones, "transformer_target_sfreq", None),
                                epochs=int(getattr(cfg.backbones, "transformer_epochs", 50)),
                                batch_size=int(getattr(cfg.backbones, "transformer_batch_size", 16)),
                                learning_rate=float(getattr(cfg.backbones, "transformer_learning_rate", 1e-4)),
                                weight_decay=float(getattr(cfg.backbones, "transformer_weight_decay", 1e-4)),
                                dropout=float(getattr(cfg.backbones, "transformer_dropout", 0.3)),
                                d_model=int(getattr(cfg.backbones, "transformer_d_model", 256)),
                                n_heads=int(getattr(cfg.backbones, "transformer_n_heads", 8)),
                                n_layers=int(getattr(cfg.backbones, "transformer_n_layers", 4)),
                                ff_dim=int(getattr(cfg.backbones, "transformer_ff_dim", 256)),
                                val_fraction=float(getattr(cfg.backbones, "transformer_val_fraction", 0.125)),
                                patience=int(getattr(cfg.backbones, "transformer_patience", 5)),
                                device=str(getattr(cfg.backbones, "transformer_device", "cpu")),
                                random_state=int(cfg.cv.random_state) + int(sid) * 1009,
                                groups=aasd_groups,
                                paper_exact=bool(getattr(cfg.backbones, "transformer_paper_exact", False)),
                            )
                        else:
                            raise ValueError(
                                f"Unsupported backbone for permutation null: {backbone}"
                            )
                        null_acc.append(float(np.mean(nres.fold_accuracies)))
                        del nres

                    p_emp = empirical_chance_p_value(mean_acc, null_acc)

                    if (
                        cfg.permutation.adaptive_step_up
                        and _is_borderline_p_value(p_emp, cfg)
                        and int(cfg.permutation.step_up_n_subject_level) > base_n
                    ):
                        extra_n = int(cfg.permutation.step_up_n_subject_level) - base_n
                        log.info(
                            f"  Step-up permutations for s{sid} {backbone} {pipeline}: "
                            f"{base_n} -> {cfg.permutation.step_up_n_subject_level} (borderline p={p_emp:.4f})"
                        )
                        for _ in tqdm(
                            range(extra_n),
                            desc=f"  step-up s{sid} {backbone} {pipeline}",
                            leave=False,
                            total=extra_n,
                        ):
                            y_shuffled = np.random.permutation(y)
                            if backbone == "csp":
                                if is_mrcp_gedai:
                                    n_fold = []
                                    n_pc = 0
                                    n_pt = 0
                                    for (train_idx, test_idx), (X_train_fold, X_test_fold) in zip(cv_splits, fold_X_eval):
                                        y_train = y_shuffled[train_idx]
                                        y_test = y_shuffled[test_idx]
                                        X_both = np.concatenate([X_train_fold, X_test_fold], axis=0)
                                        y_both = np.concatenate([y_train, y_test], axis=0)
                                        tr = np.arange(len(y_train))
                                        te = np.arange(len(y_train), len(y_train) + len(y_test))
                                        cres = run_csp_cv_preprocessed(X_proc=X_both, y=y_both, cv_splits=[(tr, te)])
                                        n_fold.extend(cres.fold_accuracies)
                                        n_pc += int(cres.pooled_test_correct)
                                        n_pt += int(cres.pooled_test_total)
                                        del cres, X_both, y_both
                                    class _Tmp:
                                        pass
                                    nres = _Tmp()
                                    nres.fold_accuracies = n_fold
                                    nres.pooled_test_correct = n_pc
                                    nres.pooled_test_total = n_pt
                                else:
                                    nres = run_csp_cv_preprocessed(
                                        X_proc=X_eval,
                                        y=y_shuffled,
                                        cv_splits=cv_splits,
                                    )
                            elif backbone == "tangent":
                                nres = run_tangent_cv_precomputed_features(
                                    y=y_shuffled,
                                    cv_splits=cv_splits,
                                    fold_features=tangent_fold_features,
                                )
                            elif backbone == "time_lda":
                                nres = run_time_lda_cv_precomputed_features(
                                    y=y_shuffled,
                                    cv_splits=cv_splits,
                                    fold_features=time_lda_fold_features,
                                )
                            elif backbone == "transformer":
                                (
                                    run_transformer_cv_preprocessed,
                                    _fit_t2,
                                ) = _lazy_transformer_backbone()
                                nres = run_transformer_cv_preprocessed(
                                    X_proc=X_eval,
                                    y=y_shuffled,
                                    cv_splits=cv_splits,
                                    sfreq=subj_data.sfreq,
                                    target_sfreq=getattr(cfg.backbones, "transformer_target_sfreq", None),
                                    epochs=int(getattr(cfg.backbones, "transformer_epochs", 50)),
                                    batch_size=int(getattr(cfg.backbones, "transformer_batch_size", 16)),
                                    learning_rate=float(getattr(cfg.backbones, "transformer_learning_rate", 1e-4)),
                                    weight_decay=float(getattr(cfg.backbones, "transformer_weight_decay", 1e-4)),
                                    dropout=float(getattr(cfg.backbones, "transformer_dropout", 0.3)),
                                    d_model=int(getattr(cfg.backbones, "transformer_d_model", 256)),
                                    n_heads=int(getattr(cfg.backbones, "transformer_n_heads", 8)),
                                    n_layers=int(getattr(cfg.backbones, "transformer_n_layers", 4)),
                                    ff_dim=int(getattr(cfg.backbones, "transformer_ff_dim", 256)),
                                    val_fraction=float(getattr(cfg.backbones, "transformer_val_fraction", 0.125)),
                                    patience=int(getattr(cfg.backbones, "transformer_patience", 5)),
                                    device=str(getattr(cfg.backbones, "transformer_device", "cpu")),
                                    random_state=int(cfg.cv.random_state) + int(sid) * 1009,
                                    groups=aasd_groups,
                                    paper_exact=bool(getattr(cfg.backbones, "transformer_paper_exact", False)),
                                )
                            elif backbone == "eegnet":
                                from ..backbones.eegnet_eeg import run_eegnet_cv_preprocessed

                                nres = run_eegnet_cv_preprocessed(
                                    X_proc=X_eval,
                                    y=y_shuffled,
                                    cv_splits=cv_splits,
                                    sfreq=subj_data.sfreq,
                                    target_sfreq=getattr(cfg.backbones, "eegnet_target_sfreq", None),
                                    epochs=int(getattr(cfg.backbones, "eegnet_epochs", 100)),
                                    batch_size=int(getattr(cfg.backbones, "eegnet_batch_size", 32)),
                                    learning_rate=float(getattr(cfg.backbones, "eegnet_learning_rate", 1e-3)),
                                    weight_decay=float(getattr(cfg.backbones, "eegnet_weight_decay", 1e-4)),
                                    dropout=float(getattr(cfg.backbones, "eegnet_dropout", 0.25)),
                                    F1=int(getattr(cfg.backbones, "eegnet_F1", 8)),
                                    D=int(getattr(cfg.backbones, "eegnet_D", 2)),
                                    F2=int(getattr(cfg.backbones, "eegnet_F2", 16)),
                                    kernel_length=getattr(cfg.backbones, "eegnet_kernel_length", None),
                                    val_fraction=float(getattr(cfg.backbones, "eegnet_val_fraction", 0.125)),
                                    patience=int(getattr(cfg.backbones, "eegnet_patience", 5)),
                                    device=str(getattr(cfg.backbones, "eegnet_device", "cpu")),
                                    random_state=int(cfg.cv.random_state) + int(sid) * 1009,
                                    groups=aasd_groups,
                                    paper_exact=bool(getattr(cfg.backbones, "eegnet_paper_exact", False)),
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported backbone for step-up permutation: {backbone}"
                                )
                            null_acc.append(float(np.mean(nres.fold_accuracies)))
                            del nres
                        p_emp = empirical_chance_p_value(mean_acc, null_acc)
                    del null_acc

                del res

                subject_performances.append(
                    SubjectPerformance(
                        subject_id=sid,
                        backbone=backbone,
                        pipeline=pipeline,
                        fold_accuracies=fold_acc,
                        mean_accuracy=mean_acc,
                        std_accuracy=std_acc,
                        p_empirical=p_emp,
                        alpha_ratio=alpha_ratio,
                        beta_ratio=beta_ratio,
                        pooled_test_correct=pooled_c if use_chance_binom else None,
                        pooled_test_total=pooled_t if use_chance_binom else None,
                        p_vs_chance_method=p_method,
                        mean_auc=float(np.mean(fold_auc_vals)) if fold_auc_vals else 0.0,
                        fold_aucs=list(fold_auc_vals),
                        lateralization_index=li,
                    )
                )
                log.info(
                    f"  Subject {sid} | {backbone} | {pipeline} | acc={mean_acc:.3f} | p={p_emp:.4f} ({p_method})"
                )

                # Save trained model after each run (full-data fit) for reuse / deployment
                if cfg.memory.save_models:
                    if is_mrcp_gedai:
                        log.warning(
                            f"Skipping model export for subject {sid} {backbone} {pipeline} "
                            "because fold-specific GEDAI preprocessing has no single full-data model."
                        )
                        continue
                    model = None
                    try:
                        X_proc_full = (
                            X_eval
                            if X_eval is not None
                            else np.concatenate([a for a, _ in fold_X_eval], axis=0)
                        )
                        if backbone == "csp":
                            model = fit_csp_model_preprocessed(
                                X_proc=X_eval if X_eval is not None else np.concatenate([a for a, _ in fold_X_eval], axis=0),
                                y=y,
                                sfreq=subj_data.sfreq,
                                ch_names=eval_ch_names,
                                l_freq=cfg.bandpass.l_freq,
                                h_freq=cfg.bandpass.h_freq,
                                denoising=pipeline,
                            )
                        elif backbone == "tangent":
                            model = fit_tangent_model_preprocessed(
                                X_proc=X_eval if X_eval is not None else np.concatenate([a for a, _ in fold_X_eval], axis=0),
                                y=y,
                                sfreq=subj_data.sfreq,
                                ch_names=eval_ch_names,
                                l_freq=cfg.bandpass.l_freq,
                                h_freq=cfg.bandpass.h_freq,
                                denoising=pipeline,
                            )
                        elif backbone == "time_lda":
                            model = fit_time_lda_model_preprocessed(
                                X_proc=X_eval if X_eval is not None else np.concatenate([a for a, _ in fold_X_eval], axis=0),
                                y=y,
                                sfreq=subj_data.sfreq,
                                ch_names=eval_ch_names,
                                l_freq=cfg.bandpass.l_freq,
                                h_freq=cfg.bandpass.h_freq,
                                denoising=pipeline,
                                target_sfreq=getattr(cfg.backbones, "time_lda_target_sfreq", None),
                            )
                        elif backbone == "transformer":
                            _rt, fit_transformer_model_preprocessed = _lazy_transformer_backbone()
                            model = fit_transformer_model_preprocessed(
                                X_proc=X_proc_full,
                                y=y,
                                sfreq=subj_data.sfreq,
                                ch_names=eval_ch_names,
                                l_freq=cfg.bandpass.l_freq,
                                h_freq=cfg.bandpass.h_freq,
                                denoising=pipeline,
                                target_sfreq=getattr(cfg.backbones, "transformer_target_sfreq", None),
                                epochs=int(getattr(cfg.backbones, "transformer_epochs", 50)),
                                batch_size=int(getattr(cfg.backbones, "transformer_batch_size", 16)),
                                learning_rate=float(getattr(cfg.backbones, "transformer_learning_rate", 1e-4)),
                                weight_decay=float(getattr(cfg.backbones, "transformer_weight_decay", 1e-4)),
                                dropout=float(getattr(cfg.backbones, "transformer_dropout", 0.3)),
                                d_model=int(getattr(cfg.backbones, "transformer_d_model", 256)),
                                n_heads=int(getattr(cfg.backbones, "transformer_n_heads", 8)),
                                n_layers=int(getattr(cfg.backbones, "transformer_n_layers", 4)),
                                ff_dim=int(getattr(cfg.backbones, "transformer_ff_dim", 256)),
                                device=str(getattr(cfg.backbones, "transformer_device", "cpu")),
                                random_state=int(cfg.cv.random_state) + int(sid) * 1009,
                            )
                            del _rt
                        elif backbone == "eegnet":
                            _re, fit_eegnet_model_preprocessed = _lazy_eegnet_backbone()
                            model = fit_eegnet_model_preprocessed(
                                X_proc=X_proc_full,
                                y=y,
                                sfreq=subj_data.sfreq,
                                ch_names=eval_ch_names,
                                l_freq=cfg.bandpass.l_freq,
                                h_freq=cfg.bandpass.h_freq,
                                denoising=pipeline,
                                target_sfreq=getattr(cfg.backbones, "eegnet_target_sfreq", None),
                                epochs=int(getattr(cfg.backbones, "eegnet_epochs", 100)),
                                batch_size=int(getattr(cfg.backbones, "eegnet_batch_size", 32)),
                                learning_rate=float(getattr(cfg.backbones, "eegnet_learning_rate", 1e-3)),
                                weight_decay=float(getattr(cfg.backbones, "eegnet_weight_decay", 1e-4)),
                                dropout=float(getattr(cfg.backbones, "eegnet_dropout", 0.25)),
                                F1=int(getattr(cfg.backbones, "eegnet_F1", 8)),
                                D=int(getattr(cfg.backbones, "eegnet_D", 2)),
                                F2=int(getattr(cfg.backbones, "eegnet_F2", 16)),
                                kernel_length=getattr(cfg.backbones, "eegnet_kernel_length", None),
                                device=str(getattr(cfg.backbones, "eegnet_device", "cpu")),
                                random_state=int(cfg.cv.random_state) + int(sid) * 1009,
                            )
                            del _re
                        else:
                            raise ValueError(f"Model export not implemented for backbone: {backbone}")
                        path = models_dir / f"subject_{sid}_{backbone}_{pipeline}.joblib"
                        joblib.dump(model, path)
                    except Exception as e:
                        import warnings

                        warnings.warn(
                            f"Model save failed for subject {sid} {backbone} {pipeline}: {e}"
                        )
                    finally:
                        del model

            del tangent_fold_features
            del time_lda_fold_features
            if X_proc is not None:
                del X_proc
            # X_eval = X_proc for non-MRCP paths; drop the binding so the
            # array is freed before the next pipeline iteration starts.
            X_eval = None
            if is_mrcp_gedai:
                del fold_X_eval
            gc.collect()

        # Explicit cleanup: no raw data or large objects persist beyond this subject
        del X, y, cv_splits, subj_data
        gc.collect()
        _log_memory_if_debug(log)
        log.info(f"Subject {sid}/{n_subj} done.")

    # Save subject-level table
    rows = []
    for sp in subject_performances:
        for fold_idx, acc in enumerate(sp.fold_accuracies):
            rows.append(
                {
                    "subject": sp.subject_id,
                    "backbone": sp.backbone,
                    "pipeline": sp.pipeline,
                    "fold": fold_idx,
                    "accuracy": acc,
                    "auc": sp.fold_aucs[fold_idx] if fold_idx < len(sp.fold_aucs) else None,
                    "mean_accuracy": sp.mean_accuracy,
                    "mean_auc": sp.mean_auc,
                    "std_accuracy": sp.std_accuracy,
                    "p_empirical": sp.p_empirical,
                    "p_vs_chance_method": sp.p_vs_chance_method,
                    "pooled_test_correct": sp.pooled_test_correct,
                    "pooled_test_total": sp.pooled_test_total,
                    "alpha_ratio": sp.alpha_ratio,
                    "beta_ratio": sp.beta_ratio,
                    "lateralization_index": sp.lateralization_index,
                }
            )
    df_subject = pd.DataFrame(rows)
    df_subject.to_csv(tables_dir / "subject_level_performance.csv", index=False)
    log.info(
        "Subject-level table written. Computing between-pipeline tests "
        f"({cfg.statistics.pipeline_comparison_method})..."
    )

    # Between-pipeline comparisons (within backbone): GEDAI vs Bandpass, ICALabel vs Bandpass, GEDAI vs ICALabel
    comparison_rows: List[Dict] = []
    for backbone in backbones:
        for (p1, p2) in [
            ("gedai", "baseline"),
            ("gedai_mrcp", "baseline"),
            ("gedai_mrcp", "gedai"),
            ("icalabel", "baseline"),
            ("gedai", "icalabel"),
            ("asr", "baseline"),
            ("gedai", "asr"),
            ("icalabel", "asr"),
            ("pylossless", "baseline"),
            ("gedai", "pylossless"),
            ("icalabel", "pylossless"),
        ]:
            if p1 not in pipelines or p2 not in pipelines:
                continue

            scores1: List[float] = []
            scores2: List[float] = []
            # Aggregate over subjects using mean CV per subject
            for sp in subject_performances:
                if sp.backbone != backbone:
                    continue
                if sp.pipeline == p1:
                    scores1.append(sp.mean_accuracy)
                elif sp.pipeline == p2:
                    scores2.append(sp.mean_accuracy)

            if not scores1 or not scores2:
                continue

            pcm = cfg.statistics.pipeline_comparison_method
            if pcm == "mann_whitney":
                p_val = mann_whitney_pipeline_p_value(scores1, scores2)
            elif pcm == "wilcoxon":
                p_val = wilcoxon_paired_pipeline_p_value(scores1, scores2)
            else:
                p_val = paired_permutation_p_value(
                    scores1, scores2, n_resamples=cfg.permutation.n_pipeline_level
                )
            d_eff = cohen_d_pooled(scores1, scores2)

            comparison_rows.append(
                dict(
                    backbone=backbone,
                    comparison=f"{p1} - {p2}",
                    p_value=p_val,
                    cohen_d=d_eff,
                    mean_diff=float(np.mean(scores1) - np.mean(scores2)),
                    comparison_method=pcm,
                )
            )

    col_names = [
        "backbone",
        "comparison",
        "p_value",
        "cohen_d",
        "mean_diff",
        "comparison_method",
    ]
    df_comp = pd.DataFrame(comparison_rows)
    if df_comp.empty:
        df_comp = pd.DataFrame(columns=col_names)
    df_comp.to_csv(stats_dir / "pipeline_comparisons.csv", index=False)
    log.info("Pipeline comparisons and stats written.")

    return ExperimentResult(
        subject_performances=subject_performances,
        pipeline_comparisons=df_comp,
    )
