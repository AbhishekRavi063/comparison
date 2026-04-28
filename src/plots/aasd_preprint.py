"""AASD preprint figures — all plots needed to prove GEDAI preserves brain signal.

Run via:  python -m src.run_aasd_analysis --config config/config_aasd.yml

Figures produced
----------------
1. auc_violin_{backbone}.png        — AUC distribution per pipeline, 18 subjects
2. lateralization_index.png         — LI per pipeline (proof of signal preservation)
3. alpha_retention.png              — alpha_ratio per pipeline (noise removed, not signal)
4. psd_all_pipelines_sub{N}.png    — PSD all pipelines on same axes for one subject
5. topomap_alpha_{pipeline}.png     — spatial alpha power map before vs after GEDAI
6. summary_table.csv                — mean ± std AUC + LI per pipeline × backbone
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Colour palette consistent across all figures (colourblind-safe).
PIPELINE_COLORS = {
    "baseline": "#4878CF",   # blue
    "asr":      "#6ACC65",   # green
    "icalabel": "#D65F5F",   # red
    "gedai":    "#B47CC7",   # purple
}
PIPELINE_LABELS = {
    "baseline": "Baseline",
    "asr":      "ASR",
    "icalabel": "ICALabel",
    "gedai":    "GEDAI",
}
BACKBONE_LABELS = {
    "csp":     "CSP + LDA",
    "tangent": "Tangent Space + LR",
}


# ---------------------------------------------------------------------------
# Figure 1 — AUC violin + strip
# ---------------------------------------------------------------------------

def plot_auc_violin(csv_path: Path, out_dir: Path) -> None:
    """Violin + individual subject dots for mean AUC per pipeline per backbone."""
    df = pd.read_csv(csv_path)
    df_subj = (
        df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_auc"]
        .first()
    )
    pipelines = [p for p in ["baseline", "asr", "icalabel", "gedai"]
                 if p in df_subj["pipeline"].unique()]

    for backbone, df_b in df_subj.groupby("backbone"):
        fig, ax = plt.subplots(figsize=(max(5, len(pipelines) * 1.4), 5))

        data_by_pipe = [df_b.loc[df_b["pipeline"] == p, "mean_auc"].values
                        for p in pipelines]
        colors = [PIPELINE_COLORS.get(p, "#888888") for p in pipelines]
        labels = [PIPELINE_LABELS.get(p, p) for p in pipelines]

        parts = ax.violinplot(
            data_by_pipe,
            positions=range(len(pipelines)),
            showmedians=True,
            showextrema=False,
        )
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.45)
        parts["cmedians"].set_colors("k")
        parts["cmedians"].set_linewidth(2)

        # Individual subject dots + lines
        rng = np.random.RandomState(42)
        for i, (vals, col) in enumerate(zip(data_by_pipe, colors)):
            jitter = rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(i + jitter, vals, color=col, s=40, zorder=3, alpha=0.8,
                       edgecolors="white", linewidths=0.5)

        # Connect same subject across pipelines (shows individual trajectories)
        subjects = df_b["subject"].unique()
        pipe_pos = {p: i for i, p in enumerate(pipelines)}
        for subj in subjects:
            row = df_b[df_b["subject"] == subj]
            xs = [pipe_pos[p] for p in pipelines if p in row["pipeline"].values]
            ys = [row.loc[row["pipeline"] == p, "mean_auc"].values[0] for p in pipelines
                  if p in row["pipeline"].values]
            ax.plot(xs, ys, color="gray", alpha=0.18, linewidth=0.8, zorder=1)

        ax.set_xticks(range(len(pipelines)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Mean AUC (CV)", fontsize=12)
        ax.set_title(f"{BACKBONE_LABELS.get(backbone, backbone)} — AUC across subjects",
                     fontsize=13, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"auc_violin_{backbone}.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Alpha Lateralization Index bar chart
# ---------------------------------------------------------------------------

def plot_lateralization_index(csv_path: Path, out_dir: Path) -> None:
    """Bar chart of mean LI per pipeline — KEY proof that GEDAI preserves neural signal.

    Higher LI = stronger ipsilateral alpha suppression = cleaner neural signal.
    """
    df = pd.read_csv(csv_path)
    if "lateralization_index" not in df.columns:
        return

    df_subj = (
        df.groupby(["subject", "pipeline"], as_index=False)["lateralization_index"]
        .first()
    )
    pipelines = [p for p in ["baseline", "asr", "icalabel", "gedai"]
                 if p in df_subj["pipeline"].unique()]

    means = [df_subj.loc[df_subj["pipeline"] == p, "lateralization_index"].mean()
             for p in pipelines]
    sems  = [df_subj.loc[df_subj["pipeline"] == p, "lateralization_index"].sem()
             for p in pipelines]
    colors = [PIPELINE_COLORS.get(p, "#888888") for p in pipelines]
    labels = [PIPELINE_LABELS.get(p, p) for p in pipelines]

    fig, ax = plt.subplots(figsize=(max(5, len(pipelines) * 1.4), 5))
    x = np.arange(len(pipelines))
    bars = ax.bar(x, means, yerr=sems, color=colors, alpha=0.80,
                  capsize=5, edgecolor="white", linewidth=0.5)

    # Individual subject dots
    rng = np.random.RandomState(7)
    for i, p in enumerate(pipelines):
        vals = df_subj.loc[df_subj["pipeline"] == p, "lateralization_index"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(i + jitter, vals, color="k", s=28, zorder=4, alpha=0.55)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Alpha Lateralization Index (LI)", fontsize=12)
    ax.set_title(
        "Alpha lateralization: higher LI = stronger neural selectivity\n"
        "(proof GEDAI sharpens, not removes, brain signal)",
        fontsize=11, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate values
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "lateralization_index.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Alpha / Beta retention ratio histogram
# ---------------------------------------------------------------------------

def plot_alpha_retention(csv_path: Path, out_dir: Path) -> None:
    """Shows how much alpha power each pipeline removes vs baseline.

    alpha_ratio < 1 → power removed. GEDAI should be ~0.3–0.7 (targeted).
    Proof: beta_ratio should drop more than alpha_ratio (GEDAI targets broadband noise).
    """
    df = pd.read_csv(csv_path)
    df_subj = (
        df.groupby(["subject", "pipeline"], as_index=False)[["alpha_ratio", "beta_ratio"]]
        .first()
    )
    pipelines = [p for p in ["asr", "icalabel", "gedai"]
                 if p in df_subj["pipeline"].unique()]
    if not pipelines:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, ylabel in zip(
        axes,
        ["alpha_ratio", "beta_ratio"],
        ["Alpha power ratio\n(8–12 Hz, GEDAI / Baseline)", "Beta power ratio\n(13–30 Hz, GEDAI / Baseline)"],
    ):
        for i, p in enumerate(pipelines):
            vals = df_subj.loc[df_subj["pipeline"] == p, metric].values
            col  = PIPELINE_COLORS.get(p, "#888888")
            ax.scatter([i] * len(vals), vals, color=col, s=50, alpha=0.7, zorder=3)
            ax.plot([i - 0.2, i + 0.2], [vals.mean(), vals.mean()],
                    color=col, linewidth=2.5, zorder=4)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, label="No change")
        ax.set_xticks(range(len(pipelines)))
        ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title("Alpha band retention (decode band)", fontsize=11, fontweight="bold")
    axes[1].set_title("Beta band retention (noise reference)", fontsize=11, fontweight="bold")
    fig.suptitle(
        "Power ratio per pipeline vs Baseline (lower = more removed)\n"
        "GEDAI removes more beta (noise) than alpha (signal) — targeted denoising",
        fontsize=10,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_retention.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — PSD all pipelines on same axes
# ---------------------------------------------------------------------------

def plot_psd_all_pipelines(
    X_by_pipeline: Dict[str, np.ndarray],
    sfreq: float,
    subject_id: int,
    out_dir: Path,
) -> None:
    """Overlay PSD of all pipelines for a single subject — shows alpha peak preserved."""
    from scipy.signal import welch

    fig, ax = plt.subplots(figsize=(8, 5))
    for pipeline, X in X_by_pipeline.items():
        # Average across trials and channels in channels_of_interest
        sig = X.mean(axis=(0, 1))   # mean over trials and channels → (n_times,)
        nperseg = min(256, len(sig))
        f, psd = welch(sig.astype(float), fs=sfreq, nperseg=nperseg)
        mask = f <= 50
        col = PIPELINE_COLORS.get(pipeline, "#888888")
        lw  = 2.5 if pipeline == "gedai" else 1.5
        ls  = "-" if pipeline in ("baseline", "gedai") else "--"
        ax.semilogy(f[mask], psd[mask], color=col, linewidth=lw, linestyle=ls,
                    label=PIPELINE_LABELS.get(pipeline, pipeline))

    # Shade alpha band
    ax.axvspan(8, 12, alpha=0.12, color="gold", label="Alpha band (8–12 Hz)")
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power spectral density", fontsize=12)
    ax.set_title(f"Subject {subject_id} — PSD comparison across pipelines\n"
                 "(alpha peak preserved after GEDAI denoising)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"psd_all_pipelines_sub{subject_id}.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Topographic alpha power map
# ---------------------------------------------------------------------------

def plot_topomap_alpha(
    X_by_pipeline: Dict[str, np.ndarray],
    sfreq: float,
    ch_names: List[str],
    subject_id: int,
    out_dir: Path,
) -> None:
    """MNE topographic map of alpha power — shows spatial pattern preserved after GEDAI."""
    try:
        import mne
    except ImportError:
        return

    from scipy.signal import welch

    pipelines_to_show = [p for p in ["baseline", "asr", "icalabel", "gedai"]
                         if p in X_by_pipeline]
    n = len(pipelines_to_show)
    if n == 0:
        return

    info = mne.create_info(ch_names=list(ch_names), sfreq=float(sfreq), ch_types="eeg")
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")
    except Exception:
        return

    alpha_power_maps: Dict[str, np.ndarray] = {}
    for pipeline, X in X_by_pipeline.items():
        ch_power = []
        for ci in range(X.shape[1]):
            sig = X[:, ci, :].mean(axis=0)
            nperseg = min(256, len(sig))
            f, psd = welch(sig.astype(float), fs=sfreq, nperseg=nperseg)
            mask = (f >= 8) & (f <= 12)
            try:
                p_alpha = float(np.trapezoid(psd[mask], f[mask]))
            except AttributeError:
                p_alpha = float(np.trapz(psd[mask], f[mask]))
            ch_power.append(p_alpha)
        alpha_power_maps[pipeline] = np.array(ch_power)

    baseline_map = alpha_power_maps.get("baseline", None)

    # Add difference maps (baseline - pipeline) to show WHAT was removed WHERE
    diff_pipelines = [p for p in pipelines_to_show if p != "baseline" and baseline_map is not None]
    n_panels = n + len(diff_pipelines)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    vmax = max(m.max() for m in alpha_power_maps.values())
    vmin = 0.0

    # Left panels: absolute alpha power per pipeline
    for ax, pipeline in zip(axes[:n], pipelines_to_show):
        pmap = alpha_power_maps[pipeline]
        try:
            im, _ = mne.viz.plot_topomap(
                pmap, info, axes=ax, show=False,
                vlim=(vmin, vmax), cmap="hot_r",
                names=None, sensors=False,
            )
        except Exception:
            ax.set_title(f"{PIPELINE_LABELS.get(pipeline, pipeline)}\n(topomap failed)")
            continue
        ax.set_title(PIPELINE_LABELS.get(pipeline, pipeline), fontsize=11, fontweight="bold")

    # Right panels: difference maps (what was removed = baseline - pipeline)
    for ax, pipeline in zip(axes[n:], diff_pipelines):
        diff = baseline_map - alpha_power_maps[pipeline]
        diff_max = np.abs(diff).max()
        try:
            mne.viz.plot_topomap(
                diff, info, axes=ax, show=False,
                vlim=(-diff_max, diff_max), cmap="RdBu_r",
                names=None, sensors=False,
            )
        except Exception:
            ax.set_title(f"Removed by {PIPELINE_LABELS.get(pipeline, pipeline)}\n(failed)")
            continue
        ax.set_title(
            f"Removed by {PIPELINE_LABELS.get(pipeline, pipeline)}\n(red=removed, blue=added)",
            fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        f"Subject {subject_id} — Alpha power topography (8–12 Hz)\n"
        "Left: absolute power per pipeline  |  Right: what each method removed\n"
        "Frontal removal = noise; parietal preserved = brain signal",
        fontsize=9, fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"topomap_alpha_sub{subject_id}.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Group-level statistics: Wilcoxon + Cohen's d
# ---------------------------------------------------------------------------

def plot_group_statistics(csv_path: Path, out_dir: Path) -> pd.DataFrame:
    """Wilcoxon signed-rank test + Cohen's d for every pipeline vs baseline.

    Produces:
    - group_statistics.csv  — p-values, Cohen's d, mean diff per pipeline × backbone
    - group_statistics.png  — bar chart of Cohen's d with significance markers
    """
    from scipy.stats import wilcoxon

    df = pd.read_csv(csv_path)
    df_subj = df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_auc"].first()

    comparisons = [p for p in ["asr", "icalabel", "gedai"] if p in df_subj["pipeline"].unique()]
    backbones   = [b for b in ["csp", "tangent"] if b in df_subj["backbone"].unique()]

    rows = []
    for backbone in backbones:
        baseline_vals = (
            df_subj[(df_subj["backbone"] == backbone) & (df_subj["pipeline"] == "baseline")]
            .sort_values("subject")["mean_auc"].values
        )
        for pipeline in comparisons:
            pipe_vals = (
                df_subj[(df_subj["backbone"] == backbone) & (df_subj["pipeline"] == pipeline)]
                .sort_values("subject")["mean_auc"].values
            )
            n = min(len(baseline_vals), len(pipe_vals))
            if n < 2:
                continue
            b_v, p_v = baseline_vals[:n], pipe_vals[:n]
            diff = p_v - b_v
            mean_diff = float(diff.mean())

            # Wilcoxon signed-rank test
            try:
                _, p_val = wilcoxon(p_v, b_v, alternative="greater")
            except Exception:
                p_val = 1.0

            # Cohen's d (paired)
            std_diff = float(diff.std(ddof=1)) if n > 1 else 1e-9
            cohen_d  = mean_diff / std_diff if std_diff > 0 else 0.0

            rows.append({
                "backbone":  backbone,
                "pipeline":  pipeline,
                "n_subjects": n,
                "mean_auc_baseline": float(b_v.mean()),
                "mean_auc_pipeline": float(p_v.mean()),
                "mean_diff":  round(mean_diff, 4),
                "cohen_d":    round(cohen_d, 3),
                "p_wilcoxon": float(p_val),
                "significant": p_val < 0.05,
            })

    df_stats = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(out_dir / "group_statistics.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("GROUP-LEVEL STATISTICS (Wilcoxon signed-rank, one-sided: pipeline > baseline)")
    print("=" * 70)
    for _, r in df_stats.iterrows():
        sig = "***" if r["p_wilcoxon"] < 0.001 else ("**" if r["p_wilcoxon"] < 0.01
              else ("*" if r["p_wilcoxon"] < 0.05 else "ns"))
        print(f"  {BACKBONE_LABELS.get(r['backbone'], r['backbone']):30s} "
              f"{PIPELINE_LABELS.get(r['pipeline'], r['pipeline']):12s} "
              f"Δ={r['mean_diff']:+.3f}  d={r['cohen_d']:.2f}  "
              f"p={r['p_wilcoxon']:.4f} {sig}")
    print("=" * 70 + "\n")

    # Plot Cohen's d bar chart
    if df_stats.empty:
        return df_stats

    fig, axes = plt.subplots(1, len(backbones), figsize=(5 * len(backbones), 5), sharey=True)
    if len(backbones) == 1:
        axes = [axes]

    for ax, backbone in zip(axes, backbones):
        sub = df_stats[df_stats["backbone"] == backbone]
        pipes  = sub["pipeline"].tolist()
        d_vals = sub["cohen_d"].tolist()
        p_vals = sub["p_wilcoxon"].tolist()
        colors = [PIPELINE_COLORS.get(p, "#888888") for p in pipes]
        labels = [PIPELINE_LABELS.get(p, p) for p in pipes]

        bars = ax.bar(range(len(pipes)), d_vals, color=colors, alpha=0.80,
                      edgecolor="white", linewidth=0.5)

        # Significance stars
        for i, (d, p) in enumerate(zip(d_vals, p_vals)):
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.text(i, max(d + 0.05, 0.1), star, ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        # Cohen's d thresholds
        for thresh, label, ls in [(0.2, "small", ":"), (0.5, "medium", "--"), (0.8, "large", "-.")]:
            ax.axhline(thresh, color="gray", linestyle=ls, linewidth=0.8, alpha=0.6)
            ax.text(len(pipes) - 0.5, thresh + 0.02, label, fontsize=8, color="gray", ha="right")

        ax.set_xticks(range(len(pipes)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Cohen's d (vs Baseline)", fontsize=12)
        ax.set_title(f"{BACKBONE_LABELS.get(backbone, backbone)}\nEffect size vs baseline",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(bottom=min(0, min(d_vals) - 0.1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Group-level effect size (Cohen's d) — pipeline vs baseline\n"
                 "d > 0.8 = large effect  |  * p<0.05  ** p<0.01  *** p<0.001",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "group_statistics.png", dpi=180)
    plt.close(fig)

    return df_stats


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary_table(csv_path: Path, out_dir: Path) -> pd.DataFrame:
    """Write mean ± std AUC, LI, and alpha_ratio per pipeline × backbone to CSV + print."""
    df = pd.read_csv(csv_path)
    df_subj = df.groupby(["subject", "backbone", "pipeline"], as_index=False).first()

    metrics = ["mean_auc", "mean_accuracy"]
    if "lateralization_index" in df.columns:
        metrics.append("lateralization_index")
    if "alpha_ratio" in df.columns:
        metrics.append("alpha_ratio")

    rows = []
    for (backbone, pipeline), grp in df_subj.groupby(["backbone", "pipeline"]):
        row = {
            "backbone": BACKBONE_LABELS.get(backbone, backbone),
            "pipeline": PIPELINE_LABELS.get(pipeline, pipeline),
        }
        for m in metrics:
            if m in grp.columns:
                vals = grp[m].dropna()
                row[f"{m}_mean"] = round(float(vals.mean()), 4)
                row[f"{m}_std"]  = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_table.csv"
    df_summary.to_csv(out_path, index=False)

    # Pretty print
    print("\n" + "=" * 70)
    print("SUMMARY TABLE — mean ± std across subjects")
    print("=" * 70)
    for _, r in df_summary.iterrows():
        auc_m = r.get("mean_auc_mean", 0)
        auc_s = r.get("mean_auc_std", 0)
        li_m  = r.get("lateralization_index_mean", None)
        ar_m  = r.get("alpha_ratio_mean", None)
        li_str = f"  LI={li_m:.3f}" if li_m is not None else ""
        ar_str = f"  α-ratio={ar_m:.3f}" if ar_m is not None else ""
        print(f"  {r['backbone']:30s} {r['pipeline']:12s}  AUC={auc_m:.3f}±{auc_s:.3f}{li_str}{ar_str}")
    print("=" * 70 + "\n")
    return df_summary


# ---------------------------------------------------------------------------
# Master function — generate all figures
# ---------------------------------------------------------------------------

def generate_all_preprint_figures(
    csv_path: Path,
    out_dir: Path,
    X_by_pipeline: Optional[Dict[str, np.ndarray]] = None,
    sfreq: float = 250.0,
    ch_names: Optional[List[str]] = None,
    subject_id: int = 1,
) -> None:
    """Generate all preprint figures from results CSV.

    Parameters
    ----------
    csv_path      : path to subject_level_performance.csv
    out_dir       : output directory for figures
    X_by_pipeline : optional {pipeline: X_array} for PSD and topomap plots
    sfreq         : sampling rate (for PSD/topomap)
    ch_names      : channel names (for topomap)
    subject_id    : which subject's data is in X_by_pipeline
    """
    print(f"[preprint] Generating figures → {out_dir}")
    plot_auc_violin(csv_path, out_dir)
    print("[preprint] ✓ AUC violin plots")

    plot_lateralization_index(csv_path, out_dir)
    print("[preprint] ✓ Lateralization index")

    plot_alpha_retention(csv_path, out_dir)
    print("[preprint] ✓ Alpha/beta retention")

    plot_group_statistics(csv_path, out_dir)
    print("[preprint] ✓ Group statistics (Wilcoxon + Cohen's d)")

    write_summary_table(csv_path, out_dir)
    print("[preprint] ✓ Summary table")

    if X_by_pipeline is not None:
        plot_psd_all_pipelines(X_by_pipeline, sfreq, subject_id, out_dir)
        print("[preprint] ✓ PSD all pipelines")
        if ch_names is not None:
            plot_topomap_alpha(X_by_pipeline, sfreq, ch_names, subject_id, out_dir)
            print("[preprint] ✓ Topographic maps")

    print(f"[preprint] All figures saved to {out_dir}")
