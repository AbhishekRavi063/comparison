from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_performance(subject_csv: Path, out_dir: Path) -> None:
    """Create backbone-wise performance plots from the subject-level table.

    Parameters
    ----------
    subject_csv : Path
        Path to subject_level_performance.csv.
    out_dir : Path
        Directory where figures will be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(subject_csv)
    if df.empty:
        return

    # One row per (subject, backbone, pipeline)
    df_subj = (
        df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_accuracy"]
        .first()
        .rename(columns={"mean_accuracy": "accuracy"})
    )

    sns.set(style="whitegrid", context="talk")

    for backbone, df_b in df_subj.groupby("backbone"):
        plt.figure(figsize=(6, 4))
        ax = sns.boxplot(
            data=df_b,
            x="pipeline",
            y="accuracy",
            color="lightgray",
            showfliers=False,
        )
        sns.stripplot(
            data=df_b,
            x="pipeline",
            y="accuracy",
            hue="subject",
            dodge=True,
            ax=ax,
            palette="tab10",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean CV accuracy")
        ax.set_title(f"Performance by pipeline – {backbone.upper()}")
        ax.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        fname = out_dir / f"performance_{backbone}.png"
        plt.savefig(fname, dpi=150)
        plt.close()


def plot_variability(subject_csv: Path, out_dir: Path) -> None:
    """Plot variability: distribution of delta accuracy (denoising − baseline) per backbone.

    Produces one figure per backbone showing Δ(GEDAI−baseline) and Δ(ICALabel−baseline)
    across subjects (histogram or boxplot), matching the professor's variability analysis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(subject_csv)
    if df.empty:
        return

    df_subj = (
        df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_accuracy"]
        .first()
        .rename(columns={"mean_accuracy": "accuracy"})
    )

    pipelines = df_subj["pipeline"].unique().tolist()
    if "baseline" not in pipelines:
        return

    sns.set(style="whitegrid", context="talk")

    for backbone, df_b in df_subj.groupby("backbone"):
        base = df_b[df_b["pipeline"] == "baseline"].set_index("subject")["accuracy"]

        deltas = []
        for pipe in ["gedai", "icalabel"]:
            if pipe not in pipelines:
                continue
            acc = df_b[df_b["pipeline"] == pipe].set_index("subject")["accuracy"]
            delta = (acc - base).dropna()
            if delta.empty:
                continue
            deltas.append(pd.DataFrame({"delta": delta, "comparison": f"{pipe} − baseline"}))

        if not deltas:
            continue

        df_delta = pd.concat(deltas, ignore_index=True)
        plt.figure(figsize=(6, 4))
        ax = sns.boxplot(data=df_delta, x="comparison", y="delta", color="lightsteelblue")
        sns.stripplot(data=df_delta, x="comparison", y="delta", color="black", alpha=0.6, ax=ax)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_ylabel("Δ accuracy (denoising − baseline)")
        ax.set_title(f"Variability: improvement over baseline – {backbone.upper()}")
        plt.tight_layout()
        plt.savefig(out_dir / f"variability_{backbone}.png", dpi=150)
        plt.close()

