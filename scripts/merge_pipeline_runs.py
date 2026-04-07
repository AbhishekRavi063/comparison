from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import cohen_d_pooled, paired_permutation_p_value
from src.plots.performance import plot_performance, plot_variability
from src.stats.backbone_interaction import compute_backbone_interaction
from src.stats.variability import compute_variability


def _copy_models(src_models: Path, dst_models: Path) -> None:
    if not src_models.exists():
        return
    dst_models.mkdir(parents=True, exist_ok=True)
    for model_path in src_models.glob("*.joblib"):
        shutil.copy2(model_path, dst_models / model_path.name)


def _compute_pipeline_comparisons(
    subject_csv: Path,
    out_csv: Path,
    n_resamples: int,
) -> None:
    df = pd.read_csv(subject_csv)
    if df.empty:
        pd.DataFrame(
            columns=["backbone", "comparison", "p_value", "cohen_d", "mean_diff"]
        ).to_csv(out_csv, index=False)
        return

    agg = (
        df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_accuracy"]
        .first()
        .rename(columns={"mean_accuracy": "accuracy"})
    )
    pipelines = set(agg["pipeline"].unique().tolist())

    rows = []
    for backbone in agg["backbone"].unique():
        b = agg[agg["backbone"] == backbone]
        for p1, p2 in [
            ("gedai", "baseline"),
            ("icalabel", "baseline"),
            ("gedai", "icalabel"),
            ("pylossless", "baseline"),
            ("gedai", "pylossless"),
            ("icalabel", "pylossless"),
        ]:
            if p1 not in pipelines or p2 not in pipelines:
                continue
            s1 = b[b["pipeline"] == p1].set_index("subject")["accuracy"]
            s2 = b[b["pipeline"] == p2].set_index("subject")["accuracy"]
            common = s1.index.intersection(s2.index)
            if len(common) < 2:
                continue
            x1 = s1.loc[common].values.astype(float)
            x2 = s2.loc[common].values.astype(float)
            p_val = paired_permutation_p_value(x1, x2, n_resamples=n_resamples)
            d = cohen_d_pooled(x1, x2)
            rows.append(
                {
                    "backbone": backbone,
                    "comparison": f"{p1} - {p2}",
                    "p_value": p_val,
                    "cohen_d": d,
                    "mean_diff": float(np.mean(x1) - np.mean(x2)),
                }
            )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame(
            columns=["backbone", "comparison", "p_value", "cohen_d", "mean_diff"]
        )
    out_df.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two-pass benchmark outputs into one results directory.",
    )
    parser.add_argument("--base-results", required=True, help="Results dir with baseline/icalabel.")
    parser.add_argument("--gedai-results", required=True, help="Results dir with gedai-only.")
    parser.add_argument("--out-results", required=True, help="Output merged results dir.")
    parser.add_argument(
        "--pipeline-permutations",
        type=int,
        default=10000,
        help="Resamples for merged pipeline_comparisons.csv.",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default="merged",
        help="Dataset label to write into variability/interactions.",
    )
    args = parser.parse_args()

    base = Path(args.base_results)
    gedai = Path(args.gedai_results)
    out = Path(args.out_results)
    out_tables = out / "tables"
    out_stats = out / "stats"
    out_figs = out / "figures"
    out_models = out / "models"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_stats.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    base_csv = base / "tables" / "subject_level_performance.csv"
    gedai_csv = gedai / "tables" / "subject_level_performance.csv"
    if not base_csv.exists() or not gedai_csv.exists():
        raise FileNotFoundError("Both input subject tables must exist.")

    df_base = pd.read_csv(base_csv)
    df_gedai = pd.read_csv(gedai_csv)
    merged = pd.concat([df_base, df_gedai], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["subject", "backbone", "pipeline", "fold"], keep="last"
    )
    out_subject = out_tables / "subject_level_performance.csv"
    merged.to_csv(out_subject, index=False)

    _compute_pipeline_comparisons(
        subject_csv=out_subject,
        out_csv=out_stats / "pipeline_comparisons.csv",
        n_resamples=args.pipeline_permutations,
    )
    compute_variability(out_subject, out_stats, dataset_label=args.dataset_label)
    compute_backbone_interaction(out_subject, out_stats, dataset_label=args.dataset_label)
    plot_performance(out_subject, out_figs)
    plot_variability(out_subject, out_figs)

    _copy_models(base / "models", out_models)
    _copy_models(gedai / "models", out_models)


if __name__ == "__main__":
    main()
