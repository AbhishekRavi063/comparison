"""Merge subject_level_performance.csv from parallel shard runs and recompute group stats.

After multiple ``python -m src.run_all`` invocations with disjoint ``--subjects`` and
distinct ``--results-root`` directories, run:

  python -m src.merge_sharded_results \\
    --shards results/alljoined_w01 results/alljoined_w02 \\
    --out results/alljoined_merged \\
    --n-pipeline-perm 10000

Reads each shard's ``tables/subject_level_performance.csv``, concatenates, writes
``out/tables/subject_level_performance.csv`` and ``out/stats/pipeline_comparisons.csv``.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .evaluation.metrics import cohen_d_pooled, paired_permutation_p_value


def _per_subject_means(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, backbone, pipeline) with mean_accuracy."""
    return (
        df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_accuracy"]
        .first()
        .sort_values(["subject", "backbone", "pipeline"])
        .reset_index(drop=True)
    )


def merge_pipeline_comparisons(
    df_subject: pd.DataFrame,
    n_pipeline_perm: int,
    pipelines: List[str],
) -> pd.DataFrame:
    """Match experiment.run_experiment between-pipeline logic (paired tests on subject means)."""
    g = _per_subject_means(df_subject)
    backbones = sorted(g["backbone"].unique().tolist())
    comparison_pairs: List[Tuple[str, str]] = [
        ("gedai", "baseline"),
        ("icalabel", "baseline"),
        ("gedai", "icalabel"),
    ]
    rows = []
    for backbone in backbones:
        for p1, p2 in comparison_pairs:
            if p1 not in pipelines or p2 not in pipelines:
                continue
            d1 = g[(g["backbone"] == backbone) & (g["pipeline"] == p1)].set_index(
                "subject"
            )["mean_accuracy"]
            d2 = g[(g["backbone"] == backbone) & (g["pipeline"] == p2)].set_index(
                "subject"
            )["mean_accuracy"]
            common = sorted(d1.index.intersection(d2.index))
            if len(common) < 2:
                continue
            scores1 = d1.loc[common].values.astype(float)
            scores2 = d2.loc[common].values.astype(float)
            p_val = paired_permutation_p_value(
                scores1, scores2, n_resamples=n_pipeline_perm
            )
            d_eff = cohen_d_pooled(scores1, scores2)
            rows.append(
                {
                    "backbone": backbone,
                    "comparison": f"{p1} - {p2}",
                    "p_value": p_val,
                    "cohen_d": d_eff,
                    "mean_diff": float(scores1.mean() - scores2.mean()),
                }
            )
    col_names = ["backbone", "comparison", "p_value", "cohen_d", "mean_diff"]
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=col_names)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge benchmark CSVs from sharded run_all processes."
    )
    parser.add_argument(
        "--shards",
        type=str,
        nargs="+",
        required=True,
        help="Result directories (each must contain tables/subject_level_performance.csv).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for merged tables/ and stats/.",
    )
    parser.add_argument(
        "--n-pipeline-perm",
        type=int,
        default=10_000,
        help="Resamples for paired pipeline permutation tests (match your YAML).",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="baseline,icalabel,gedai",
        help="Comma-separated pipelines that were enabled (for comparison table).",
    )
    parser.add_argument(
        "--copy-models",
        action="store_true",
        help="Also copy results/shard/models/*.joblib into out/models/ (name collisions must not occur).",
    )
    args = parser.parse_args()

    shard_paths = [Path(s).resolve() for s in args.shards]
    out_root = Path(args.out).resolve()
    tables_out = out_root / "tables"
    stats_out = out_root / "stats"
    tables_out.mkdir(parents=True, exist_ok=True)
    stats_out.mkdir(parents=True, exist_ok=True)

    pieces = []
    for sp in shard_paths:
        csv_path = sp / "tables" / "subject_level_performance.csv"
        if not csv_path.is_file():
            raise SystemExit(f"Missing {csv_path}")
        pieces.append(pd.read_csv(csv_path))

    merged = pd.concat(pieces, ignore_index=True)
    # Same subject should not appear twice across shards
    dup = merged.duplicated(subset=["subject", "backbone", "pipeline", "fold"], keep=False)
    if dup.any():
        raise SystemExit(
            "Duplicate (subject, backbone, pipeline, fold) rows after merge — "
            "check shard subject splits."
        )

    merged.to_csv(tables_out / "subject_level_performance.csv", index=False)

    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    df_comp = merge_pipeline_comparisons(merged, args.n_pipeline_perm, pipelines)
    df_comp.to_csv(stats_out / "pipeline_comparisons.csv", index=False)

    if args.copy_models:
        models_out = out_root / "models"
        models_out.mkdir(parents=True, exist_ok=True)
        for sp in shard_paths:
            md = sp / "models"
            if not md.is_dir():
                continue
            for f in md.glob("*.joblib"):
                dest = models_out / f.name
                if dest.exists():
                    raise SystemExit(f"Model name collision: {dest}")
                shutil.copy2(f, dest)

    print(f"Wrote {tables_out / 'subject_level_performance.csv'}")
    print(f"Wrote {stats_out / 'pipeline_comparisons.csv'}")


if __name__ == "__main__":
    main()
