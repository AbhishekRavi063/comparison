"""
Validation summary: print mean accuracy per pipeline, p-values, Cohen's d, % improved by GEDAI.

Used after a 5-subject (or any) run to verify implementation:
- Mean accuracy per pipeline (per backbone)
- Pipeline comparison p-values and Cohen's d
- % subjects improved by GEDAI vs baseline
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def print_validation_summary(
    results_root: Path,
    dataset_label: Optional[str] = None,
) -> None:
    """
    Read tables and stats from results_root; print validation summary to stdout.

    Expects:
    - results_root/tables/subject_level_performance.csv
    - results_root/stats/pipeline_comparisons.csv
    """
    results_root = Path(results_root)
    tables_dir = results_root / "tables"
    stats_dir = results_root / "stats"
    subject_csv = tables_dir / "subject_level_performance.csv"
    pipe_csv = stats_dir / "pipeline_comparisons.csv"

    if not subject_csv.exists():
        print(f"[Validation] No {subject_csv}; skip summary.")
        return

    import pandas as pd

    df = pd.read_csv(subject_csv)
    if df.empty:
        print("[Validation] Subject table empty; skip summary.")
        return

    label = dataset_label or results_root.name
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY — {label}")
    print("=" * 60)

    # One row per (subject, backbone, pipeline)
    agg = (
        df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_accuracy"]
        .first()
        .rename(columns={"mean_accuracy": "accuracy"})
    )
    n_subjects = agg["subject"].nunique()

    # Mean accuracy per pipeline (per backbone)
    print("\n--- Mean accuracy per pipeline (per backbone) ---")
    for backbone in agg["backbone"].unique():
        b = agg[agg["backbone"] == backbone]
        print(f"  {backbone.upper()}:")
        for pipeline in b["pipeline"].unique():
            accs = b[b["pipeline"] == pipeline]["accuracy"]
            mean_acc = accs.mean()
            std_acc = accs.std(ddof=1) if len(accs) > 1 else 0.0
            print(f"    {pipeline}: mean = {mean_acc:.4f}  (std = {std_acc:.4f}, n = {len(accs)})")

    # Pipeline comparisons: p-values and Cohen's d
    if pipe_csv.exists():
        comp = pd.read_csv(pipe_csv)
        if not comp.empty:
            print("\n--- Between-pipeline (paired permutation) — p-value, Cohen's d, mean diff ---")
            for _, row in comp.iterrows():
                print(
                    f"  {row['backbone']} | {row['comparison']}: "
                    f"p = {row['p_value']:.4f}, Cohen's d = {row['cohen_d']:.4f}, "
                    f"mean_diff = {row['mean_diff']:.4f}"
                )

    # % subjects improved by GEDAI vs baseline
    if "baseline" in agg["pipeline"].values and "gedai" in agg["pipeline"].values:
        print("\n--- % subjects improved by GEDAI (vs baseline) ---")
        for backbone in agg["backbone"].unique():
            b = agg[agg["backbone"] == backbone]
            base = b[b["pipeline"] == "baseline"].set_index("subject")["accuracy"]
            gedai = b[b["pipeline"] == "gedai"].set_index("subject")["accuracy"]
            delta = (gedai - base).dropna()
            if len(delta) > 0:
                pct_improved = 100.0 * (delta > 0).sum() / len(delta)
                pct_worsened = 100.0 * (delta < 0).sum() / len(delta)
                print(f"  {backbone.upper()}: {pct_improved:.1f}% improved, {pct_worsened:.1f}% worsened (n = {len(delta)})")
    else:
        print("\n--- % subjects improved by GEDAI: (baseline or gedai missing) ---")

    print(f"\nTotal subjects: {n_subjects}")
    print("=" * 60 + "\n")
