"""
TEST BLOCK 3 — Subject-level variability (per dataset).

Computes Δ(GEDAI − Baseline), Δ(ICALabel − Baseline) per subject,
and across subjects: % improved, % worsened, median Δ, SD.
Do not pool across datasets; run once per dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def compute_variability(
    subject_csv: Path,
    stats_dir: Path,
    dataset_label: Optional[str] = None,
) -> None:
    """
    Read subject_level_performance.csv; compute deltas and variability summary;
    write subject_level_deltas.csv and variability_summary.csv under stats_dir.
    """
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(subject_csv)
    if df.empty:
        return

    # One row per (subject, backbone, pipeline) with mean_accuracy
    agg = (
        df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_accuracy"]
        .first()
        .rename(columns={"mean_accuracy": "accuracy"})
    )

    pipelines = agg["pipeline"].unique().tolist()
    if "baseline" not in pipelines:
        return

    # Per-subject deltas
    rows = []
    for (subject, backbone), g in agg.groupby(["subject", "backbone"]):
        base = g[g["pipeline"] == "baseline"]["accuracy"]
        if base.empty:
            continue
        base_val = base.iloc[0]
        row = {"subject": subject, "backbone": backbone}
        if "gedai" in pipelines:
            gedai_vals = g[g["pipeline"] == "gedai"]["accuracy"]
            row["delta_gedai_baseline"] = (gedai_vals.iloc[0] - base_val) if not gedai_vals.empty else None
        if "icalabel" in pipelines:
            ica_vals = g[g["pipeline"] == "icalabel"]["accuracy"]
            row["delta_icalabel_baseline"] = (ica_vals.iloc[0] - base_val) if not ica_vals.empty else None
        rows.append(row)

    df_deltas = pd.DataFrame(rows)
    if not df_deltas.empty:
        out_deltas = stats_dir / "subject_level_deltas.csv"
        df_deltas.to_csv(out_deltas, index=False)

    # Variability summary: % improved, % worsened, median Δ, SD (per backbone, per comparison)
    summary_rows = []
    for backbone in df_deltas["backbone"].unique():
        b = df_deltas[df_deltas["backbone"] == backbone]
        for col in ["delta_gedai_baseline", "delta_icalabel_baseline"]:
            if col not in b.columns or b[col].isna().all():
                continue
            d = b[col].dropna()
            if d.empty:
                continue
            n = len(d)
            n_improved = (d > 0).sum()
            n_worsened = (d < 0).sum()
            summary_rows.append({
                "dataset": dataset_label or "unknown",
                "backbone": backbone,
                "comparison": col.replace("delta_", "").replace("_baseline", " − baseline"),
                "n_subjects": n,
                "pct_improved": 100.0 * n_improved / n,
                "pct_worsened": 100.0 * n_worsened / n,
                "median_delta": d.median(),
                "sd_delta": d.std(),
                "mean_delta": d.mean(),
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(stats_dir / "variability_summary.csv", index=False)

    # TEST BLOCK 1 summary: above-chance validation (how many subjects p_empirical < 0.05)
    agg2 = df.groupby(["subject", "backbone", "pipeline"], as_index=False).first()
    above = agg2[agg2["p_empirical"] < 0.05].groupby(["backbone", "pipeline"]).size().reset_index(name="n_above_chance")
    total = agg2.groupby(["backbone", "pipeline"]).size().reset_index(name="n_subjects")
    ac_summary = total.merge(above, on=["backbone", "pipeline"], how="left").fillna(0)
    ac_summary["n_above_chance"] = ac_summary["n_above_chance"].astype(int)
    ac_summary["pct_above_chance"] = 100.0 * ac_summary["n_above_chance"] / ac_summary["n_subjects"]
    if "dataset" in ac_summary.columns:
        pass
    else:
        ac_summary.insert(0, "dataset", dataset_label or "unknown")
    ac_summary.to_csv(stats_dir / "above_chance_summary.csv", index=False)
