"""
TEST BLOCK 4 — Backbone interaction (within dataset).

Does denoising (GEDAI or ICALabel) help CSP more than Tangent Space, or vice versa?
Paired comparison across subjects: (Δ for CSP) vs (Δ for Tangent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import permutation_test


def compute_backbone_interaction(
    subject_csv: Path,
    stats_dir: Path,
    dataset_label: Optional[str] = None,
    n_resamples: int = 10_000,
) -> None:
    """
    For each denoising comparison (GEDAI−baseline, ICALabel−baseline):
    - Per subject: delta_csp, delta_tangent.
    - Paired permutation test: is delta_csp different from delta_tangent?
    Writes backbone_interaction.csv.
    """
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(subject_csv)
    if df.empty:
        return

    agg = (
        df.groupby(["subject", "backbone", "pipeline"], as_index=False)["mean_accuracy"]
        .first()
        .rename(columns={"mean_accuracy": "accuracy"})
    )

    pipelines = agg["pipeline"].unique().tolist()
    if "baseline" not in pipelines:
        return

    rows = []
    for comp_name, denoise in [("GEDAI − baseline", "gedai"), ("ICALabel − baseline", "icalabel")]:
        if denoise not in pipelines:
            continue
        # Pivot: one row per subject, columns = csp_baseline, csp_denoise, tangent_baseline, tangent_denoise
        wide = agg.pivot_table(
            index="subject",
            columns=["backbone", "pipeline"],
            values="accuracy",
            aggfunc="first",
        )
        if ("csp", "baseline") not in wide.columns or ("csp", denoise) not in wide.columns:
            continue
        if ("tangent", "baseline") not in wide.columns or ("tangent", denoise) not in wide.columns:
            continue

        delta_csp = wide[("csp", denoise)] - wide[("csp", "baseline")]
        delta_tangent = wide[("tangent", denoise)] - wide[("tangent", "baseline")]
        # Drop subjects with missing
        mask = delta_csp.notna() & delta_tangent.notna()
        delta_csp = delta_csp[mask].values.astype(float)
        delta_tangent = delta_tangent[mask].values.astype(float)
        if len(delta_csp) < 2:
            continue

        def stat(x, y):
            return np.mean(x - y)

        res = permutation_test(
            (delta_csp, delta_tangent),
            stat,
            permutation_type="pairings",
            n_resamples=n_resamples,
            alternative="two-sided",
        )
        mean_diff_csp_minus_tangent = float(np.mean(delta_csp) - np.mean(delta_tangent))
        interpretation = (
            "CSP benefits more" if mean_diff_csp_minus_tangent > 0 else "Tangent benefits more"
        )
        rows.append({
            "dataset": dataset_label or "unknown",
            "comparison": comp_name,
            "mean_delta_csp": float(np.mean(delta_csp)),
            "mean_delta_tangent": float(np.mean(delta_tangent)),
            "mean_diff_csp_minus_tangent": mean_diff_csp_minus_tangent,
            "p_value": res.pvalue,
            "n_subjects": len(delta_csp),
            "interpretation": interpretation,
        })

    if rows:
        pd.DataFrame(rows).to_csv(stats_dir / "backbone_interaction.csv", index=False)
