"""
STEP 2 — Cross-dataset consistency (no pooling).

After running each dataset independently, interpret consistency:
- Does GEDAI improve in both? Similar magnitude? Similar variability?
- Do NOT pool subjects across datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def write_cross_dataset_report(
    dataset_results: List[Tuple[str, Path]],
    out_path: Path,
) -> None:
    """
    dataset_results: list of (dataset_label, results_root), e.g. [("physionet_eegbci", Path("results/physionet_eegbci")), ("bnci2014_001", Path("results/bnci2014_001"))].
    Writes a qualitative consistency report to out_path (e.g. .md or .txt).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Cross-dataset consistency report",
        "",
        "**Important:** Subjects are NOT pooled across datasets. Each dataset was evaluated independently; here we interpret consistency of findings.",
        "",
    ]

    summaries = []
    for label, results_root in dataset_results:
        results_root = Path(results_root)
        stats_dir = results_root / "stats"
        pipe_csv = stats_dir / "pipeline_comparisons.csv"
        var_csv = stats_dir / "variability_summary.csv"

        if not pipe_csv.exists():
            lines.append(f"## {label}\n\nNo pipeline_comparisons.csv found.\n")
            continue

        import pandas as pd
        try:
            comp = pd.read_csv(pipe_csv)
        except pd.errors.EmptyDataError:
            lines.append(f"## {label}\n\nNo between-pipeline comparisons (pipeline_comparisons.csv empty or header-only).\n")
            continue
        if comp.empty:
            lines.append(f"## {label}\n\nNo between-pipeline comparisons (only baseline or single pipeline).\n")
            continue

        lines.append(f"## {label}")
        lines.append("")
        lines.append("### Between-pipeline (per backbone)")
        lines.append("```")
        lines.append(comp.to_string())
        lines.append("```")
        lines.append("")

        if var_csv.exists():
            var_df = pd.read_csv(var_csv)
            var_sub = var_df[var_df["dataset"] == label] if "dataset" in var_df.columns else var_df
            lines.append("### Subject-level variability")
            lines.append("```")
            lines.append(var_sub.to_string())
            lines.append("```")
            lines.append("")
            summaries.append((label, comp, var_sub))
        else:
            summaries.append((label, comp, None))

    lines.append("---")
    lines.append("## Consistency interpretation")
    lines.append("")
    lines.append("- Compare **mean difference** and **Cohen's d** for GEDAI−baseline and ICALabel−baseline across the two datasets.")
    lines.append("- Compare **% improved** and **median Δ** across datasets.")
    lines.append("- Effects may differ between datasets (dataset-dependent); that is a strength of the design.")
    lines.append("- Do not average across datasets; interpret qualitatively.")
    lines.append("")
    lines.append("### Expected framing (professor's philosophy)")
    lines.append("")
    lines.append("- Denoising may show statistically significant improvements in *some* subjects; effects are heterogeneous.")
    lines.append("- Improvements are backbone-dependent (CSP vs tangent); one backbone may benefit more.")
    lines.append("- Effects may differ across datasets (e.g. EEGBCI vs BNCI2014_001); either conclusion is acceptable if statistically supported.")
    lines.append("- GEDAI/ICALabel should preserve physiologically meaningful oscillations (alpha, mu, beta); interpret cautiously if not.")
    lines.append("")

    if len(summaries) >= 2:
        lines.append("### Summary")
        lines.append("")
        for label, comp, var in summaries:
            lines.append(f"- **{label}**: Check pipeline_comparisons and variability_summary above.")
        lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
