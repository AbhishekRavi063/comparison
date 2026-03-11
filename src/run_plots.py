from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .plots.performance import plot_performance, plot_variability


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate performance plots from experiment CSV outputs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    subject_csv = cfg.results_root / "tables" / "subject_level_performance.csv"
    figures_dir = cfg.results_root / "figures"

    if not subject_csv.exists():
        raise FileNotFoundError(
            f"Subject-level performance file not found: {subject_csv}. "
            "Run src.run_all first."
        )

    plot_performance(subject_csv, figures_dir)
    plot_variability(subject_csv, figures_dir)


if __name__ == "__main__":
    main()

