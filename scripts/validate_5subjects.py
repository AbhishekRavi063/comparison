#!/usr/bin/env python3
"""
Print validation summary after a 5-subject (or any) run.

Usage:
  python scripts/validate_5subjects.py --results-dir results/physionet_5subjects

Reads:
  results_dir/tables/subject_level_performance.csv
  results_dir/stats/pipeline_comparisons.csv

Prints:
  Mean accuracy per pipeline (per backbone)
  p-values and Cohen's d for each comparison
  % subjects improved by GEDAI vs baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Add project root so we can import src
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats.validation_summary import print_validation_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print validation summary from a completed run (e.g. 5-subject test).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/physionet_5subjects",
        help="Results root (e.g. results/physionet_5subjects).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional dataset label for the summary header.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.is_dir():
        raise SystemExit(f"Results dir not found: {results_root}")

    print_validation_summary(results_root, dataset_label=args.label)


if __name__ == "__main__":
    main()
