from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ExperimentConfig
from .evaluation.experiment import run_experiment


def _configure_logging() -> None:
    """Ensure INFO-level logs (including ``mrcp_diag``) reach stdout."""
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root.addHandler(h)
    root.setLevel(logging.INFO)
    for name in ("run_full_test", "mrcp_diag"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Run systematic EEG denoising benchmark (CSP and Tangent Space)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_alljoined_smoke_1sub.yml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        metavar="N",
        help="Override: use first N subjects.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Override: use specific subject IDs (e.g. --subjects 6 7 8 9 10).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Override results_root from YAML (for parallel shard runs).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        metavar="N",
        help="Override: stratified subsample to at most N trials per subject (smoke / speed).",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    if args.max_trials is not None:
        cfg.max_trials = int(args.max_trials)

    if args.results_root is not None:
        cfg.results_root = Path(args.results_root)
    
    if args.subjects is not None:
        cfg.subjects = list(args.subjects)
    elif args.n_subjects is not None:
        n = int(args.n_subjects)
        if n < 1:
            raise SystemExit("--n-subjects must be >= 1")
        cfg.subjects = list(range(1, n + 1))
    cfg.results_root.mkdir(parents=True, exist_ok=True)

    run_experiment(cfg)


if __name__ == "__main__":
    main()

