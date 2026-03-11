from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .evaluation.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run systematic EEG denoising benchmark (CSP and Tangent Space)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        metavar="N",
        help="Override number of subjects: use subjects 1..N (e.g. --n-subjects 10 for first 10).",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.n_subjects is not None:
        n = int(args.n_subjects)
        if n < 1:
            raise SystemExit("--n-subjects must be >= 1")
        cfg.subjects = list(range(1, n + 1))
    cfg.results_root.mkdir(parents=True, exist_ok=True)

    run_experiment(cfg)


if __name__ == "__main__":
    main()

