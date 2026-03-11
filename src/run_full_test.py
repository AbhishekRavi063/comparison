"""
Full test (professor's structure): run the complete evaluation pipeline
**separately** on each dataset, then produce cross-dataset consistency report.

- Do NOT mix datasets.
- Within each dataset: above-chance validation, denoising effect (permutation + Cohen's d),
  subject-level variability, backbone interaction, signal integrity.
- Cross-dataset: qualitative consistency interpretation (no pooling).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ExperimentConfig
from .evaluation.experiment import run_experiment
from .plots.performance import plot_performance, plot_variability
from .stats.variability import compute_variability
from .stats.backbone_interaction import compute_backbone_interaction
from .stats.cross_dataset import write_cross_dataset_report
from .stats.validation_summary import print_validation_summary


def _log(msg: str) -> None:
    """Live log to stdout (flushed)."""
    logger = logging.getLogger("run_full_test")
    logger.info(msg)


def run_one_dataset(
    cfg: ExperimentConfig,
    config_path: Path | None = None,
    run_signal_integrity_subject: int | str | list | None = 1,
    n_resamples_interaction: int = 10_000,
) -> None:
    """Run full within-dataset analysis: experiment, plots, variability, backbone interaction, optional signal integrity.
    run_signal_integrity_subject: 1 = one subject (default), "all" = every subject, [s1,s2,...] = those subjects, None = skip.
    """
    results_root = Path(cfg.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    tables_dir = results_root / "tables"
    stats_dir = results_root / "stats"
    figures_dir = results_root / "figures"

    # TEST BLOCK 1 & 2: Experiment (above-chance + between-pipeline permutation + Cohen's d)
    _log("TEST BLOCK 1 & 2 — Running experiment (CV + above-chance permutations + between-pipeline tests)...")
    run_experiment(cfg)
    _log("Experiment done. Writing tables and stats.")

    subject_csv = tables_dir / "subject_level_performance.csv"
    if not subject_csv.exists():
        raise FileNotFoundError(f"Expected {subject_csv} after run_experiment.")

    # Plots (performance + variability)
    _log("Plotting performance and variability...")
    plot_performance(subject_csv, figures_dir)
    plot_variability(subject_csv, figures_dir)

    # TEST BLOCK 3: Subject-level variability (Δ, % improved/worsened, median, SD)
    dataset_label = getattr(cfg, "dataset_label", None) or results_root.name
    _log("TEST BLOCK 3 — Computing subject-level variability...")
    compute_variability(subject_csv, stats_dir, dataset_label=dataset_label)

    # TEST BLOCK 4: Backbone interaction
    _log("TEST BLOCK 4 — Computing backbone interaction...")
    compute_backbone_interaction(
        subject_csv,
        stats_dir,
        dataset_label=dataset_label,
        n_resamples=n_resamples_interaction,
    )

    # TEST BLOCK 5: Physiological integrity (overlay + PSD). One subject, list of subjects, or all.
    if run_signal_integrity_subject is not None and config_path is not None:
        try:
            import sys
            from .run_signal_integrity import main as run_signal_integrity_main
            old_argv = list(sys.argv)
            if run_signal_integrity_subject == "all":
                subject_ids = cfg.subjects
                _log(f"TEST BLOCK 5 — Running signal integrity for ALL subjects ({len(subject_ids)})...")
            elif isinstance(run_signal_integrity_subject, list):
                subject_ids = run_signal_integrity_subject
                _log(f"TEST BLOCK 5 — Running signal integrity for {len(subject_ids)} subjects...")
            else:
                subject_ids = [run_signal_integrity_subject]
                _log("TEST BLOCK 5 — Running signal integrity (one subject)...")
            for sid in subject_ids:
                _log(f"  Signal integrity subject {sid}")
                sys.argv = [
                    "run_signal_integrity",
                    "--config", str(config_path),
                    "--subject", str(sid),
                    "--trial", "0",
                ]
                try:
                    run_signal_integrity_main()
                except Exception as e:
                    import warnings
                    warnings.warn(f"Signal integrity failed for subject {sid}: {e}")
            sys.argv = old_argv
        except Exception as e:
            import warnings
            warnings.warn(f"Signal integrity skipped: {e}")
    else:
        _log("TEST BLOCK 5 — Skipped (--no-signal-integrity or no config).")

    # Validation summary: mean acc per pipeline, p-values, Cohen's d, % improved by GEDAI
    print_validation_summary(results_root, dataset_label=dataset_label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full test: each dataset independently, then cross-dataset consistency report.",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Paths to config YAML files (one per dataset), e.g. config/config_full_subject_physionet_example.yml config/config_full_subject_bnci_example.yml",
    )
    parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Skip running experiment (use existing results); only recompute variability, interaction, and cross-dataset report.",
    )
    parser.add_argument(
        "--no-signal-integrity",
        action="store_true",
        help="Do not run signal integrity (no overlay/PSD plots).",
    )
    parser.add_argument(
        "--signal-integrity-all-subjects",
        action="store_true",
        help="Run overlay and PSD plots for every subject (default: one subject only).",
    )
    parser.add_argument(
        "--n-signal-integrity-subjects",
        type=int,
        default=None,
        metavar="N",
        help="Run overlay/PSD for first N subjects only (e.g. 10). Overrides default (1) unless --signal-integrity-all-subjects is set.",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="results/cross_dataset_consistency_report.md",
        help="Path for the cross-dataset consistency report.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        metavar="N",
        help="Override number of subjects: use subjects 1..N (e.g. --n-subjects 10 for first 10).",
    )
    args = parser.parse_args()

    # Live logging to stdout
    log = logging.getLogger("run_full_test")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        h.terminator = "\n"
        log.addHandler(h)
        # Force unbuffered stdout so live log is visible when piping
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass
    log.info("Full test started (live log).")

    config_paths = [Path(p) for p in args.configs]
    if not config_paths:
        raise SystemExit("Provide at least one config path.")

    dataset_results: list[tuple[str, Path]] = []

    for cpath in config_paths:
        if not cpath.exists():
            raise FileNotFoundError(f"Config not found: {cpath}")
        cfg = ExperimentConfig.from_yaml(cpath)
        if args.n_subjects is not None:
            n = int(args.n_subjects)
            if n < 1:
                raise SystemExit("--n-subjects must be >= 1")
            cfg.subjects = list(range(1, n + 1))
        label = getattr(cfg, "dataset_label", None) or Path(cfg.results_root).name
        log.info(f"Dataset: {label} | subjects={len(cfg.subjects)} | results_root={cfg.results_root}")
        print(f"\n{'='*60}\nDataset: {label}\n{'='*60}\n", flush=True)

        if not args.skip_experiment:
            if args.no_signal_integrity:
                sig_subj = None
            elif args.signal_integrity_all_subjects:
                sig_subj = "all"
            elif args.n_signal_integrity_subjects is not None:
                n_sig = int(args.n_signal_integrity_subjects)
                sig_subj = list(cfg.subjects)[: max(1, n_sig)]  # first N subject IDs
            else:
                sig_subj = 1
            run_one_dataset(
                cfg,
                config_path=cpath,
                run_signal_integrity_subject=sig_subj,
            )
        else:
            # Still run variability and backbone interaction from existing CSVs
            subject_csv = Path(cfg.results_root) / "tables" / "subject_level_performance.csv"
            stats_dir = Path(cfg.results_root) / "stats"
            if subject_csv.exists():
                compute_variability(subject_csv, stats_dir, dataset_label=label)
                compute_backbone_interaction(subject_csv, stats_dir, dataset_label=label)

        dataset_results.append((label, Path(cfg.results_root)))

    # Cross-dataset consistency (no pooling)
    log = logging.getLogger("run_full_test")
    if len(dataset_results) >= 2:
        out_report = Path(args.out_report)
        log.info("Writing cross-dataset consistency report...")
        write_cross_dataset_report(dataset_results, out_report)
        print(f"\nCross-dataset report written to: {out_report}\n", flush=True)
        log.info("Full test finished.")
    else:
        print("\nSingle dataset run; no cross-dataset report generated.\n", flush=True)
        log.info("Full test finished.")


if __name__ == "__main__":
    main()
