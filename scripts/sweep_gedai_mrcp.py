"""Reproducible GEDAI MRCP noise_multiplier × prior sweep.

For each (subject, noise_multiplier, prior) combination, generates a temp config
YAML inherited from the smoke gedai_mrcp config, runs ``src.run_all`` as a
subprocess (to isolate memory + gedai state), parses ``subject_level_performance.csv``
and the run log, and writes a summary CSV.

Usage (from repo root)::

    .venv/bin/python -m scripts.sweep_gedai_mrcp \
        --subjects 1 2 12 \
        --nm 1.0 0.5 0.25 0.0 \
        --priors grand_avg_erp \
        --retention-min 0.0

Outputs:
- results/sweeps/gedai_mrcp_sweep_summary.csv
- results/sweeps/logs/<run_tag>.log
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "config" / "config_eeg_emg_mrcp_smoke_gedai_mrcp.yml"
SWEEP_ROOT = REPO_ROOT / "results" / "sweeps"
SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
(SWEEP_ROOT / "logs").mkdir(parents=True, exist_ok=True)
(SWEEP_ROOT / "configs").mkdir(parents=True, exist_ok=True)
(SWEEP_ROOT / "runs").mkdir(parents=True, exist_ok=True)


def make_config(
    subject: int,
    nm: float,
    prior: str,
    retention_min: float,
    refcov_rank_max: int | None,
    refcov_motor_weight: float | None,
    refcov_move_mix: float | None,
) -> tuple[Path, Path]:
    with BASE_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    tag = f"s{subject}_nm{nm}_{prior}_ret{retention_min}".replace(".", "p")
    results_root = SWEEP_ROOT / "runs" / tag
    cfg["subjects"] = [int(subject)]
    cfg["results_root"] = str(results_root)
    cfg.setdefault("denoising", {})
    cfg["denoising"]["use_baseline"] = True
    cfg["denoising"]["use_gedai"] = False
    cfg["denoising"]["use_gedai_mrcp"] = True
    cfg["denoising"]["gedai_mrcp_noise_multiplier"] = float(nm)
    cfg["denoising"]["gedai_mrcp_prior"] = str(prior)
    cfg["denoising"]["gedai_mrcp_retention_min"] = float(retention_min)
    cfg["denoising"]["gedai_mrcp_refcov_rank_max"] = refcov_rank_max
    cfg["denoising"]["gedai_mrcp_refcov_motor_weight"] = refcov_motor_weight
    cfg["denoising"]["gedai_mrcp_refcov_move_mix"] = refcov_move_mix
    # Keep the fold count small for sweep speed but retain 2 folds (CV-safe).
    cfg.setdefault("cv", {})["n_splits"] = 2
    cfg_path = SWEEP_ROOT / "configs" / f"{tag}.yml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path, results_root


def run_single(cfg_path: Path, log_path: Path) -> int:
    env = dict(os.environ)
    env.update(
        {
            "HOME": "/tmp",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/tmp/mplconfig",
            "MNE_DONTWRITE_HOME": "true",
            "PYGEDAI_FORCE_CPU": "1",
        }
    )
    with log_path.open("w") as logf:
        cp = subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                "-m", "src.run_all",
                "--config", str(cfg_path),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return cp.returncode


def _first_pipeline_value(df: pd.DataFrame, pipeline: str, column: str) -> float:
    rows = df[df["pipeline"] == pipeline]
    if rows.empty:
        return float("nan")
    if "backbone" in rows.columns and "tangent" in set(rows["backbone"]):
        rows = rows[rows["backbone"] == "tangent"]
    return float(rows[column].iloc[0])


def _parse_diag_tokens(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value.strip().strip(",")
    return out


def parse_results(results_root: Path, log_path: Path) -> dict:
    out = {
        "baseline_mean_accuracy": float("nan"),
        "delta_vs_baseline": float("nan"),
        "mean_accuracy": float("nan"),
        "std_accuracy": float("nan"),
        "alpha_ratio": float("nan"),
        "beta_ratio": float("nan"),
        "retention_on_motor_subset_mean": float("nan"),
        "retention_on_motor_subset_min": float("nan"),
        "retention_ratios": "",
        "fallback_rate": float("nan"),
        "n_fallback_events": 0,
        "n_gedai_session_attempts": 0,
    }
    csv_path = results_root / "tables" / "subject_level_performance.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        out["baseline_mean_accuracy"] = _first_pipeline_value(
            df, "baseline", "mean_accuracy"
        )
        out["mean_accuracy"] = _first_pipeline_value(
            df, "gedai_mrcp", "mean_accuracy"
        )
        out["std_accuracy"] = _first_pipeline_value(df, "gedai_mrcp", "std_accuracy")
        out["alpha_ratio"] = _first_pipeline_value(df, "gedai_mrcp", "alpha_ratio")
        out["beta_ratio"] = _first_pipeline_value(df, "gedai_mrcp", "beta_ratio")
        if pd.notna(out["baseline_mean_accuracy"]) and pd.notna(out["mean_accuracy"]):
            out["delta_vs_baseline"] = (
                out["mean_accuracy"] - out["baseline_mean_accuracy"]
            )

    # Parse log: use run_gedai=True post summaries so baseline diagnostic lines do
    # not inflate the denominator.
    if log_path.exists():
        text = log_path.read_text(errors="ignore")
        post_attempts = 0
        post_fallbacks = 0
        retention_vals: list[float] = []
        for line in text.splitlines():
            if "[MRCP_DIAG] stage=post" not in line:
                continue
            kv = _parse_diag_tokens(line)
            if kv.get("run_gedai") != "True":
                continue
            post_attempts += int(kv.get("n_sessions", "0"))
            post_fallbacks += int(kv.get("n_fallback_sessions", "0"))
            try:
                val = float(kv.get("retention_on_motor_subset", "nan"))
            except ValueError:
                val = float("nan")
            if pd.notna(val):
                retention_vals.append(val)
        if post_attempts > 0:
            out["n_fallback_events"] = post_fallbacks
            out["n_gedai_session_attempts"] = post_attempts
            out["fallback_rate"] = post_fallbacks / post_attempts
        else:
            n_fb = len(re.findall(r"\[MRCP_FALLBACK\]", text))
            n_pre = len(
                re.findall(
                    r"\[MRCP_DIAG\] stage=pre .*refcov_source=(mrcp_ndarray|distance_montage)",
                    text,
                )
            )
            n_fb_legacy = len(re.findall(r"using paper baseline", text))
            out["n_fallback_events"] = n_fb or n_fb_legacy
            out["n_gedai_session_attempts"] = n_pre
            if n_pre > 0:
                out["fallback_rate"] = out["n_fallback_events"] / n_pre
        if retention_vals:
            out["retention_on_motor_subset_mean"] = float(
                pd.Series(retention_vals).mean()
            )
            out["retention_on_motor_subset_min"] = float(min(retention_vals))
            out["retention_ratios"] = "|".join(f"{v:.4f}" for v in retention_vals)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 12])
    ap.add_argument("--nm", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.0])
    ap.add_argument(
        "--priors",
        type=str,
        nargs="+",
        default=["grand_avg_erp", "class_contrast", "trial_cov_mean"],
    )
    ap.add_argument("--retention-min", type=float, default=0.0)
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (subject,nm,prior) combos whose results CSV already exists.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output CSV path for the sweep summary.",
    )
    ap.add_argument(
        "--refcov-rank-max",
        type=int,
        default=None,
        help="Optional maximum rank for MRCP refcov priors such as class_contrast.",
    )
    ap.add_argument(
        "--refcov-motor-weight",
        type=float,
        default=None,
        help="Optional motor-strip weighting applied to the MRCP refcov.",
    )
    ap.add_argument(
        "--refcov-move-mix",
        type=float,
        default=None,
        help="Optional movement-covariance blend for class_contrast.",
    )
    args = ap.parse_args()

    rows = []
    for subject in args.subjects:
        for prior in args.priors:
            for nm in args.nm:
                cfg_path, results_root = make_config(
                    subject,
                    nm,
                    prior,
                    args.retention_min,
                    args.refcov_rank_max,
                    args.refcov_motor_weight,
                    args.refcov_move_mix,
                )
                log_path = SWEEP_ROOT / "logs" / (cfg_path.stem + ".log")
                csv_path = results_root / "tables" / "subject_level_performance.csv"
                if args.skip_existing and csv_path.exists() and log_path.exists():
                    print(f"[SKIP] {cfg_path.stem} (exists)")
                else:
                    print(f"[RUN] subject={subject} nm={nm} prior={prior}")
                    rc = run_single(cfg_path, log_path)
                    if rc != 0:
                        print(
                            f"  -> non-zero exit ({rc}); see {log_path}",
                            file=sys.stderr,
                        )
                res = parse_results(results_root, log_path)
                row = {
                    "subject": subject,
                    "pipeline": "gedai_mrcp",
                    "prior": prior,
                    "noise_multiplier": nm,
                    "retention_min": args.retention_min,
                    "refcov_rank_max": args.refcov_rank_max,
                    "refcov_motor_weight": args.refcov_motor_weight,
                    "refcov_move_mix": args.refcov_move_mix,
                    **res,
                }
                rows.append(row)
                print(
                    f"  -> mean_acc={res['mean_accuracy']:.4f} "
                    f"baseline={res['baseline_mean_accuracy']:.4f} "
                    f"delta={res['delta_vs_baseline']:+.4f} "
                    f"fallback={res['n_fallback_events']}/{res['n_gedai_session_attempts']} "
                    f"rate={res['fallback_rate']:.3f}"
                )

    summary = pd.DataFrame(rows)
    out_csv = Path(args.out) if args.out else (SWEEP_ROOT / "gedai_mrcp_sweep_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Pivoted table: (subject, prior) × nm → mean_accuracy, with fallback marker.
    try:
        def _fmt(row):
            acc = row["mean_accuracy"]
            delta = row["delta_vs_baseline"]
            rate = row["fallback_rate"]
            if pd.isna(acc):
                return "N/A"
            tag = f" (d={delta:+.3f})" if pd.notna(delta) else ""
            if pd.notna(rate) and rate > 0:
                tag = f"{tag} [fb={rate:.2f}]"
            return f"{acc:.3f}{tag}"

        summary["cell"] = summary.apply(_fmt, axis=1)
        pivot = summary.pivot_table(
            index=["subject", "prior"],
            columns="noise_multiplier",
            values="cell",
            aggfunc="first",
        )
        print("\n=== gedai_mrcp accuracy (fb = fallback_rate) ===")
        print(pivot.to_string())
    except Exception as e:
        print(f"pivot failed: {e}")


if __name__ == "__main__":
    main()
