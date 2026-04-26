from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.denoising.pipelines import MRCP_MOTOR_CHANNELS, _resolve_eeg_emg_mrcp_raw_root, _run_mrcp_raw_pipeline


def _motor_indices(ch_names: list[str]) -> np.ndarray:
    wanted = {c.upper() for c in MRCP_MOTOR_CHANNELS}
    return np.array([i for i, c in enumerate(ch_names) if c.upper() in wanted], dtype=int)


def _mean_and_sem(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    avg = X.mean(axis=0)
    sem = X.std(axis=0, ddof=1) / np.sqrt(max(len(X), 1))
    return avg, sem


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot movement-locked MRCP waveform before/after GEDAI.")
    parser.add_argument("--subject", type=int, default=2)
    parser.add_argument("--data-root", default="data/eeg_emg_mrcp/processed")
    parser.add_argument(
        "--raw-root",
        default="data/EEG and EMG Dataset for Analyzing Movement-Related",
    )
    parser.add_argument(
        "--out",
        default="results/mrcp_fix_plots/subject_02/SUBJECT02_MRCP_waveform.png",
    )
    parser.add_argument(
        "--style",
        choices=("full", "linkedin"),
        default="full",
        help="Use 'linkedin' for a cleaner public-facing MRCP figure.",
    )
    args = parser.parse_args()

    npz_path = ROOT / args.data_root / f"subject_{args.subject}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing prepared subject file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as z:
        y = z["y"].astype(int)
        sfreq = float(np.asarray(z["sfreq"]).item())
        ch_names = [str(c) for c in z["ch_names"].tolist()]

    raw_root = _resolve_eeg_emg_mrcp_raw_root(ROOT / args.raw_root)
    X_gedai, X_ref = _run_mrcp_raw_pipeline(
        subject_id=args.subject,
        raw_root=raw_root,
        l_freq=0.1,
        h_freq=1.0,
        sfreq=sfreq,
        gedai_n_jobs=1,
        run_gedai=True,
    )

    n = min(len(y), len(X_ref), len(X_gedai))
    y = y[:n]
    X_ref = X_ref[:n]
    X_gedai = X_gedai[:n]

    motor_idx = _motor_indices(ch_names)
    if motor_idx.size == 0:
        raise ValueError("Could not find MRCP motor channels in prepared subject file.")

    # Average across motor channels first, then across trials.
    ref_motor = X_ref[:, motor_idx, :].mean(axis=1)
    gedai_motor = X_gedai[:, motor_idx, :].mean(axis=1)

    move_mask = y == 1
    rest_mask = y == 0

    ref_move_mean, ref_move_sem = _mean_and_sem(ref_motor[move_mask])
    gedai_move_mean, gedai_move_sem = _mean_and_sem(gedai_motor[move_mask])
    ref_rest_mean, _ = _mean_and_sem(ref_motor[rest_mask])
    gedai_rest_mean, _ = _mean_and_sem(gedai_motor[rest_mask])

    times = np.arange(ref_move_mean.shape[0]) / sfreq - 2.0

    if args.style == "linkedin":
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.plot(times, ref_move_mean, color="#1f77b4", lw=3.2, label="Reference MRCP")
        ax.plot(times, gedai_move_mean, color="#d62728", lw=3.0, label="GEDAI MRCP")
        ax.axvline(0.0, color="0.2", lw=1.2, ls=":")
        ax.axhline(0.0, color="0.75", lw=0.9)
        ax.text(
            0.985,
            0.08,
            "Movement onset",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="0.35",
        )
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time relative to movement onset (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.set_title(f"Subject {args.subject:02d} Motor-Channel MRCP (0.1-1 Hz)")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(True, alpha=0.18)
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(times, ref_move_mean, color="#1f77b4", lw=2.2, label="Movement Reference")
        ax.fill_between(times, ref_move_mean - ref_move_sem, ref_move_mean + ref_move_sem, color="#1f77b4", alpha=0.18)
        ax.plot(times, gedai_move_mean, color="#d62728", lw=2.0, label="Movement GEDAI")
        ax.fill_between(times, gedai_move_mean - gedai_move_sem, gedai_move_mean + gedai_move_sem, color="#d62728", alpha=0.16)

        ax.plot(times, ref_rest_mean, color="#1f77b4", lw=1.2, ls="--", alpha=0.65, label="Rest Reference")
        ax.plot(times, gedai_rest_mean, color="#d62728", lw=1.2, ls="--", alpha=0.65, label="Rest GEDAI")

        ax.axvline(0.0, color="black", lw=1.0, ls=":")
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time relative to movement onset (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.set_title(f"Subject {args.subject:02d} Grand-Average MRCP on Motor Channels (0.1-1 Hz)")
        ax.legend(loc="upper left", ncol=2)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
