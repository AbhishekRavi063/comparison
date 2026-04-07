#!/bin/bash
# Run full test on 10 subjects and generate overlay/PSD plots for all 10 at the end.
# Use this to check whether ICALabel (and GEDAI) produce non-identical results vs baseline.
#
# Usage: from project root
#   bash scripts/run_10subjects_with_overlays.sh
#
# Results: results/physionet_5subjects_quick/
#   tables/subject_level_performance.csv
#   figures/performance_*.png, variability_*.png
#   figures/signal_integrity/  (prepost_overlay_subj*_trial0_icalabel.png, time_*, psd_*, etc.)

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MPLBACKEND=Agg

CONFIG="config/config_alljoined_full_win.yml"
N=10

echo "=============================================="
echo "Full test: $N subjects, overlays for all $N"
echo "=============================================="

python -m src.run_full_test "$CONFIG" --n-subjects "$N" --n-signal-integrity-subjects "$N"

echo ""
echo "Done. Overlays/PSD: results/physionet_5subjects_quick/figures/signal_integrity/"
