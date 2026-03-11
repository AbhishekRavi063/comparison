#!/bin/bash
# First pass: full dataset (109 subjects), baseline + ICALabel only.
# - Data: streamed one subject at a time (no full load).
# - At the end: overlay plot, PSD, performance/variability figures, tables, stats.
# - Then: brain signal removal report (% over-removal, which frequency, details).
#
# Usage: from project root:
#   bash scripts/run_first_pass_full.sh
# Optional: set N_SIG=5 to generate overlay/PSD for first 5 subjects (default 1).
# Optional: set N_REPORT=20 to check brain signal on first 20 subjects (default 10).

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MPLBACKEND=Agg

# Overlay/PSD: 1 = subject 1 only, 5 = first 5 subjects, etc.
N_SIG="${N_SIG:-1}"
# Brain signal report: how many subjects to check (10 = quick, 109 = full)
N_REPORT="${N_REPORT:-10}"

CONFIG="config/config_real_physionet_full.yml"
RESULTS_DIR="results/physionet_full"

echo "=============================================="
echo "First pass: baseline + ICALabel (full dataset)"
echo "Data: streamed per subject (no full load)"
echo "Overlay/PSD: first $N_SIG subject(s)"
echo "=============================================="

if [ "$N_SIG" = "1" ]; then
  python -m src.run_full_test "$CONFIG" --n-signal-integrity-subjects 1
else
  python -m src.run_full_test "$CONFIG" --n-signal-integrity-subjects "$N_SIG"
fi

echo ""
echo "=============================================="
echo "Brain signal removal check (first $N_REPORT subjects)"
echo "=============================================="
python scripts/brain_signal_removal_report.py \
  --config "$CONFIG" \
  --n-subjects "$N_REPORT" \
  --channel C3 \
  --out "$RESULTS_DIR/brain_signal_removal_report.md" \
  --verbose

echo ""
echo "Done. Results: $RESULTS_DIR/"
echo "  tables/subject_level_performance.csv"
echo "  stats/pipeline_comparisons.csv, variability_summary.csv"
echo "  figures/performance_*.png, variability_*.png"
echo "  figures/signal_integrity/ (overlay, PSD)"
echo "  brain_signal_removal_report.md"
