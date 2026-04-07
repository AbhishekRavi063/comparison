#!/usr/bin/env bash
# Download Cho2017 subject 1 via MOABB, then run a fast smoke benchmark (baseline vs GEDAI).
# Usage: from repo root,  bash scripts/smoke_cho2017_1sub.sh
# Requires: moabb (pip install moabb), mne, gedai (pip install -e ./gedai_official), and other deps.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MNE_DATA="${MNE_DATA:-$ROOT/.mne_home/MNE-data}"
mkdir -p "$MNE_DATA"

echo "==> Preparing Cho2017 subject 1 (MOABB download + .npz)..."
python -m src.data.prepare_cho2017 --subjects 1 --out-root data/cho2017/processed

export MPLBACKEND="${MPLBACKEND:-Agg}"
echo "==> Running Alljoined 1-sub smoke (config/config_alljoined_smoke_1sub.yml)..."
python -m src.run_all --config config/config_alljoined_smoke_1sub.yml

echo "==> Done. Results: results/cho2017_smoke_1sub/"
