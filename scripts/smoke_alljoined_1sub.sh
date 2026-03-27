#!/usr/bin/env bash
# Smoke test: prepare subject 1 from Hugging Face, then run full pipeline paths (see config_alljoined_smoke_1sub.yml).
# Usage: from repo root,  bash scripts/smoke_alljoined_1sub.sh
# Requires: huggingface_hub, mne, and project deps (see README / WINDOWS_SETUP.md).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_EDFS="${MAX_EDFS:-2}"

echo "==> Preparing Alljoined subject 1 (max_edfs=${MAX_EDFS})..."
python -m src.data.prepare_alljoined --subjects 1 --max-edfs "${MAX_EDFS}" --out-root data/alljoined/processed

export MPLBACKEND="${MPLBACKEND:-Agg}"
echo "==> Running benchmark smoke (config/config_alljoined_smoke_1sub.yml)..."
python -m src.run_all --config config/config_alljoined_smoke_1sub.yml

echo "==> Done. Results: results/alljoined_smoke_1sub/"
