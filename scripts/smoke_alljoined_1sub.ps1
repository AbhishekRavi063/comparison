# Smoke test: prepare subject 1 from Hugging Face, then run full pipeline paths (see config_alljoined_smoke_1sub.yml).
# Usage: from repo root,  .\scripts\smoke_alljoined_1sub.ps1
# Optional: $env:MAX_EDFS = "5"  for more EDFs per subject before running.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$maxEdfs = if ($env:MAX_EDFS) { $env:MAX_EDFS } else { "2" }

Write-Host "==> Preparing Alljoined subject 1 (max_edfs=$maxEdfs)..."
python -m src.data.prepare_alljoined --subjects 1 --max-edfs $maxEdfs --out-root data/alljoined/processed

if (-not $env:MPLBACKEND) { $env:MPLBACKEND = "Agg" }
Write-Host "==> Running benchmark smoke (config/config_alljoined_smoke_1sub.yml)..."
python -m src.run_all --config config/config_alljoined_smoke_1sub.yml

Write-Host "==> Done. Results: results/alljoined_smoke_1sub/"
