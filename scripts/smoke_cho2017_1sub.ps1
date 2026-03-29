# Download Cho2017 subject 1 via MOABB, then run a fast smoke benchmark (baseline vs GEDAI).
# Usage: from repo root,  .\scripts\smoke_cho2017_1sub.ps1
# Requires: moabb (pip install moabb), mne, gedai (pip install -e .\gedai_official), and other deps.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$root = (Get-Location).Path
if (-not $env:MNE_DATA) {
    $env:MNE_DATA = Join-Path $root ".mne_home\MNE-data"
}
New-Item -ItemType Directory -Force -Path $env:MNE_DATA | Out-Null

Write-Host "==> Preparing Cho2017 subject 1 (MOABB download + .npz)..."
python -m src.data.prepare_cho2017 --subjects 1 --out-root data/cho2017/processed

if (-not $env:MPLBACKEND) { $env:MPLBACKEND = "Agg" }
Write-Host "==> Running Cho2017 smoke (config/config_cho2017_smoke_1sub.yml)..."
python -m src.run_all --config config/config_cho2017_smoke_1sub.yml

Write-Host "==> Done. Results: results/cho2017_smoke_1sub/"
