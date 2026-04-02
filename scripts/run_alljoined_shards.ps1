# Launch parallel benchmark shards (disjoint subjects, separate results dirs), then wait.
# Tune NumShards × memory.n_jobs so total CPU/RAM stays within the machine (e.g. 4 shards × n_jobs:4).
#
# Usage (from repo root, venv active):
#   $env:MPLBACKEND = "Agg"
#   .\scripts\run_alljoined_shards.ps1
#
param(
    [string] $Config = "config/config_alljoined_workstation.yml",
    [string] $ResultsPrefix = "results/alljoined_w",
    [int] $NumShards = 4,
    [int] $FirstSubject = 1,
    [int] $LastSubject = 20
)

$ErrorActionPreference = "Stop"
$total = $LastSubject - $FirstSubject + 1
if ($total -lt 1) { throw "Invalid subject range" }

$chunk = [int][math]::Ceiling($total / $NumShards)
$procs = @()
$idx = $FirstSubject
$shardNum = 1

while ($idx -le $LastSubject) {
    $end = [math]::Min($idx + $chunk - 1, $LastSubject)
    $subjects = @($idx..$end)
    $rr = "$ResultsPrefix$shardNum"

    $argList = @(
        "-m", "src.run_all",
        "--config", $Config,
        "--results-root", $rr,
        "--subjects"
    ) + $subjects

    Write-Host "Shard $shardNum : subjects $($subjects -join ' ') -> $rr"
    $procs += Start-Process -FilePath "python" -ArgumentList $argList -WorkingDirectory (Get-Location) -PassThru
    $idx = $end + 1
    $shardNum++
}

if ($procs.Count -eq 0) { throw "No processes started" }
Wait-Process -InputObject $procs
$shardDirs = 1..($procs.Count) | ForEach-Object { "$ResultsPrefix$_" }
Write-Host "All shards finished. Merge with:"
Write-Host "python -m src.merge_sharded_results --shards $($shardDirs -join ' ') --out results/alljoined_merged --n-pipeline-perm 10000 --pipelines baseline,icalabel,gedai"
