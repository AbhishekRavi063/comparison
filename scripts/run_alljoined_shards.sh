#!/usr/bin/env bash
# Parallel shards for Unix/macOS (same idea as run_alljoined_shards.ps1).
set -euo pipefail
CONFIG="${CONFIG:-config/config_alljoined_workstation.yml}"
PREFIX="${RESULTS_PREFIX:-results/alljoined_w}"
NUM_SHARDS="${NUM_SHARDS:-4}"
FIRST="${FIRST_SUBJECT:-1}"
LAST="${LAST_SUBJECT:-20}"

total=$((LAST - FIRST + 1))
chunk=$(( (total + NUM_SHARDS - 1) / NUM_SHARDS ))

pids=()
idx=$FIRST
s=1
while (( idx <= LAST )); do
  end=$(( idx + chunk - 1 ))
  (( end > LAST )) && end=$LAST
  rr="${PREFIX}${s}"
  subs=()
  for (( j=idx; j<=end; j++ )); do subs+=("$j"); done
  echo "Shard $s: ${subs[*]} -> $rr"
  MPLBACKEND="${MPLBACKEND:-Agg}" python -m src.run_all \
    --config "$CONFIG" --results-root "$rr" --subjects "${subs[@]}" &
  pids+=($!)
  idx=$((end + 1))
  s=$((s + 1))
done

for pid in "${pids[@]}"; do wait "$pid"; done
echo "Merge: python -m src.merge_sharded_results --shards ${PREFIX}1 ... --out results/alljoined_merged --n-pipeline-perm 10000 --pipelines baseline,gedai --pipeline-comparison-method mann_whitney"
