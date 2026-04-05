# Alljoined: four-step workflow (workstation)

Configs: `config/config_alljoined_smoke_1sub_full.yml` (1-subject smoke with ICALabel), `config/config_alljoined_workstation.yml` (20-subject production: **baseline + GEDAI only**, ICALabel off).

## 1) Download one subject (all EDFs for that subject)

**PowerShell**

```powershell
python -m src.data.prepare_alljoined --subjects 1 --all-edfs --out-root data/alljoined/processed
```

**bash**

```bash
python -m src.data.prepare_alljoined --subjects 1 --all-edfs --out-root data/alljoined/processed
```

## 2) Smoke test: one subject, full pipeline (baseline + ICALabel + GEDAI, CSP + tangent)

**PowerShell**

```powershell
$env:MPLBACKEND = "Agg"
python -m src.run_all --config config/config_alljoined_smoke_1sub_full.yml
```

**bash**

```bash
export MPLBACKEND=Agg
python -m src.run_all --config config/config_alljoined_smoke_1sub_full.yml
```

## 3) Download all 20 subjects (all EDFs, parallel prep)

**PowerShell**

```powershell
python -m src.data.prepare_alljoined --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --all-edfs --workers 3 --out-root data/alljoined/processed
```

**bash**

```bash
python -m src.data.prepare_alljoined --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --all-edfs --workers 3 --out-root data/alljoined/processed
```

Tune `--workers` (1–4) if Hugging Face or RAM complains.

## 4) Full 20-subject benchmark

**Option A — parallel shards (recommended on 24-core / 64 GB)**

**PowerShell**

```powershell
$env:MPLBACKEND = "Agg"
.\scripts\run_alljoined_shards.ps1 -Config config/config_alljoined_workstation.yml -NumShards 4
python -m src.merge_sharded_results --shards results/alljoined_w1 results/alljoined_w2 results/alljoined_w3 results/alljoined_w4 --out results/alljoined_merged --n-pipeline-perm 10000 --pipelines baseline,gedai
```

**bash**

```bash
export MPLBACKEND=Agg
NUM_SHARDS=4 ./scripts/run_alljoined_shards.sh
python -m src.merge_sharded_results --shards results/alljoined_w1 results/alljoined_w2 results/alljoined_w3 results/alljoined_w4 --out results/alljoined_merged --n-pipeline-perm 10000 --pipelines baseline,gedai
```

**Option B — single process (simpler, slower)**

```powershell
$env:MPLBACKEND = "Agg"
python -m src.run_all --config config/config_alljoined_workstation.yml
```

---

Faster 1-subject sanity check (no ICALabel): `config/config_alljoined_smoke_1sub.yml`.
