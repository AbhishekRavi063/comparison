# Professional Setup Guide: 24-Core Workstation Migration

This guide will help you set up the project on the Windows machine (CIBM_2) for maximum performance.

## 1. Prerequisites (In Windows)
*   **Python 3.10+**: Ensure Python is installed and "Add to PATH" was checked.
*   **Git for Windows**: Download from gitforwindows.org.

## 2. Setting Up the Project
Open **PowerShell** and run these commands:

```powershell
# 1. Clone & Switch to Optimized Branch
git clone https://github.com/AbhishekRavi063/comparison.git
cd comparison
git fetch origin
git checkout fix-windows-memory

# 2. Setup Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies & Optimized Engine
pip install numpy mne scikit-learn pandas joblib tqdm pywavelets h5py matplotlib click scipy psutil asrpy requests huggingface_hub pyarrow mne-icalabel torch
pip install -e .\gedai_official
```

## 3. High-Performance Settings & Stability (Crucial)

Windows often "throttles" the CPU or crashes when RAM hits 16GB. To make the machine stable:

### A. Expand Virtual Memory (Faux-RAM)
This prevents `ArrayMemoryError` by using the SSD as "Emergency RAM".
1.  Open the **Start Menu**, type `performance`, and select **"Adjust the appearance and performance of Windows"**.
2.  Go to the **Advanced** tab.
3.  Under **Virtual memory**, click **Change...**
4.  **Uncheck** "Automatically manage paging file size for all drives".
5.  Select your **C:** drive.
6.  Select **"Custom size"**:
    *   **Initial size (MB)**: `32768` (32 GB)
    *   **Maximum size (MB)**: `65536` (64 GB)
7.  Click **Set**, then **OK**, then **Apply**.
8.  **Restart the computer** for changes to take effect.

### C. Subject Partitioning (Long-Run Stability)
For the 20-subject "Gold Standard" run, it is safer to run in batches of 5. This clears the memory cache between each block.

1.  **Batch 1 (Subjects 1-5)**:
    ```powershell
    python -m src.run_all --config config/config_alljoined_preprint_full.yml --subjects 1 2 3 4 5
    ```
2.  **Batch 2 (Subjects 6-10)**:
    ```powershell
    python -m src.run_all --config config/config_alljoined_preprint_full.yml --subjects 6 7 8 9 10
    ```
3.  **Batch 3 (Subjects 11-15)**:
    ```powershell
    python -m src.run_all --config config/config_alljoined_preprint_full.yml --subjects 11 12 13 14 15
    ```
4.  **Batch 4 (Subjects 16-20)**:
    ```powershell
    python -m src.run_all --config config/config_alljoined_preprint_full.yml --subjects 16 17 18 19 20
    ```

## 4. Running the Benchmark
Use the two "Pre-print" configurations we prepared:

**Step A: Download Data (Do this first)**

Full benchmark (every EDF per subject — large download, use parallel subjects sparingly for RAM/HF limits):

```powershell
python -m src.data.prepare_alljoined --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --all-edfs --workers 3 --out-root data/alljoined/processed
```

Smoke / quick prep (caps EDFs per subject):

```powershell
python -m src.data.prepare_alljoined --subjects 1 2 3 4 5 6 7 8 9 10 --max-edfs 5
```

**Step B: End-to-End Validation (Smoke Test — 1 subject, baseline vs GEDAI; ICALabel off for speed)**

Recommended (prepares subject 1 from Hugging Face, then runs benchmark):

```powershell
.\scripts\smoke_alljoined_1sub.ps1
```

Optional: `$env:MAX_EDFS = "5"` before running for more EDFs per subject.

Or run only the benchmark if `data\alljoined\processed\subject_1.npz` already exists:

```powershell
$env:MPLBACKEND = "Agg"
python -m src.run_all --config config/config_alljoined_smoke_1sub.yml
```

**Cho2017 (MOABB) 1-subject smoke** — checks download + pipeline without Hugging Face:

```powershell
python -m pip install moabb
.\scripts\smoke_cho2017_1sub.ps1
```

Legacy 2-subject quick check (baseline + GEDAI only, no tangent):

```powershell
python -m src.run_all --config config/config_alljoined_professor_smoke.yml
```

**Step C: Full 20-subject workstation run (parallel shards + merge)**

Use `config/config_alljoined_workstation.yml` (sets `memory.n_jobs` for GEDAI threading). Run four processes with disjoint subjects, then merge CSVs:

```powershell
$env:MPLBACKEND = "Agg"
.\scripts\run_alljoined_shards.ps1 -Config config/config_alljoined_workstation.yml -NumShards 4
python -m src.merge_sharded_results --shards results/alljoined_w1 results/alljoined_w2 results/alljoined_w3 results/alljoined_w4 --out results/alljoined_merged --n-pipeline-perm 10000 --pipelines baseline,icalabel,gedai
```

Single-process (simpler, slower): `python -m src.run_all --config config/config_alljoined_workstation.yml`

**Legacy 10-subject config:** `python -m src.run_all --config config/config_alljoined_full_win.yml`

---
GEDAI runs on CPU unless you configure PyTorch/CUDA; `memory.n_jobs` controls GEDAI’s internal thread count (lower it if you run many shard processes in parallel).
