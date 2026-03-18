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
```powershell
python src/data/prepare_alljoined.py --subjects 1 2 3 4 5 6 7 8 9 10 --max-edfs 5
```

**Step B: End-to-End Validation (Smoke Test)**
```powershell
python -m src.run_all --config config/config_alljoined_professor_smoke.yml
```

**Step C: The Full 10-Subject Run**
```powershell
python -m src.run_all --config config/config_alljoined_preprint_full.yml
```

---
*Note: Since the GPU is broken, the script will automatically fallback to the CPU. With 24 cores, it will still be extremely fast!*
