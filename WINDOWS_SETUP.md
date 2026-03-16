# Professional Setup Guide: 24-Core Workstation Migration

This guide will help you set up the project on the Windows machine (CIBM_2) for maximum performance.

## 1. Prerequisites (In Windows)
*   **Python 3.10+**: Ensure Python is installed and "Add to PATH" was checked.
*   **Git for Windows**: Download from gitforwindows.org.

## 2. Setting Up the Project
Open **PowerShell** and run these commands:

```powershell
# 1. Clone your code
git clone https://github.com/AbhishekRavi063/comparison.git
cd comparison

# 2. Setup the Professor's Fix
git clone https://github.com/neurotuning/gedai.git gedai_professor_fix
cd gedai_professor_fix
git checkout fix
cd ..

# 3. Create a Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

# 4. Install dependencies
pip install numpy mne scikit-learn pandas joblib tqdm pywavelets h5py matplotlib click scipy psutil asrpy
pip install -e ./gedai_professor_fix
```

## 3. High-Performance Settings
Windows often "throttles" the CPU to save power. To use all 24 cores:
1.  Search for **"Power Plan"** in the Windows Start menu.
2.  Choose **"High Performance"** or **"Ultimate Performance"**.

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
