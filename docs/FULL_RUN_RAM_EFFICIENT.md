# Full-Dataset (109 Subjects) Run: RAM-Efficient Two-Pass Workflow

**Data loading:** The pipeline **streams** one subject at a time (`iter_subjects()`); it does **not** load all 109 subjects into memory. Each subject is loaded from disk, processed (baseline + ICALabel or GEDAI), results written, then freed before the next subject.

On a **16 GB RAM** machine, running all three pipelines (baseline + ICALabel + GEDAI) in a **single** full run can push memory because:

- **ICALabel** builds ICA and large MNE Raw/epoch temporaries.
- **GEDAI** also uses substantial memory (wavelet, leadfield, etc.).
- The code does `gc.collect()` before GEDAI per subject, but peak RAM with both in the same run can still get high.

**Recommended approach:** run in **two passes**, then **merge** results. This keeps peak RAM lower and is the intended way for full 109-subject runs.

---

## Two-pass workflow (efficient for 16 GB RAM)

### Pass 1: Baseline + ICALabel only

Uses config that has **GEDAI off** (baseline + ICALabel only). Results go to a dedicated directory.

```bash
cd /Users/abhishekr/Documents/EEG/comparison
source .venv/bin/activate
export MPLBACKEND=Agg

# Full 109 subjects, baseline + ICALabel only (no GEDAI)
python -m src.run_full_test config/config_alljoined_workstation.yml
```

- **Config:** `config/config_alljoined_workstation.yml` (already has `use_gedai: false`).
- **Output:** `results/physionet_full/` (tables, stats, figures, models for **baseline** and **icalabel** only).

Optional: skip overlay/PSD to save time:

```bash
python -m src.run_full_test config/config_alljoined_workstation.yml --no-signal-integrity
```

---

### Pass 2: GEDAI only

Uses the **GEDAI-only** config. Same 109 subjects, same CV (same `random_state`), so results are comparable.

```bash
# GEDAI-only pass (same subjects, same CV splits)
python -m src.run_full_test config/config_alljoined_workstation.yml --no-signal-integrity
```

- **Config:** `config/config_alljoined_workstation.yml` (`use_baseline: false`, `use_icalabel: false`, `use_gedai: true`).
- **Output:** `results/physionet_full_gedai_only/` (tables with **gedai** only).

---

### Pass 3: Merge and compare

Merge the two result directories into one. The script combines the subject-level CSVs, recomputes pipeline comparisons (baseline vs ICALabel vs GEDAI), variability, backbone interaction, and regenerates performance/variability figures.

```bash
python scripts/merge_pipeline_runs.py \
  --base-results results/physionet_full \
  --gedai-results results/physionet_full_gedai_only \
  --out-results results/physionet_full_merged \
  --pipeline-permutations 10000 \
  --dataset-label physionet_eegbci_full
```

- **Merged output:** `results/physionet_full_merged/`
  - `tables/subject_level_performance.csv` — all subjects, all three pipelines (baseline, icalabel, gedai).
  - `stats/pipeline_comparisons.csv` — gedai vs baseline, icalabel vs baseline, gedai vs icalabel.
  - `stats/variability_summary.csv`, `stats/backbone_interaction.csv`.
  - `figures/performance_*.png`, `figures/variability_*.png`.
  - `models/` — models from both passes (if saved).

---

## Summary

| Run        | Pipelines           | Config                              | Results dir                      |
|-----------|---------------------|-------------------------------------|----------------------------------|
| Pass 1    | baseline + ICALabel (heavy) | `config_alljoined_smoke_1sub_full.yml` or a copy with `use_gedai: false` | whatever `results_root` you set |
| Pass 2    | baseline + GEDAI      | `config_alljoined_workstation.yml` (ICALabel off) | `./results/alljoined_workstation` (default in YAML) |
| Merge     | —                   | `scripts/merge_pipeline_runs.py`    | your chosen merged output dir   |

**Why this is better for RAM:** Pass 1 never loads GEDAI; Pass 2 only runs GEDAI (and no ICA). Peak memory per run stays lower than running all three pipelines in one process.

**Single-run option:** You can still run all three in one go (e.g. with a config that has baseline + icalabel + gedai all true). It may work on 16 GB due to per-subject processing and `gc.collect()`, but if you see OOM or swapping, use the two-pass + merge workflow above.

---

## First pass with overlay, PSD, and brain-signal report

To run **Pass 1** (baseline + ICALabel) with:
- automatic overlay and PSD plots (e.g. first 1 or 5 subjects),
- all result scores (tables, stats, performance/variability figures),
- then a **brain signal removal report** (% over-removal, which frequency band, details),

use either the script or the commands below.

### Option A: One script (recommended)

```bash
cd /Users/abhishekr/Documents/EEG/comparison
bash scripts/run_first_pass_full.sh
```

- Overlay/PSD for **subject 1 only** (set `N_SIG=5` before running to get first 5 subjects).
- After the run, `scripts/brain_signal_removal_report.py` runs and writes **`results/alljoined_workstation/brain_signal_removal_report.md`** (or under the `results_root` in your config) with:
  - % subjects with alpha (8–12 Hz) over-removal
  - % subjects with beta (13–30 Hz) over-removal
  - mean/min ratios per pipeline
  - which frequency was most reduced and worst subject IDs

To check more subjects in the report or more overlay subjects:

```bash
N_SIG=5 N_REPORT=20 bash scripts/run_first_pass_full.sh
```

### Option B: Commands step by step

```bash
cd /Users/abhishekr/Documents/EEG/comparison
source .venv/bin/activate
export MPLBACKEND=Agg

# 1) First pass: baseline + ICALabel, overlay/PSD for subject 1
python -m src.run_full_test config/config_alljoined_workstation.yml

# Or overlay/PSD for first 5 subjects:
# python -m src.run_full_test config/config_alljoined_workstation.yml --n-signal-integrity-subjects 5

# 2) Brain signal removal report (first 10 subjects; use 109 for full)
python scripts/brain_signal_removal_report.py \
  --config config/config_alljoined_workstation.yml \
  --n-subjects 10 \
  --channel C3 \
  --out results/alljoined_workstation/brain_signal_removal_report.md \
  --verbose
```

The report states whether brain signal was removed, what percentage of subjects (if any) showed over-removal, and which band (alpha vs beta) and which subject(s) were worst.
