# Commands to run the full evaluation

Use **live log** (no `tee`/`tail`) so you see progress bars and timestamps. Run from project root.

---

## Cross-subject benchmark (new thesis protocol)

This repo now includes a dedicated cross-subject runner implementing GroupKFold by subject ID with streaming fold loading:

```bash
cd /Users/abhishekr/Documents/EEG/comparison
source .venv/bin/activate
export HOME="$PWD/.mne_home"
export MPLBACKEND=Agg
export PYGEDAI_FORCE_CPU=1
export PYTHONPATH=.

python run_evaluation.py \
  --config config/config_alljoined_workstation.yml \
  --stream-subjects \
  --pipelines A,B,C,D
```

Outputs under `results/cross_subject_full/`:
- `fold_metrics.csv`
- `subject_metrics.csv`
- `stats_summary.json`
- `groupkfold_splits.json`

---

## Before you start (streaming + 16 GB memory)

- **Data is streamed:** One subject is loaded at a time via `iter_subjects()`; the full dataset is **not** loaded into memory. See `src/io/dataset.py` and `MEMORY_AUDIT.md`.
- **Memory handling:** After each subject we `del X, y, subj_data` and `gc.collect()`. Models and pipeline results are freed after save. float32 and `n_jobs=1` throughout. Optimized for **16 GB MacBook RAM**; peak per subject is on the order of a few hundred MB.
- **Time optimization now enabled in code:** preprocessing is cached per subject/pipeline, CV splits are reused for null permutations, tangent-space fold features are reused across permutations, and subject-level permutations support adaptive step-up (1000 -> 10000 only for borderline p-values by default).
- **Fast research workflow:** run benchmark with `memory.save_models: false` first, then export models only with `python -m src.run_export_models --config <config.yml>`.

---

## 1. Prepare PhysioNet data (5 subjects)

```bash
cd /Users/abhishekr/Documents/EEG/comparison
source .venv/bin/activate
export HOME="$PWD/.mne_home"
export MPLBACKEND=Agg

python -m src.data.prepare_physionet_eegbci --subjects 1 2 3 4 5 --out-root data/physionet_eegbci
```

---

## 2. Full test — quick (5 subjects, reduced permutations, ~15–30 min)

```bash
cd /Users/abhishekr/Documents/EEG/comparison
source .venv/bin/activate
export HOME="$PWD/.mne_home"
export MPLBACKEND=Agg

python -m src.run_full_test config/config_alljoined_professor_smoke.yml --out-report results/alljoined_quick_report.md
```

---

**Two-step to reduce memory:** The real-data configs have `use_gedai: false` by default. Run once to get baseline + ICALabel; then set `use_gedai: true` and re-run (or use a GEDAI-only script) to add GEDAI results without holding all three pipelines in one run.

## 3. Full test — professor-style (5 subjects, 1000 + 10000 permutations, ~2–5 h)

```bash
cd /Users/abhishekr/Documents/EEG/comparison
source .venv/bin/activate
export HOME="$PWD/.mne_home"
export MPLBACKEND=Agg

python -m src.run_full_test config/config_alljoined_preprint_full.yml --out-report results/alljoined_preprint_report.md
```

---

## 4. Overnight full dataset (109 subjects) — baseline + ICALabel only

Run this when you want to test **baseline and ICALabel** on the full 109 subjects and save models, results, and overlay plots. Do **not** use `--no-signal-integrity` so overlay/PSD are generated.

**Step 1 — Prepare data (once):**
```bash
cd /Users/abhishekr/Documents/EEG/comparison
source .venv/bin/activate
export HOME="$PWD/.mne_home"
export MPLBACKEND=Agg

python -m src.data.prepare_physionet_eegbci --subjects $(seq 1 109) --out-root data/physionet_eegbci
```

**Step 2 — Run full evaluation (overnight):**
```bash
cd /Users/abhishekr/Documents/EEG/comparison && source .venv/bin/activate && export HOME="$PWD/.mne_home" && export MPLBACKEND=Agg && python -m src.run_full_test config/config_alljoined_workstation.yml --out-report results/alljoined_workstation_report.md
```

**What gets saved (under `results/physionet_full/`):**
- **Tables:** `tables/subject_level_performance.csv` (accuracy per subject/backbone/pipeline)
- **Stats:** `stats/pipeline_comparisons.csv`, `stats/variability_summary.csv`, `stats/above_chance_summary.csv`, `stats/backbone_interaction.csv`
- **Models:** `models/subject_<id>_<backbone>_<pipeline>.joblib` (one per subject × backbone × pipeline; baseline + icalabel only)
- **Figures:** `figures/performance_*.png`, `figures/variability_*.png`, `figures/signal_integrity/` (overlay + PSD; default **one subject**; use `--signal-integrity-all-subjects` for all 109)

GEDAI is disabled in this config; add it in a second run later if needed.

**Where results are stored (all under `results/physionet_full/`):**

| What | Path |
|------|------|
| Per-subject accuracy (all subjects) | `tables/subject_level_performance.csv` |
| Pipeline comparisons (p-values, Cohen's d) | `stats/pipeline_comparisons.csv` |
| Variability (Δ, % improved) | `stats/variability_summary.csv`, `stats/subject_level_deltas.csv` |
| Above-chance summary | `stats/above_chance_summary.csv` |
| Backbone interaction | `stats/backbone_interaction.csv` |
| Saved models (one per subject/backbone/pipeline) | `models/subject_<id>_<backbone>_<pipeline>.joblib` |
| Performance boxplots (total) | `figures/performance_csp.png`, `figures/performance_tangent.png` |
| Variability boxplots (total) | `figures/variability_csp.png`, `figures/variability_tangent.png` |
| Overlay + PSD (subject 1 by default) | `figures/signal_integrity/*.png` |
| Cross-dataset report (if 2 configs) | Path from `--out-report` (e.g. `results/physionet_full_cross_dataset_report.md`) |

**Rough time to complete (109 subjects, baseline + ICALabel):**
- **Prepare (download):** ~1–3 hours (depends on network).
- **Full evaluation:** ~**12–36+ hours** (overnight to ~1.5 days). Per subject: 4 pipeline runs (2 backbones × 2 pipelines), each with 5-fold CV + 1000 null permutations; ICALabel and permutation tests dominate. Plan for overnight; 24+ hours is normal.

---

## 5. What the plots show: per subject vs total

| Plot | Scope | Description |
|------|--------|-------------|
| **Performance** (`performance_csp.png`, `performance_tangent.png`) | **Total** | One figure per backbone; **all subjects** on one plot (boxplot + points per pipeline). |
| **Variability** (`variability_csp.png`, `variability_tangent.png`) | **Total** | One figure per backbone; distribution of Δ(denoising − baseline) **across all subjects**. |
| **Overlay / PSD** (signal integrity) | **Per subject** | One set of overlay + PSD **per subject** (or one subject by default). |

So: performance and variability are **total** (all subjects in one figure). Overlay and PSD are **per subject**; use `--signal-integrity-all-subjects` to get one set for every subject.

- **Default:** Overlay and PSD for **one** subject (subject 1).
- **All subjects:** Add `--signal-integrity-all-subjects` to generate overlay + PSD for **every** subject.
- **Skip:** Append `--no-signal-integrity` to skip overlay/PSD entirely.

---

## 6. Memory-stability test (20 subjects) before full 109

See **MEMORY_AUDIT.md**. Run 20 subjects with memory debug and watch RSS; if stable, run 109 overnight.

```bash
# Prepare 1–20
python -m src.data.prepare_physionet_eegbci --subjects $(seq 1 20) --out-root data/physionet_eegbci

# Run with RAM logging (optional: pip install psutil)
export EEG_MEMORY_DEBUG=1
python -m src.run_full_test config/config_alljoined_full_win.yml
```

## 7. Validation summary (after run)

After a run, the pipeline prints a **validation summary** (mean accuracy per pipeline, p-values, Cohen's d, % improved by GEDAI). You can also print it from existing results:

```bash
python scripts/validate_5subjects.py --results-dir results/physionet_5subjects
```

**Success criteria for 5-subject test:**
- 3 pipelines × 2 backbones computed
- Permutation tests return valid p-values (GEDAI vs Bandpass, ICALabel vs Bandpass, GEDAI vs ICALabel)
- Cohen's d computed for each comparison
- Overlay plot and PSD plot in `results/<label>/figures/signal_integrity/`
- RAM stable (use `EEG_MEMORY_DEBUG=1` and watch Activity Monitor)

## 8. Optional: save live log to file and still see output

```bash
python -m src.run_full_test config/config_alljoined_smoke_1sub.yml 2>&1 | tee results/run.log
```

You will see the same output on the terminal and in `results/run.log`.
