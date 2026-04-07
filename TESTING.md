# Testing

There are **two levels** of testing:

1. **Pytest (synthetic data)** – fast, no download; checks code paths and outputs. **Does not use real EEG or full subject counts.**
2. **Full-subject testing (real datasets)** – run the full pipeline on **PhysioNet EEGBCI** and/or **BNCI 2014-001** with real subjects. This is required to validate the full system end-to-end.

---

## 1. Pytest: synthetic data only

Current **pytest** tests use **synthetic data only** (generated in a temp dir): 40 trials × 8 channels × 250 time points per “subject”. They do **not** use PhysioNet or BNCI or full subject lists.

### Test cases

| Case | What it does | Depends on |
|------|----------------|------------|
| **1** | Full experiment: **2 subjects**, **baseline only** (CSP + tangent). Checks tables, stats, and **model saving** (4 `.joblib` files). | None |
| **2** | Full experiment: **1 subject**, **baseline + ICALabel**. Checks that ICALabel pipeline runs and models are saved. | MNE, mne-icalabel |
| **3** | After experiment: runs **performance + variability plots**. Checks `performance_csp.png`, `performance_tangent.png` exist. | Matplotlib (Agg backend) |
| **4** | **Signal integrity**: time overlay and PSD figures for one trial/channel. | Matplotlib (Agg) |
| **5** | **Model load**: loads saved CSP and tangent joblibs and checks keys + float32 dtypes. | joblib |
| **6** | **Dataset loader**: loads synthetic `subject_1.npz` and checks shape, dtype float32, channel names. | None |
| **synthetic** | Minimal smoke test (existing): 1 subject, baseline only, in a temp dir. | None |

## Run all tests

From the **project root**:

```bash
source .venv/bin/activate
export MPLBACKEND=Agg
pytest tests/ -v
```

- Use `MPLBACKEND=Agg` so matplotlib does not try to open a GUI (avoids crashes in headless/CI).
- `tests/conftest.py` also sets `MPLBACKEND=Agg` for pytest; the env var ensures it’s set before any import.

## Run a subset

```bash
# Only fast baseline-only tests (no MNE/ICALabel)
pytest tests/test_integration.py -v -k "case_1 or case_3 or case_4 or case_5 or case_6"

# Only the original synthetic smoke test
pytest tests/test_synthetic.py -v

# Skip the ICALabel test (case_2)
pytest tests/test_integration.py -v -k "not case_2"
```

## Test config

- **`config/config_alljoined_test.yml`** is used by integration tests (overridden `data_root` and `results_root` point to a temp dir).
- It uses **2 subjects**, **small permutations** (`n_subject_level: 15`, `n_pipeline_level: 50`) so tests finish quickly.
- Denoising: **baseline** always; **ICALabel** only in case 2; **GEDAI** off in tests.

---

## 2. Full-subject testing (real data)

The checked-in YAML files under `config/config_alljoined_*.yml` target **Alljoined-1.6M**
(`data/alljoined/processed`). Prepare data with `python -m src.data.prepare_alljoined …`,
then run e.g. `python -m src.run_all --config config/config_alljoined_workstation.yml`.

To benchmark **other** `.npz` datasets, copy any `config_alljoined_*.yml`, set `data_root` and
`subjects` to your prepared folder, and keep the same schema.

### Summary

- **Pytest** = synthetic data only; fast; no real dataset.
- **Full runs** = use an Alljoined config (or a copy with your `data_root`).

---

## 3. Full test (professor’s structure): do not mix datasets

The evaluation has **three levels**:

1. **Within-dataset analysis** — run **separately** for each dataset (no mixing).
2. **Between-pipeline testing** — per dataset, per backbone (already in `pipeline_comparisons.csv`).
3. **Cross-dataset consistency** — qualitative interpretation only; **no pooling** of subjects across datasets.

### Within-dataset (each dataset run independently)

For **each** dataset (e.g. PhysioNet, then BNCI):

- **TEST BLOCK 1 — Above-chance validation:** Per subject, per pipeline: empirical p-value (≥1000 label permutations). Summary: `stats/above_chance_summary.csv` (n and % subjects with p &lt; 0.05).
- **TEST BLOCK 2 — Denoising effect:** GEDAI vs baseline, ICALabel vs baseline (paired permutation, ≥10k resamples, Cohen’s d). In `stats/pipeline_comparisons.csv`.
- **TEST BLOCK 3 — Subject-level variability:** Δ(GEDAI−baseline), Δ(ICALabel−baseline); % improved, % worsened, median Δ, SD. In `stats/subject_level_deltas.csv` and `stats/variability_summary.csv`.
- **TEST BLOCK 4 — Backbone interaction:** Does denoising help CSP more than Tangent (or vice versa)? In `stats/backbone_interaction.csv`.
- **TEST BLOCK 5 — Physiological integrity:** Pre–post overlay and PSD (e.g. C3/C4, alpha/beta). In `figures/signal_integrity/`.

### `run_full_test` with multiple configs

Pass one or more YAML paths (each with its own `data_root` / `results_root`), for example:

```bash
python -m src.run_full_test \
  config/config_alljoined_workstation.yml \
  --out-report results/alljoined_report.md
```

Options:

- `--skip-experiment` — reuse existing results; only recompute variability, backbone interaction, and cross-dataset report.
- `--no-signal-integrity` — skip signal integrity plots.
- `--out-report PATH` — path for the cross-dataset report.

### Expected outcome (professor’s philosophy)

- **Do not** claim “GEDAI is best overall.”
- **Do** report: denoising gives statistically significant improvements in **some** subjects; effects are **heterogeneous** and **backbone-dependent**; effects **may differ across datasets** (dataset-dependent); GEDAI/ICALabel should preserve physiologically meaningful oscillations. Either “advanced denoising helps” or “bandpass is enough” is acceptable if **statistically supported**.
