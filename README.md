## Systematic Benchmarking of EEG Denoising Strategies for Motor Imagery

This repository implements a **research-grade**, memory-constrained benchmark
of EEG denoising strategies for motor imagery decoding using two validated
backbones:

- **CSP → LDA**
- **Covariance Tangent Space (Riemannian) → Logistic Regression**

The design follows:

- *Rethinking Generalized BCIs: Benchmarking 340,000+ Unique Algorithmic Configurations for EEG Mental Command Decoding*  
  (backbone choices, within-subject CV, subject-level variability)
- Combrisson & Jerbi (2015), *Exceeding Chance Level by Chance*  
  (empirical permutation tests for above-chance decoding)

All implementations are designed for **local execution under 16 GB RAM**, with
`float32` throughout and `n_jobs = 1`.

> Update: a dedicated cross-subject thesis runner is available at
> `src/run_cross_subject_benchmark.py` (entrypoint: `run_evaluation.py`),
> with GroupKFold subject-disjoint splits and outputs:
> `fold_metrics.csv`, `subject_metrics.csv`, `stats_summary.json`.

---

## Project Structure

- `config/config.yml` – experiment configuration (subjects, CV, permutations, memory)
- `requirements.txt` – Python dependencies
- `src/`
  - `data/` – dataset preparation scripts
    - `prepare_physionet_eegbci.py` – PhysioNet EEG Motor Movement/Imagery (EEGBCI)
    - `prepare_bnci2014_001.py` – BNCI 2014-001 (BCI Competition IV 2a)
  - `config.py` – dataclass wrapper for the YAML config
  - `io/dataset.py` – loader for per-subject `.npz` motor imagery datasets
  - `denoising/pipelines.py` – bandpass, ICALabel, and GEDAI *hook*
  - `backbones/csp.py` – CSP → LDA backbone
  - `backbones/tangent_space.py` – covariance → tangent space → logistic regression
  - `evaluation/metrics.py` – empirical p-values, Cohen’s d, paired permutation tests
  - `evaluation/experiment.py` – full experiment loop (CV + permutations + tables)
  - `plots/performance.py` – group-level accuracy plots by backbone/pipeline
  - `plots/signal_integrity.py` – utilities for time/PSD overlay plots
  - `run_all.py` – run full experiment over all subjects
  - `run_plots.py` – generate performance plots from CSV outputs
  - `run_signal_integrity.py` – generate time/PSD plots for a chosen subject
- `results/`
  - `tables/subject_level_performance.csv` – per-subject, per-fold metrics
  - `stats/pipeline_comparisons.csv` – between-pipeline permutation tests + effect sizes
  - `models/` – one saved model per (subject, backbone, pipeline), e.g. `subject_1_csp_baseline.joblib` (load with `joblib.load`; dict has `W`/`clf` for CSP or `C_ref`/`clf` for tangent)
  - `figures/` – performance and signal integrity plots
- `tests/test_synthetic.py` – synthetic smoke test for baseline pipelines

---

## Supported Datasets

Two real motor imagery datasets are supported out of the box, both converted
to the common `.npz` format expected by `NpzMotorImageryDataset`:

- **PhysioNet EEG Motor Movement/Imagery (EEGBCI)**  
  - Prepared via `src/data/prepare_physionet_eegbci.py`.  
  - Default configuration uses hands vs feet runs `[6,10,14]` as in the MNE
    CSP tutorial, and epochs from \(-1\) to \(+4\) s around the cue.
  - Usage example:
    ```bash
    # From project root
    python -m src.data.prepare_physionet_eegbci --subjects 1 2 3 --out-root data/physionet_eegbci
    ```
  - Then set in `config/config.yml`:
    ```yaml
    data_root: ./data/physionet_eegbci
    subjects: [1, 2, 3]
    ```

- **BNCI 2014-001 (BCI Competition IV 2a)**  
  - Prepared via `src/data/prepare_bnci2014_001.py` using MOABB’s
    `BNCI2014_001` dataset and the `LeftRightImagery` paradigm
    (binary left vs right hand MI).
  - Usage example:
    ```bash
    python -m src.data.prepare_bnci2014_001 --subjects 1 2 3 --out-root data/bnci2014_001
    ```
  - Then set in `config/config.yml`:
    ```yaml
    data_root: ./data/bnci2014_001
    subjects: [1, 2, 3]
    ```

You can run the exact same CSP and tangent-space backbones, denoising variants,
and statistical pipeline on both datasets simply by changing `data_root` and
`subjects` in `config.yml`. This directly implements your requirement to use
**PhysioNet** plus at least one additional MI dataset in the same framework.

---

## Data Format and `float32` Precision

Per subject, data are stored as:

- `data/subject_<ID>.npz` with keys:
  - `X` – shape `(n_trials, n_channels, n_times)`
  - `y` – shape `(n_trials,)`, integer labels
  - `sfreq` – sampling frequency (scalar)
  - `ch_names` – list/array of channel names

Loading is handled by `NpzMotorImageryDataset` (`src/io/dataset.py`):

- `X` is **cast to `float32`** (or another dtype if configured) via
  `float_dtype` in `config.yml` (default: `float32`).
- All subsequent processing (bandpass, CSP, covariance, tangent space) operates
  on this `float32` representation, with careful casting inside algorithms
  where needed (e.g. tangent space projection returns `float32`).

This satisfies the **professor’s requirement** that the pipeline runs in
`float32` to reduce memory footprint.

---

## Configuration and Memory Constraints

The main configuration file is `config/config.yml`. Key fields:

- **Data and subjects**
  - `data_root: ./data`
  - `results_root: ./results`
  - `subjects: [1, 2, ...]`
- **Sampling and bandpass**
  - `sampling_rate`
  - `bandpass.l_freq`, `bandpass.h_freq` (default: 8–30 Hz)
- **Cross-validation**
  - `cv.n_splits` (e.g., 5)
  - `cv.shuffle`, `cv.random_state`
- **Permutation testing**
  - `permutation.n_subject_level` – number of label-shuffle runs per subject/pipeline  
    (recommended research setting: ≥ 1000)
  - `permutation.n_pipeline_level` – resamples for between-pipeline paired tests  
    (recommended: ≥ 10000)
- **Memory (MacBook / 16 GB RAM)**
  - `memory.float_dtype: float32` — **enforced everywhere**: load, bandpass, ICALabel, GEDAI, CSP features, tangent-space features, covariance matrices.
  - `memory.n_jobs: 1` — no multiprocessing; all backbones and scikit-learn estimators use a single process.
  - `memory.save_models: true` — after each (subject, backbone, pipeline) run, a full-data model is saved to `results/models/subject_<id>_<backbone>_<pipeline>.joblib` (CSP: filters + LDA; tangent: reference covariance + LogisticRegression). Disable with `save_models: false` if you only need tables/plots.
- **Backbones**
  - `backbones.use_csp: true/false`
  - `backbones.use_tangent_space: true/false`
- **Denoising pipelines**
  - `denoising.use_baseline: true/false`
  - `denoising.use_icalabel: true/false`
  - `denoising.use_gedai: true/false`

Memory-related design choices (optimized for MacBook / 16 GB RAM):

- **float32 throughout**: dataset load, bandpass, ICALabel, GEDAI (input and output), CSP log-variance features, tangent-space features and covariance matrices. Reduces peak RAM.
- **No multiprocessing**: all computations use `n_jobs=1` (LDA, LogReg, MNE filter, etc.).
- **GEDAI**: run in CPU-only mode with **no parallel processing** (`n_jobs=1` for `fit_raw` and `transform_raw`); input is cast to `float32`. Minimizes memory load when run sequentially per subject.
- **Sequential subject processing** in `run_experiment`; explicit `gc.collect()` after each subject.
- **No accumulation of folds** in memory; only scalar accuracies and small stats are kept.
- **Models saved incrementally** to `results/models/` after each run (one file per subject/backbone/pipeline) so you don’t keep large objects in memory.

---

## Implemented Backbones and Denoising

### CSP → LDA (`src/backbones/csp.py`)

- Bandpass (8–30 Hz) via a `scipy.signal.butter` + `filtfilt` filter.
- CSP filters computed via a generalized eigenvalue problem:
  \[
  C_1 v = \lambda (C_1 + C_2) v
  \]
  with leading and trailing components concatenated.
- Features: log-variance of CSP-projected trials.
- Classifier: `LinearDiscriminantAnalysis` (scikit-learn).

Supported denoising variants (select via `denoising` parameter and `config.yml`):

- `baseline` – bandpass only
- `icalabel` – bandpass → ICALabel artifact rejection
- `gedai` – bandpass → GEDAI (see **GEDAI integration** note below)

### Tangent Space → Logistic Regression (`src/backbones/tangent_space.py`)

- Bandpass filtering identical to CSP backbone.
- Trial-wise covariance matrices.
- Regularized mean covariance `C_ref` (arithmetic mean + small diagonal loading)
  to ensure positive definiteness for Cholesky decomposition.
- Projection to the tangent space at `C_ref` using matrix logarithms (`scipy.linalg.logm`),
  upper triangle flattened to form feature vectors (`float32`).
- Classifier: `LogisticRegression(solver="lbfgs", max_iter=1000, n_jobs=1)`.

Supported denoising variants:

- `baseline`, `icalabel`, `gedai` (same semantics as CSP backbone).

---

## Denoising Implementations

### Bandpass (`bandpass_filter`)

- Implemented in `src/denoising/pipelines.py`.
- Uses a 5th-order Butterworth filter with zero-phase `filtfilt`.
- Preserves the input dtype (`float32` when loaded through the dataset loader).

### ICALabel (`apply_icalabel`)

- Implemented in `src/denoising/pipelines.py` using MNE:
  - Builds a `RawArray` by concatenating trials in time, sets montage to `standard_1020` (required for ICALabel topoplot features).
  - Applies bandpass (again) at 8–30 Hz (with `n_jobs=1`).
  - Fits an ICA model and labels components via `mne_icalabel.label_components`.
  - Automatically excludes artifactual ICs (eye, muscle, heart).
  - Returns cleaned data reshaped back to `(n_trials, n_channels, n_times)`, cast
    to the original dtype.
- **Dependency:** `mne-icalabel` needs either `onnxruntime` or `torch` for its classifier. `requirements.txt` includes `onnxruntime`; install with `pip install -r requirements.txt` (or use PyTorch if you prefer).

### GEDAI (`apply_gedai`)

- Implemented in `src/denoising/pipelines.py` using the official `gedai` package:
  - Builds an MNE `RawArray` from epoched data.
  - Instantiates `Gedai(wavelet_type="haar", wavelet_level=0)`, then calls
    `fit_raw(raw, n_jobs=1)` and `transform_raw(raw, n_jobs=1)`.
  - **No parallel processing**: `n_jobs=1` is used for all GEDAI steps to minimize memory.
- If the package is not installed or fit/transform fails (e.g. missing leadfield), the function falls back to identity (passthrough) and issues a warning.
- For a custom leadfield or API, adapt the `Gedai(...)` construction and calls inside `apply_gedai`.

---

## Cross-Validation and Statistical Testing

### Within-Subject Cross-Validation

- Implemented in both backbones using `StratifiedKFold`:
  - `cv.n_splits` folds (e.g., 5).
  - `cv.shuffle` and `cv.random_state` control reproducible splits.
- For each subject, backbone, and pipeline:
  - Train on `k-1` folds, test on the held-out fold.
  - Store fold-level accuracies, mean, and standard deviation.

### Empirical Chance Level (Per Subject)

- Implemented in `src/evaluation/experiment.py`:
  - For each subject, backbone, and pipeline:
    - Shuffle labels and re-run the full pipeline (`run_csp_pipeline` or
      `run_tangent_space_pipeline`) for `permutation.n_subject_level` iterations.
    - Aggregate null accuracies and compute an empirical p-value:
      \[
      p = \text{proportion}(\text{null\_accuracy} \ge \text{real\_accuracy})
      \]
  - Results stored in `subject_level_performance.csv` as `p_empirical`.

### Between-Pipeline Comparisons and Effect Sizes

- Implemented in `src/evaluation/metrics.py` and `src/evaluation/experiment.py`:
  - For each backbone:
    - Compare `gedai` vs `baseline` and `icalabel` vs `baseline` (if enabled).
  - **Paired permutation test** (`scipy.stats.permutation_test`) on mean accuracies
    across subjects, with:
    - `n_resamples = permutation.n_pipeline_level`
    - `permutation_type="pairing"` and two-sided alternative.
  - **Cohen’s d** computed using pooled standard deviation:
    \[
    d = \frac{\bar{x}_1 - \bar{x}_2}{s_\text{pooled}}
    \]
  - Results written to `results/stats/pipeline_comparisons.csv`.

This realizes the **Combrisson & Jerbi** recommendation for empirical chance
assessment and supplements it with **effect size reporting**.

---

## Plots and Signal Integrity Analysis

### Performance Plots

After running `src.run_all`, generate performance plots via:

```bash
python -m src.run_plots --config config/config.yml
```

This creates figures in `results/figures/`:

**1. Performance by pipeline**

- `performance_csp.png`, `performance_tangent.png`
- Boxplots of mean CV accuracy per pipeline (baseline, icalabel, gedai).
- Stripplot of subject-level points for inter-subject variability.

**2. Variability (delta accuracy)**

- `variability_csp.png`, `variability_tangent.png`
- Distribution of Δ accuracy: (GEDAI − baseline) and (ICALabel − baseline) across subjects.
- Supports the professor’s requirement: “percentage improved, median improvement, SD across subjects”.

### Signal Integrity: Pre–Post Denoising and PSD

As recommended by the professor, pre–post denoising is visualized by **overlaying the original EEG trace with the denoised EEG trace for each channel independently**, in the style of GEDAI’s `compare.py` ([gedai/viz/compare.py](https://github.com/neurotuning/gedai/blob/main/gedai/viz/compare.py)).

Use:

```bash
python -m src.run_signal_integrity --config config/config.yml --subject 1 --trial 0
```

Optional: open GEDAI’s **interactive** overlay (keyboard scroll, scale, overlay/diff modes):

```bash
python -m src.run_signal_integrity --config config/config.yml --subject 1 --trial 0 --interactive
```

The script:

- Loads the selected subject and trial.
- **Pre–post overlay (per channel):** For each denoising method (ICALabel, GEDAI) that is enabled:
  - Builds **original** = bandpassed data (all channels) and **denoised** = method output.
  - Plots a **stacked multi-channel figure**: each channel is one row; red = original, blue = denoised (same style as GEDAI `plot_mne_style_overlay_interactive`).
  - Saves a **static PNG**, e.g. `prepost_overlay_subj1_trial0_icalabel.png`, `prepost_overlay_subj1_trial0_gedai.png`.
- **Single-channel time overlay:** Raw, bandpass, ICALabel, GEDAI on one plot for a chosen motor channel (`signal_integrity.channels_of_interest`, default C3/C4).
- **PSD comparison:** Welch PSD for each condition with alpha (8–12 Hz) and beta (13–30 Hz) bands highlighted.

Outputs in `results/figures/signal_integrity/`:

- `prepost_overlay_subj<N>_trial<K>_icalabel.png`, `prepost_overlay_subj<N>_trial<K>_gedai.png` — original vs denoised, all channels.
- `time_subj<N>_trial<K>_<ch>.png` — single-channel time overlay.
- `psd_subj<N>_<ch>.png` — PSD comparison.

This supports **physiological plausibility**: check that alpha/mu/beta are preserved while artifacts are reduced.

---

## Running the Pipeline

1. **Install and set up the environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare data**

   - Place per-subject files as `data/subject_<ID>.npz` with keys
     `X, y, sfreq, ch_names`.
   - Update `config/config.yml` with the list of `subjects`.

3. **Run the full experiment**

   ```bash
   python -m src.run_all --config config/config.yml
   ```

4. **Generate figures**

   ```bash
   python -m src.run_plots --config config/config.yml
   python -m src.run_signal_integrity --config config/config.yml --subject 1 --trial 0
   ```

5. **Run tests** (synthetic data; no real EEG required)

   ```bash
   export MPLBACKEND=Agg
   pytest tests/ -v
   ```
   See **TESTING.md** for all test cases (experiment, plots, signal integrity, model save/load, dataset loader).

---

## Summary of Alignment with Requirements

- **CSP and Tangent Space backbones** – implemented as in the benchmarking paper.
- **Denoising variants (Baseline, ICALabel, GEDAI hook)** – fully wired at the
  interface level; GEDAI requires user-specific configuration.
- **Within-subject k-fold CV** – implemented with configurable `n_splits`.
- **Empirical chance tests** – implemented per subject, per pipeline via label
  shuffling.
- **Between-pipeline paired permutation tests** – implemented with configurable
  resamples and pooled-standard-deviation **Cohen’s d**.
- **Memory constraints** – sequential subjects, `float32` precision, `n_jobs=1`,
  no multiprocessing.
- **Signal integrity** – dedicated script for time-domain and PSD comparisons
  focused on motor cortex channels, enabling inspection of alpha/mu/beta bands.

The only intentional **hook point** left for you to finalize is the exact GEDAI
call inside `apply_gedai`, which must be adapted to your forward model and the
particular API version you are using.
