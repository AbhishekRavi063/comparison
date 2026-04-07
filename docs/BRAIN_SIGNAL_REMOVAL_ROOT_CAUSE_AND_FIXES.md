# Brain Signal Removal: Root Causes, Possibilities, and Fixes

This document lists **root causes** and **possibilities** for denoising pipelines (ICALabel, GEDAI) removing real brain signal (alpha/mu/beta), and the **fixes** applied.

---

## 1. Root causes and possibilities (list)

| # | Possibility | Root cause | Fix applied |
|---|-------------|------------|-------------|
| 1 | **GEDAI had no motor-channel check** | Retention used only global median band power. If GEDAI over-removed in motor cortex (C3/C4/CZ), the global median could still be > threshold. | **GEDAI** now uses the same **motor-channel + full-band + alpha + beta** retention logic as ICALabel via `_retention_ratios()`. |
| 2 | **GEDAI had no alpha/beta sub-band checks** | Only the full decode band (8–30 Hz) was checked. Alpha (8–12) or beta (13–30) could be suppressed while full-band ratio stayed high. | **GEDAI** retention now uses **effective_ratio = min(full, motor, alpha_band, beta_band)** via `_retention_ratios()`. |
| 3 | **Retention thresholds were too permissive** | 0.65 (median) and 0.50 (hard) allowed up to 35–50% power loss. | **Both pipelines**: `RETENTION_MEDIAN_MIN = 0.75`, `RETENTION_HARD_MIN = 0.60`. |
| 4 | **ICALabel did not enforce alpha/beta separately** | Only full-band and motor full-band were used; alpha or beta could drop without failing the guard. | **ICALabel** now uses **effective_ratio** from `_retention_ratios()` (full + motor + alpha + beta). |
| 5 | **Check script used only trial-averaged PSD** | Averaging over trials could hide over-removal on some trials. | **Check script** now reports **per-trial minimum** alpha/beta ratio (worst trial) and flags if any ratio < 0.75. |
| 6 | **No single place defining “preservation”** | Full-band, motor, alpha, and beta were computed in different ways in ICALabel vs GEDAI. | **Shared** `_retention_ratios()` in `pipelines.py` computes **effective = min(full, motor, alpha, beta)** for both pipelines. |

---

## 2. Fixes applied (summary)

### 2.1 `src/denoising/pipelines.py`

- **Constants**
  - `ICALABEL_RETENTION_MEDIAN_MIN`: 0.65 → **0.75**
  - `ICALABEL_RETENTION_HARD_MIN`: 0.50 → **0.60**
  - `GEDAI_RETENTION_MEDIAN_MIN`: 0.65 → **0.75**
  - `GEDAI_RETENTION_HARD_MIN`: 0.50 → **0.60**
  - Added **`ALPHA_BAND = (8.0, 12.0)`** and **`BETA_BAND = (13.0, 30.0)`**.

- **New helper: `_retention_ratios()`**
  - Inputs: `x_clean`, `x_band_ref`, `sfreq`, `l_freq`, `h_freq`, `ch_names`.
  - Computes: full-band ratio, motor-channel (C3/C4/CZ) ratio, alpha-band ratio, beta-band ratio.
  - Returns: **effective_ratio = min(full, motor, alpha, beta)**.
  - Used by both ICALabel and GEDAI so neither can pass by only preserving part of the band.

- **ICALabel**
  - Replaced separate `median_ratio` and `motor_ratio` with **effective_ratio = _retention_ratios(...)**.
  - Retry (ocular+heart only) and hard fallback now use this **effective_ratio**.

- **GEDAI**
  - Replaced **median_ratio** (global only) with **effective_ratio = _retention_ratios(...)** (adds motor + alpha + beta).
  - Retry (higher noise_multiplier) and hard fallback now use **effective_ratio**.

### 2.2 `scripts/check_brain_signal_preservation.py`

- **Per-trial minimum ratio**
  - For ICALabel and GEDAI, computes alpha and beta power **per trial** vs bandpass.
  - Reports **minimum ratio across all trials** (worst trial).
- **Threshold**
  - Flags **possible over-removal** when ratio < **0.75** (aligned with retention median min).
- **Message**
  - Prints “Per-trial minimum ratio (worst trial)” and “OK” or “⚠ … over-removal”.

---

## 3. Behaviour after fixes

- **When retention passes (effective_ratio ≥ 0.75)**  
  Denoised output is used; full band, motor channels, alpha, and beta are all preserved relative to bandpass.

- **When retention fails soft (effective_ratio < 0.75 but ≥ 0.60)**  
  - **ICALabel**: Retry with only ocular+heart ICs; if effective_ratio still < 0.75, continue; if then < 0.60, fall back to bandpass.  
  - **GEDAI**: Retry with higher noise_multiplier; if effective_ratio still < 0.75, continue; if then < 0.60, fall back to bandpass.

- **When retention fails hard (effective_ratio < 0.60)**  
  Both pipelines **return bandpass (no denoising)** so real brain signal is not removed.

- **Check script**  
  - Trial-averaged ratios and **per-trial minimum** both reported.  
  - Any pipeline/trial with alpha or beta ratio < 0.75 is flagged as possible over-removal.

---

## 4. How to verify

1. **Run preservation check (one subject)**  
   ```bash
   python scripts/check_brain_signal_preservation.py --config config/config_alljoined_smoke_1sub.yml --subject 1 --channel Cz
   ```  
   Confirm trial-averaged and per-trial minimum ratios ≥ 0.75 (or that fallback to bandpass occurs when not).

2. **Run integration tests**  
   ```bash
   export MPLBACKEND=Agg && pytest tests/test_integration.py -v
   ```  
   All cases (including ICALabel case_2) should pass.

3. **Run full pipeline and inspect outputs**  
   If retention often fails, more results will be identical to baseline (bandpass); that is expected and indicates the guards are preventing brain signal removal.

---

## 5. Summary

- **Root cause**: GEDAI lacked motor and alpha/beta checks; both pipelines used loose thresholds and no unified definition of “preservation.”
- **Fixes**: Unified **effective_ratio = min(full, motor, alpha, beta)** in **`_retention_ratios()`**, stricter **0.75/0.60** thresholds, and a **per-trial minimum** in the check script so no single trial can show over-removal without being detected or guarded.
