# Sleep Stage Classification Project

## Architecture

This project builds sleep staging ML models using **consumer-wearable-collectible features** extracted from clinical PSG datasets (MESA, STAGES) plus DREAMT as a held-out cross-sensor test domain. The pipeline is dataset → features → temporal → demographics → merge → train → export.

### Critical: NSRR Dataset Heterogeneity

**Each NSRR dataset has a unique file structure, naming convention, and data format.** Never assume one dataset's format applies to another. When adding a new NSRR dataset:

1. **Download samples first** — fetch metadata and a few files before writing any parser
2. **Inspect actual paths** — use the NSRR API to list files; don't assume standard paths like `polysomnography/edfs/`
3. **Write dataset-specific scripts** — parallel scripts per dataset, not one shared script with format flags

| Aspect | MESA | STAGES | DREAMT |
|---|---|---|---|
| Scoring format | NSRR XML | CSV (Start Time, Duration, Event) | Inline in 64Hz CSV |
| Actigraphy format | CSV (Actiwatch) | JSON (Huami Arc), Deflate64 ZIP | Empatica E4 (ENMO from 32Hz triaxial accel) |
| HR signal | Pre-computed HR/DHR in EDF | Derive HR from raw EKG | 1Hz from Empatica E4 PPG |
| SpO2 | Available | Available | **Not available** (no E4 sensor) |
| Subject IDs | 4-digit int | `{SITE}{5-digit}` (e.g. STNF00001) | `S###` |
| ID column in parquet | `mesa_id` (int) | `subject_id` (str) | `subject_id` (str) |
| ID column after merge | `subject_id` (`mesa_XXXX`) | `subject_id` (`stages_STNFXXXXX`) | (DREAMT stays separate) |
| File paths | Standard NSRR (`polysomnography/edfs/`) | Non-standard (`original/STAGES PSGs/{SITE}/`) | PhysioNet flat (`data_64Hz/`) |
| Valid epoch rate | ~98% | ~48% (long unscored pre/post-sleep) | ~76% |
| Scoring CSV quirks | Clean 3-column | Free-text events with commas (use `names=` + `on_bad_lines`) | n/a (one-row-per-64Hz-sample) |
| EKG availability | All subjects | ~295 subjects have SpO2-only (no EKG), skipped | none (PPG only) |
| Demographics CSV | Clean per-subject | Duplicate rows per subject (multiple visits), must deduplicate | `participant_info.csv` |
| Wall clock | Yes | Yes | **No** — DREAMT timestamps start at 0 (drop `tod_*`) |

### Wearable Feature Set (17 features)

Only features collectible by consumer wearables (optical PPG, SpO2 sensor, accelerometer, light sensor):

- **HR** (4): hr_mean, hr_std, hr_min, hr_max
- **HRV** (4): hrv_mean_nn, hrv_sdnn, hrv_rmssd, hrv_pnn50
- **SpO2** (3): spo2_mean, spo2_std, spo2_min
- **Actigraphy** (4): actig_activity, actig_wake, actig_white_light, actig_log_activity
- **Circadian** (2): tod_cos, tod_sin

**Cut 11 PSG-only features:** hrv_lf_hf, resp_rate, leg_move_* (3), pos_* (6)

With temporal features (rolling windows + deltas), expands to ~45 features.

For cross-domain eval against DREAMT, the shared-core auto-filter drops any feature with ≥80% NaN in any source — yielding **29 features** (loses SpO2×3 + spo2-derived rolling/delta + `tod_*` + several demographics).

## Key Scripts

### Data pipeline

| Script | Purpose |
|---|---|
| `data_pipeline/download/smart_download.py` | Download MESA or STAGES from NSRR (`--dataset {mesa,stages}`) |
| `data_pipeline/download/download_mesa_demographics.py` | Pull MESA demographics CSV separately |
| `data_pipeline/build/build_mesa.py` | MESA: EDF+XML+CSV → parquet |
| `data_pipeline/build/build_stages.py` | STAGES: EDF+CSV scoring+JSON actigraphy → parquet |
| `data_pipeline/build/build_dreamt.py` | DREAMT: 64Hz CSV → parquet |
| `data_pipeline/build/add_temporal_features.py` | Rolling windows + deltas (works for all three, `--id-col`) |
| `data_pipeline/build/merge_datasets.py` | Combine MESA + STAGES with provenance columns |
| `data_pipeline/demographics/add_mesa_demographics.py` | MESA demographics from sleep dataset CSV |
| `data_pipeline/demographics/add_stages_demographics.py` | STAGES demographics from harmonized CSV |
| `data_pipeline/demographics/add_dreamt_demographics.py` | DREAMT demographics from `participant_info.csv` |
| `data_pipeline/utils/probe_stages_channels.py` | Discover EDF channel names / rates before parsing |
| `data_pipeline/utils/repair_actigraphy_zip.py` | Re-extract Deflate64 JSONs via system unzip |
| `data_pipeline/utils/validate_data.py` | Per-subject quality audit |
| `data_pipeline/utils/audit_zscore_nans.py` | Per-feature NaN rate after per-subject z-scoring |

### Training pipeline

| Script | Purpose |
|---|---|
| `training_pipeline/train_baselines.py` | In-domain train & evaluate (LR, RF, XGBoost, LightGBM, 1D-CNN) |
| `training_pipeline/cross_domain_eval.py` | Cross-site A/B/C/D conditions (MESA ↔ STAGES) with z-score / HMM / self-train flags |
| `training_pipeline/dreamt_zeroshot_eval.py` | Zero-shot + few-shot fine-tune evaluation on DREAMT |
| `training_pipeline/run_training.py` | SleepKit-based training (uses `src/sleepkit_custom`) |

### Reporting

| Script | Purpose |
|---|---|
| `reporting/generate_figures.py` | Consolidated figure generator (MESA + Combined-Tier1 suites) for `docs/REPORT.md` |

## Environment

```bash
module purge && module load python/3.12.8
source ./venv/bin/activate         # or: source activate.sh
```

`activate.sh` is checked in and assumes the venv is at `./venv/`.

## Data Paths

Pipeline intermediates — each script reads the previous step's output. All under `datasets/` (which is `.gitignore`d):

| Step | Script | Output |
|---|---|---|
| 1 | `data_pipeline/build/build_mesa.py` | `datasets/mesa/mesa_clean_wearable.parquet` |
| 2 | `data_pipeline/build/add_temporal_features.py --id-col mesa_id` | `datasets/mesa/mesa_clean_wearable_temporal.parquet` |
| 3 | `data_pipeline/demographics/add_mesa_demographics.py` | `datasets/mesa/mesa_clean_wearable_full.parquet` |
| 1 | `data_pipeline/build/build_stages.py` | `datasets/stages/stages_clean_wearable.parquet` |
| 2 | `data_pipeline/build/add_temporal_features.py --id-col subject_id` | `datasets/stages/stages_clean_wearable_temporal.parquet` |
| 3 | `data_pipeline/demographics/add_stages_demographics.py` | `datasets/stages/stages_clean_wearable_full.parquet` |
| 1 | `data_pipeline/build/build_dreamt.py` | `datasets/dreamt/dreamt_clean_wearable.parquet` |
| 2 | `data_pipeline/build/add_temporal_features.py --id-col subject_id` | `datasets/dreamt/dreamt_clean_wearable_temporal.parquet` |
| 3 | `data_pipeline/demographics/add_dreamt_demographics.py` | `datasets/dreamt/dreamt_clean_wearable_full.parquet` |
| 4 | `data_pipeline/build/merge_datasets.py` | `datasets/combined/combined_clean_wearable_full.parquet` ← **training input** |

- NSRR token: `~/.nsrr_token`
- Scripts that take `--id-col`: `add_temporal_features.py`, `train_baselines.py`. Default is `subject_id` (combined/STAGES/DREAMT). Use `--id-col mesa_id` for MESA-only parquets.

## Leaky Features

`epochs_since_wake`, `sleep_pressure`, and `night_position` are derived from `sleep_stage` — excluded in training via `LEAKY_COLS`.

## Common Pitfalls

- **CSV parsing**: STAGES scoring CSVs have free-text events with commas. Always use `names=[...]` + `on_bad_lines='warn'` instead of relying on header inference.
- **Deflate64 ZIP**: Python's `zipfile` module cannot handle Deflate64 (type 9). Use system `unzip` via `subprocess.run()`.
- **Demographics deduplication**: NSRR harmonized CSVs may have multiple rows per subject (multiple visits). Always `drop_duplicates(subset='subject_code', keep='first')` before merging.
- **SpO2-only EDFs**: ~295 STAGES subjects have only SpO2 channels (no EKG). These are skipped during build — no HR/HRV derivation possible.
- **MESA still has PSG-only columns**: `mesa_clean_wearable_full.parquet` has 78 columns including PSG-only features. The merge keeps all columns; training scripts select the 17 wearable features.
- **DREAMT 64Hz HR deduplication**: HR is sampled at 1Hz but written at 64Hz row cadence — recover native samples via `[::64]` before clipping/HRV derivation.
- **DREAMT no `tod_*`**: timestamps start at 0 (no wall-clock anchor). `build_dreamt.py` emits NaN for circadian features; the 29-feature cross-domain core drops them.

## Known Data Issues

- **`night_position`** is soft leakage (fractional position within the night, 0–1, correlates with stage distribution) — added to `LEAKY_COLS` in `train_baselines.py`.
- **`demo_smoking`** is ordinal 0=never / 1=former / 2=current in all three dataset parquets after their respective demographics scripts run.
- **OxStatus not used for SpO2 gating**: MESA EDFs contain an `OxStatus` channel encoding probe-off/poor-signal events, but `build_mesa.py` does not use it to gate SpO2 validity. The current 50–100 range filter catches most artifacts, but a small number of epochs with probe-off artifacts in the 50–100 range may pass through uncaught. Enhancement: gate SpO2 extraction on `OxStatus == 0` (good signal) when the channel is available.
- **CNN test-set shuffling**: `cross_domain_eval.make_sequences` shuffles test sequences after windowing, which blocks HMM smoothing on CNN output. ~10 LoC to fix; deferred while LGBM+HMM dominated.
