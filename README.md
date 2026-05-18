# Wearable Sleep Stage Classification

ML pipelines and models for predicting PSG-scored sleep stages from **consumer-wearable-collectible features** (HR, HRV, SpO2, actigraphy, circadian time). Built on three clinical / research datasets:

- **MESA** (NSRR — Multi-Ethnic Study of Atherosclerosis), 1,744 subjects
- **STAGES** (NSRR), 1,326 subjects across 13 sites
- **DREAMT** (PhysioNet — Empatica E4 wrist PPG), 100 subjects, used as a held-out cross-sensor test domain

Best results to date: in-domain (MESA, 5-class) Cohen's κ = **0.484**; cross-site (MESA↔STAGES) κ = **0.536**; zero-shot to DREAMT (Bi-LSTM HMM) κ = **0.146**, fine-tuned (n=50 cal subjects) → **0.238**. Full results & discussion: [docs/REPORT.md](docs/REPORT.md).

---

## Repository layout

```
.
├── data_pipeline/         # Downloading and building per-dataset parquets
│   ├── download/          # NSRR / PhysioNet downloaders
│   ├── build/             # EDF + scoring + actigraphy → parquet, temporal features, merge
│   ├── demographics/      # Per-dataset demographics joins
│   └── utils/             # Probing, validation, repair helpers
├── training_pipeline/     # Train / evaluate models (baselines, cross-domain, zero-shot)
├── reporting/             # Figure generation for the report
├── src/sleepkit_custom/   # Custom feature-set registrations (consumed by run_training.py)
├── configs/               # JSON configs for sleepkit-based training
├── docs/                  # REPORT.md, figures, STAGES documentation
├── notebooks/             # Exploratory analysis (cleared outputs)
├── CLAUDE.md              # Project-internal architecture notes
├── activate.sh            # Helper to source the local venv on Kodiak HPC
├── requirements.txt
└── LICENSE                # MIT
```

---

## Data sources (not redistributed)

| Dataset | Source | Access |
|---|---|---|
| MESA | [sleepdata.org/datasets/mesa](https://sleepdata.org/datasets/mesa) | NSRR credentialed |
| STAGES | [sleepdata.org/datasets/stages](https://sleepdata.org/datasets/stages) | NSRR credentialed |
| DREAMT | [physionet.org/content/dreamt/2.1.0/](https://physionet.org/content/dreamt/2.1.0/) | PhysioNet credentialed |

This repository contains **no patient data**. Obtain the source datasets from the links above; the build scripts produce the intermediate parquet files used by the training pipeline.

---

## Quick start

```bash
# 1. Environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. NSRR token (one-time)
echo "YOUR_TOKEN" > ~/.nsrr_token

# 3. Download data (large; pick a target directory with terabytes free)
python data_pipeline/download/smart_download.py --dataset mesa
python data_pipeline/download/smart_download.py --dataset stages

# 4. Build per-dataset parquets
python data_pipeline/build/build_mesa.py
python data_pipeline/build/build_stages.py
python data_pipeline/build/build_dreamt.py

# 5. Add temporal context features
python data_pipeline/build/add_temporal_features.py --id-col mesa_id \
  --input datasets/mesa/mesa_clean_wearable.parquet
python data_pipeline/build/add_temporal_features.py --id-col subject_id \
  --input datasets/stages/stages_clean_wearable.parquet
python data_pipeline/build/add_temporal_features.py --id-col subject_id \
  --input datasets/dreamt/dreamt_clean_wearable.parquet

# 6. Add demographics
python data_pipeline/demographics/add_mesa_demographics.py
python data_pipeline/demographics/add_stages_demographics.py
python data_pipeline/demographics/add_dreamt_demographics.py

# 7. Merge MESA + STAGES (DREAMT stays separate as held-out test domain)
python data_pipeline/build/merge_datasets.py

# 8. Train baselines
python training_pipeline/train_baselines.py \
  --model xgboost --n-classes 5 \
  --data datasets/combined/combined_clean_wearable_full.parquet

# 9. Cross-domain & zero-shot evaluation
python training_pipeline/cross_domain_eval.py --zscore-subject --hmm-smooth
python training_pipeline/dreamt_zeroshot_eval.py --model bilstm --seq-len 120
```

See `<script>.py --help` for full options on every step.

---

## Pipeline (one diagram)

```
NSRR / PhysioNet
       │
       ▼
data_pipeline/download/       ─ smart_download.py per dataset
       │
       ▼
data_pipeline/build/          ─ build_{mesa,stages,dreamt}.py
                                ─ add_temporal_features.py  (rolling + delta)
       │
       ▼
data_pipeline/demographics/   ─ add_{mesa,stages,dreamt}_demographics.py
       │
       ▼
data_pipeline/build/          ─ merge_datasets.py (MESA + STAGES)
       │
       ▼
training_pipeline/            ─ train_baselines.py
                                ─ cross_domain_eval.py
                                ─ dreamt_zeroshot_eval.py
       │
       ▼
reporting/                    ─ generate_figures.py
docs/REPORT.md
```

The 17-feature wearable core, the temporal expansion, and dataset-specific quirks (NSRR file layouts, scoring formats, demographics CSVs) are documented in [CLAUDE.md](CLAUDE.md).

---

## HPC notes

The project was developed on the Kodiak PBS cluster. The `activate.sh` helper assumes the local venv lives at `./venv/`. PBS submission scripts are intentionally not included in this repository (they were Kodiak-specific) — translate the commands in the Quick Start into your own scheduler as needed.

---

## License

MIT. See [LICENSE](LICENSE).
