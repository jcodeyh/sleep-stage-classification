#!/usr/bin/env python3
"""Add demographics to the DREAMT clean wearable dataset.

DREAMT participant_info.csv columns: SID, AGE, GENDER, BMI, OAHI, AHI,
Mean_SaO2, Arousal Index, MEDICAL_HISTORY, Sleep_Disorders.

We map only what aligns with the MESA/STAGES demo_* schema. Smoking, race,
ESS, ISI, RLS, CPAP are not collected in DREAMT and are emitted as NaN so
the column set matches and cross-domain evaluation can median-impute them.

Usage:
    python scripts/add_dreamt_demographics.py
    python scripts/add_dreamt_demographics.py \
        --input datasets/dreamt/dreamt_clean_wearable_temporal.parquet \
        --output datasets/dreamt/dreamt_clean_wearable_full.parquet
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Columns padded with NaN so the schema lines up exactly with the
# combined MESA+STAGES parquet. Names (incl. the 30-char truncations from
# add_stages_demographics.py) come from inspecting combined_clean_wearable_full
# in datasets/combined/. cross_domain_eval still imputes any NaN.
NAN_DEMO_COLS = [
    "demo_smoking",
    "demo_avg_efficiency", "demo_avg_fragmentation",
    "demo_avg_onset_latency", "demo_avg_sleep_min",
    "demo_race_white",
    "demo_race_black", "demo_race_chinese", "demo_race_hispanic",
    "demo_race_black_or_african_ame", "demo_race_asian",
    "demo_race_american_indian_or_a", "demo_race_native_hawaiian_or_o",
    "demo_race_other",
    "demo_epworth", "demo_insomnia", "demo_rls", "demo_cpap",
    "demo_sleep_apnea",
]


def build_demographics(participant_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(participant_csv)
    logger.info("Loaded participant_info.csv: %d rows, columns: %s",
                len(df), list(df.columns))

    demo = pd.DataFrame({"subject_id": df["SID"].astype(str)})
    demo["demo_age"] = pd.to_numeric(df["AGE"], errors="coerce")
    demo["demo_is_male"] = (df["GENDER"].astype(str).str.upper() == "M").astype("Int64")
    demo["demo_bmi"] = pd.to_numeric(df["BMI"], errors="coerce")

    for col in NAN_DEMO_COLS:
        demo[col] = np.nan

    # Deduplicate just in case
    demo = demo.drop_duplicates(subset="subject_id", keep="first").reset_index(drop=True)

    logger.info("Built demographics for %d subjects", len(demo))
    for col in demo.columns:
        if col == "subject_id":
            continue
        nn = demo[col].notna().sum()
        logger.info("  %s: %d/%d non-null (%.0f%%)",
                    col, nn, len(demo), 100 * nn / len(demo))
    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Add demographics to DREAMT clean wearable dataset"
    )
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "datasets" / "dreamt" / "dreamt_clean_wearable_temporal.parquet",
        help="Input parquet (post-temporal-features)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "datasets" / "dreamt" / "dreamt_clean_wearable_full.parquet",
        help="Output parquet with demographics merged in",
    )
    parser.add_argument(
        "--participant-csv", type=Path,
        default=(PROJECT_ROOT / "datasets" / "dreamt" / "physionet.org"
                 / "files" / "dreamt" / "2.1.0" / "participant_info.csv"),
        help="Path to DREAMT participant_info.csv",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} epochs, {df.subject_id.nunique()} subjects")

    demo = build_demographics(args.participant_csv)

    n_before = len(df)
    df = df.merge(demo, on="subject_id", how="left")
    assert len(df) == n_before, "Merge changed row count!"

    demo_cols = [c for c in df.columns if c.startswith("demo_")]
    print(f"\nAdded {len(demo_cols)} demographic columns:")
    for col in demo_cols:
        nn = df[col].notna().sum()
        print(f"  {col:<35} non-null: {nn:,}/{len(df):,} ({100 * nn / len(df):.0f}%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nSaved: {args.output} ({size_mb:.1f} MB, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
