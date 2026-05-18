#!/usr/bin/env python3
"""
Merge demographic and clinical features from the MESA sleep dataset CSV
into the temporal parquet file.

These are per-subject features (constant across all epochs for a given subject)
that a wearable app could collect during onboarding or from a health profile.

Input:  datasets/mesa/mesa_clean_wearable_temporal.parquet
        datasets/mesa/mesa-sleep-dataset-0.8.0.csv
Output: datasets/mesa/mesa_clean_wearable_full.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPORAL_PATH = PROJECT_ROOT / "datasets" / "mesa" / "mesa_clean_wearable_temporal.parquet"
DEMO_PATH = PROJECT_ROOT / "datasets" / "mesa" / "mesa-sleep-dataset-0.8.0.csv"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "mesa" / "mesa_clean_wearable_full.parquet"

# ── Feature selection ─────────────────────────────────────────────────────────
# Only include features that:
# 1. Could be known from a wearable app onboarding (age, sex, BMI)
# 2. OR are clinical characteristics detectable by a wearable over time (AHI proxy)
# 3. Are NOT derived from PSG sleep staging (that would leak the label)

DEMOGRAPHIC_COLS = {
    # Core demographics (collected during onboarding)
    "sleepage5c": "demo_age",          # Age in years
    "gender1": "demo_is_male",         # 0=Female, 1=Male
    "race1c": "demo_race",             # 1=White, 2=Chinese, 3=Black, 4=Hispanic
    "bmi5c": "demo_bmi",              # Body mass index

    # Self-reported sleep habits (collected via questionnaire)
    "epslpscl5c": "demo_epworth",      # Epworth Sleepiness Scale (0-24)
    "slpapnea5": "demo_sleep_apnea",   # Self-reported sleep apnea (0/1)
    "insmnia5": "demo_insomnia",       # Self-reported insomnia (0/1)
    "rstlesslgs5": "demo_rls",         # Restless legs syndrome (0/1)
    "cpap5": "demo_cpap",             # CPAP use (0/1)
    "smkstat5": "demo_smoking",        # Smoking: 0=Never, 1=Former, 2+=Current

    # Habitual sleep patterns from actigraphy (wearable-derivable over time)
    "avgmainsleep5": "demo_avg_sleep_min",       # Average main sleep duration (min)
    "avgefficiency5": "demo_avg_efficiency",      # Average sleep efficiency (%)
    "avgfragmentation5": "demo_avg_fragmentation", # Average sleep fragmentation index
    "avgonsetlatency5": "demo_avg_onset_latency",  # Average sleep onset latency (min)
}


def main():
    print(f"Loading temporal parquet: {TEMPORAL_PATH}")
    df = pd.read_parquet(TEMPORAL_PATH)
    print(f"  {len(df):,} epochs, {df.mesa_id.nunique()} subjects, {len(df.columns)} columns")

    print(f"\nLoading MESA sleep dataset: {DEMO_PATH}")
    demo = pd.read_csv(DEMO_PATH, low_memory=False)
    print(f"  {len(demo)} subjects, {len(demo.columns)} columns")

    # Select and rename demographic columns
    keep_cols = ["mesaid"] + list(DEMOGRAPHIC_COLS.keys())
    demo_subset = demo[keep_cols].rename(columns={"mesaid": "mesa_id", **DEMOGRAPHIC_COLS})
    print(f"  Selected {len(DEMOGRAPHIC_COLS)} demographic features")

    # Check coverage
    our_ids = set(df.mesa_id.unique())
    demo_ids = set(demo_subset.mesa_id.unique())
    matched = our_ids & demo_ids
    print(f"  Matched: {len(matched)}/{len(our_ids)} subjects ({100*len(matched)/len(our_ids):.1f}%)")

    # Encode race as one-hot (4 categories)
    print("\n  Encoding race as one-hot...")
    race_dummies = pd.get_dummies(demo_subset["demo_race"], prefix="demo_race").astype(np.float32)
    race_dummies.columns = ["demo_race_white", "demo_race_chinese", "demo_race_black", "demo_race_hispanic"]
    demo_subset = pd.concat([demo_subset, race_dummies], axis=1)
    demo_subset = demo_subset.drop(columns=["demo_race"])

    # Recode smoking to ordinal: 0=never, 1=former, 2=current (matches STAGES encoding)
    # smkstat5 is coded 0=Never, 1=Former, 2+=Current; clip to cap any values above 2
    print("  Recoding smoking to ordinal (0=never, 1=former, 2=current)...")
    demo_subset["demo_smoking"] = demo_subset["demo_smoking"].clip(0, 2).astype(np.float32)

    # Summary of demographic features
    print(f"\n  Demographic features ({len(demo_subset.columns) - 1}):")
    for col in demo_subset.columns:
        if col == "mesa_id":
            continue
        vals = demo_subset[col].dropna()
        miss = demo_subset[col].isna().sum()
        print(f"    {col:>30}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"range=[{vals.min():.1f}, {vals.max():.1f}], missing={miss}")

    # Merge into epoch-level data (left join — all epochs kept)
    print(f"\nMerging into epoch data...")
    n_before = len(df.columns)
    df = df.merge(demo_subset, on="mesa_id", how="left")
    n_after = len(df.columns)
    print(f"  Columns: {n_before} → {n_after} (+{n_after - n_before} demographic features)")

    # Fill missing demographics with median (only 0.2-1.4% missing)
    demo_feature_cols = [c for c in demo_subset.columns if c != "mesa_id"]
    for col in demo_feature_cols:
        miss = df[col].isna().sum()
        if miss > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median).astype(np.float32)
            print(f"  Imputed {miss:,} NaN in {col} with median={median:.2f}")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    df.to_parquet(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"  Saved: {size_mb:.1f} MB, {len(df.columns)} columns, {len(df):,} epochs")
    print("Done!")


if __name__ == "__main__":
    main()
