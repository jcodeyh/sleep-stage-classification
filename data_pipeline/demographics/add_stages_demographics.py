#!/usr/bin/env python3
"""Add demographics and clinical features to the STAGES clean wearable dataset.

Merges harmonized demographics (age, sex, BMI, race, smoking) and clinical
variables (ESS, ISI, RLS) from STAGES questionnaire CSVs into the per-epoch
parquet file. Produces a "full" parquet with demographic columns appended.

Usage:
    python scripts/add_stages_demographics.py
    python scripts/add_stages_demographics.py --input datasets/stages/stages_clean_wearable.parquet
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


def load_harmonized_demographics(datasets_dir: Path) -> pd.DataFrame:
    """Load NSRR harmonized demographics CSV."""
    harm_path = datasets_dir / "datasets" / "stages-harmonized-dataset-0.3.0.csv"
    if not harm_path.exists():
        logger.warning("Harmonized demographics not found at %s", harm_path)
        return pd.DataFrame()

    df = pd.read_csv(harm_path)
    logger.info("Loaded harmonized demographics: %d rows, columns: %s",
                len(df), list(df.columns))
    return df


def load_clinical_variables(datasets_dir: Path) -> pd.DataFrame:
    """Load clinical variables from the main STAGES dataset CSV."""
    main_path = datasets_dir / "datasets" / "stages-dataset-0.3.0.csv"
    if not main_path.exists():
        logger.warning("Main dataset CSV not found at %s", main_path)
        return pd.DataFrame()

    # Only load the columns we need
    cols = [
        "subject_code",
        "ess_0900",     # Epworth Sleepiness Scale total
        "isi_score",    # Insomnia Severity Index total
        "rls_probability",  # RLS probability category
        "pap_0100",     # PAP/CPAP usage
    ]
    try:
        df = pd.read_csv(main_path, usecols=cols)
    except ValueError:
        # Some columns might not exist in all versions
        available = pd.read_csv(main_path, nrows=0).columns
        cols = [c for c in cols if c in available]
        df = pd.read_csv(main_path, usecols=cols)

    logger.info("Loaded clinical variables: %d rows", len(df))
    return df


def build_demographics(datasets_dir: Path) -> pd.DataFrame:
    """Build a per-subject demographics table from STAGES data.

    Output columns (matching MESA schema where possible):
        demo_age        - Age in years
        demo_is_male    - 1=male, 0=female
        demo_bmi        - BMI kg/m²
        demo_race_*     - One-hot race columns
        demo_smoking    - 0=never, 1=former, 2=current
        demo_epworth    - Epworth Sleepiness Scale (0-24)
        demo_insomnia   - Insomnia Severity Index (0-28)
        demo_rls        - RLS probability (0=unlikely, 1=possible, 2=likely)
        demo_cpap       - CPAP use (0=no, 1=yes)
    """
    harm = load_harmonized_demographics(datasets_dir)
    clinical = load_clinical_variables(datasets_dir)

    if harm.empty:
        logger.error("Cannot build demographics without harmonized CSV")
        return pd.DataFrame()

    # Deduplicate by subject — keep first visit
    harm = harm.drop_duplicates(subset="subject_code", keep="first").reset_index(drop=True)
    demo = pd.DataFrame({"subject_id": harm["subject_code"]})

    # Age
    demo["demo_age"] = harm["nsrr_age"].astype(float)

    # Sex: 1=male, 0=female (matches MESA demo_is_male encoding)
    sex_map = {"male": 1, "female": 0}
    demo["demo_is_male"] = harm["nsrr_sex"].str.lower().map(sex_map).astype("Int64")

    # BMI
    demo["demo_bmi"] = harm["nsrr_bmi"].astype(float)

    # Race — one-hot encode
    race_categories = ["white", "black or african american", "asian",
                       "american indian or alaska native",
                       "native hawaiian or other pacific islander"]
    race = harm["nsrr_race"].str.lower().fillna("unknown")
    for cat in race_categories:
        col_name = "demo_race_" + cat.replace(" ", "_").replace("/", "_")[:20]
        demo[col_name] = (race == cat).astype(int)
    demo["demo_race_other"] = (~race.isin(race_categories + ["unknown"])).astype(int)

    # Smoking: 0=never, 1=former, 2=current
    smoking = pd.Series(np.nan, index=demo.index)
    if "nsrr_current_smoker" in harm.columns:
        current = harm["nsrr_current_smoker"].str.lower()
        smoking = smoking.where(current == "not reported", 0)
        smoking[current == "no"] = 0
        smoking[current == "yes"] = 2
    if "nsrr_ever_smoker" in harm.columns:
        ever = harm["nsrr_ever_smoker"].str.lower()
        # Former = ever_smoker=yes but current_smoker=no
        smoking[(ever == "yes") & (smoking == 0)] = 1
    demo["demo_smoking"] = smoking

    # Merge clinical variables
    if not clinical.empty:
        clinical = clinical.rename(columns={"subject_code": "subject_id"})
        clinical = clinical.drop_duplicates(subset="subject_id", keep="first")

        if "ess_0900" in clinical.columns:
            demo = demo.merge(
                clinical[["subject_id", "ess_0900"]].rename(
                    columns={"ess_0900": "demo_epworth"}),
                on="subject_id", how="left",
            )

        if "isi_score" in clinical.columns:
            demo = demo.merge(
                clinical[["subject_id", "isi_score"]].rename(
                    columns={"isi_score": "demo_insomnia"}),
                on="subject_id", how="left",
            )

        if "rls_probability" in clinical.columns:
            rls_map = {"unlikely": 0, "possible": 1, "likely": 2, "definite": 2}
            rls = clinical.set_index("subject_id")["rls_probability"].str.lower().map(rls_map)
            demo = demo.merge(
                rls.rename("demo_rls").reset_index(),
                on="subject_id", how="left",
            )

        if "pap_0100" in clinical.columns:
            cpap = clinical.set_index("subject_id")["pap_0100"]
            demo = demo.merge(
                cpap.rename("demo_cpap").reset_index(),
                on="subject_id", how="left",
            )

    logger.info("Built demographics for %d subjects", len(demo))
    logger.info("  Columns: %s", list(demo.columns))
    logger.info("  Non-null rates:")
    for col in demo.columns:
        if col == "subject_id":
            continue
        nn = demo[col].notna().sum()
        logger.info("    %s: %d/%d (%.0f%%)", col, nn, len(demo), 100 * nn / len(demo))

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Add demographics to STAGES clean wearable dataset"
    )
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "datasets" / "stages" / "stages_clean_wearable.parquet",
        help="Input parquet file",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "datasets" / "stages" / "stages_clean_wearable_full.parquet",
        help="Output parquet file with demographics",
    )
    parser.add_argument(
        "--stages-dir", type=Path,
        default=PROJECT_ROOT / "datasets" / "stages",
        help="STAGES dataset directory (contains datasets/ subdir with CSVs)",
    )
    args = parser.parse_args()

    # Load epoch data
    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} epochs, {df.subject_id.nunique()} subjects")

    # Build demographics
    demo = build_demographics(args.stages_dir)
    if demo.empty:
        print("ERROR: Could not build demographics. Saving without them.")
        df.to_parquet(args.output, index=False)
        return

    # Merge demographics into epoch data (broadcast per subject)
    n_before = len(df)
    df = df.merge(demo, on="subject_id", how="left")
    assert len(df) == n_before, "Merge changed row count!"

    # Report
    demo_cols = [c for c in df.columns if c.startswith("demo_")]
    print(f"\nAdded {len(demo_cols)} demographic columns:")
    for col in demo_cols:
        nn = df[col].notna().sum()
        print(f"  {col:<30} non-null: {nn:,}/{len(df):,} ({100 * nn / len(df):.0f}%)")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nSaved: {args.output} ({size_mb:.1f} MB, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
