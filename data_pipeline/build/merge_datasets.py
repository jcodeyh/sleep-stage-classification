#!/usr/bin/env python3
"""Merge MESA and STAGES datasets into a unified multi-cohort training set.

Combines the two parquet files into a single dataset with consistent column
names, normalized subject IDs, and dataset/site provenance columns.

Usage:
    python scripts/merge_datasets.py
    python scripts/merge_datasets.py --mesa-input ... --stages-input ... --output ...
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

# STAGES site code → human-readable site name
SITE_MAP = {
    "BOGN": "bogan_sleep",
    "GSBB": "geisinger_bb",
    "GSDV": "geisinger_dv",
    "GSLH": "geisinger_lh",
    "GSSA": "geisinger_sa",
    "GSSW": "geisinger_sw",
    "MSMI": "medsleep_mi",
    "MSNF": "medsleep_nf",
    "MSQW": "medsleep_qw",
    "MSTH": "medsleep_th",
    "MSTR": "medsleep_tr",
    "STLK": "st_lukes",
    "STNF": "stanford",
}


def extract_site(subject_id: str) -> str:
    """Extract site from STAGES subject ID (e.g., 'STNF00001' → 'stanford')."""
    for code, name in SITE_MAP.items():
        if subject_id.startswith(code):
            return name
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Merge MESA and STAGES into unified training dataset"
    )
    parser.add_argument(
        "--mesa-input", type=Path,
        default=PROJECT_ROOT / "datasets" / "mesa" / "mesa_clean_wearable_full.parquet",
        help="MESA full parquet",
    )
    parser.add_argument(
        "--stages-input", type=Path,
        default=PROJECT_ROOT / "datasets" / "stages" / "stages_clean_wearable_full.parquet",
        help="STAGES full parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "datasets" / "combined",
        help="Output directory",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    dfs = {}
    for name, path in [("mesa", args.mesa_input), ("stages", args.stages_input)]:
        if path.exists():
            df = pd.read_parquet(path)
            logger.info("Loaded %s: %d epochs, %d subjects, %d columns",
                        name, len(df), df.iloc[:, 0].nunique() if len(df) > 0 else 0,
                        len(df.columns))
            dfs[name] = df
        else:
            logger.warning("File not found, skipping: %s", path)

    if not dfs:
        logger.error("No datasets loaded!")
        return

    # Normalize subject IDs
    if "mesa" in dfs:
        mesa = dfs["mesa"].copy()
        # Rename mesa_id → subject_id if needed
        if "mesa_id" in mesa.columns and "subject_id" not in mesa.columns:
            mesa["subject_id"] = mesa["mesa_id"].astype(str).str.zfill(4).apply(
                lambda x: f"mesa_{x}"
            )
            mesa = mesa.drop(columns=["mesa_id"])
        elif "subject_id" in mesa.columns:
            mesa["subject_id"] = mesa["subject_id"].astype(str).apply(
                lambda x: f"mesa_{x}" if not x.startswith("mesa_") else x
            )
        mesa["dataset"] = "mesa"
        mesa["site"] = "mesa"
        mesa["is_clinical_population"] = False
        dfs["mesa"] = mesa

    if "stages" in dfs:
        stages = dfs["stages"].copy()
        stages["subject_id"] = stages["subject_id"].astype(str).apply(
            lambda x: f"stages_{x}" if not x.startswith("stages_") else x
        )
        stages["dataset"] = "stages"
        stages["site"] = stages["subject_id"].str.replace("stages_", "").apply(extract_site)
        stages["is_clinical_population"] = True
        dfs["stages"] = stages

    # Find common columns (wearable features + metadata)
    all_columns = set()
    for df in dfs.values():
        all_columns |= set(df.columns)

    # Align columns — pad missing with NaN
    for name in dfs:
        for col in all_columns:
            if col not in dfs[name].columns:
                dfs[name][col] = np.nan
                logger.debug("Added missing column '%s' to %s", col, name)

    # Concatenate
    combined = pd.concat(list(dfs.values()), ignore_index=True)

    # ── Site quality tiers ────────────────────────────────────────────────────
    # Tier 1 (>60% valid): MESA, STNF, MSTR
    # Tier 2 (15-60% valid): BOGN-ECG, MSQW, MSMI
    # Tier 3 (<15% valid): all Geisinger sites, MSNF, BOGN-PPG
    TIER1_SITES = {"mesa", "stanford", "medsleep_tr"}
    TIER2_SITES = {"medsleep_qw", "medsleep_mi"}
    TIER3_SITES = {"geisinger_dv", "geisinger_sa", "geisinger_bb",
                   "geisinger_lh", "geisinger_sw", "medsleep_nf"}
    # BOGN (bogan_sleep) is split by hr_source: ecg_derived→Tier 2, ppg_derived→Tier 3

    def _assign_tier(row):
        site = row["site"]
        if site in TIER1_SITES:
            return 1
        if site in TIER2_SITES:
            return 2
        if site in TIER3_SITES:
            return 3
        if site == "bogan_sleep":
            return 2 if row.get("hr_source") == "ecg_derived" else 3
        return 3  # unknown sites default to lowest tier

    combined["site_tier"] = combined.apply(_assign_tier, axis=1).astype(np.int8)
    combined["is_high_quality_site"] = combined["site_tier"] == 1

    tier_summary = combined.groupby("site_tier")["subject_id"].nunique()
    for tier, n_subj in tier_summary.items():
        logger.info("Tier %d: %d subjects", tier, n_subj)

    # Sort columns: identifiers first, then features, then demographics
    id_cols = ["subject_id", "dataset", "site", "site_tier",
               "is_high_quality_site", "is_clinical_population",
               "epoch_idx", "clock_time"]
    feature_cols = sorted([c for c in combined.columns
                          if c not in id_cols and not c.startswith("demo_")
                          and c not in ("sleep_stage", "valid_epoch")])
    demo_cols = sorted([c for c in combined.columns if c.startswith("demo_")])
    label_cols = ["sleep_stage", "valid_epoch"]
    ordered = [c for c in id_cols if c in combined.columns]
    ordered += [c for c in feature_cols if c in combined.columns]
    ordered += demo_cols
    ordered += [c for c in label_cols if c in combined.columns]
    # Add any remaining
    remaining = [c for c in combined.columns if c not in ordered]
    ordered += remaining
    combined = combined[ordered]

    # Report
    n_mesa = combined[combined.dataset == "mesa"].subject_id.nunique() if "mesa" in dfs else 0
    n_stages = combined[combined.dataset == "stages"].subject_id.nunique() if "stages" in dfs else 0

    print(f"\n{'=' * 60}")
    print(f"COMBINED DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total epochs:    {len(combined):,}")
    print(f"Total subjects:  {combined.subject_id.nunique()}")
    print(f"  MESA:          {n_mesa}")
    print(f"  STAGES:        {n_stages}")
    print(f"Valid epochs:    {combined.valid_epoch.sum():,} "
          f"({100 * combined.valid_epoch.mean():.1f}%)")
    print(f"Columns:         {len(combined.columns)}")
    print()

    # Stage distribution by dataset
    for ds in sorted(dfs.keys()):
        subset = combined[combined.dataset == ds]
        valid = subset[subset.valid_epoch]
        stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}
        print(f"  {ds.upper()} stage distribution (valid epochs only):")
        for stage, name in stage_names.items():
            count = (valid.sleep_stage == stage).sum()
            pct = 100 * count / len(valid) if len(valid) > 0 else 0
            print(f"    {name:<6}: {count:>8,} ({pct:>5.1f}%)")
        print()

    # Sites
    if "stages" in dfs:
        print("  STAGES sites:")
        sites = combined[combined.dataset == "stages"].groupby("site").subject_id.nunique()
        for site, n in sites.sort_values(ascending=False).items():
            print(f"    {site:<20}: {n:>4} subjects")
        print()

    # Save
    out_path = args.output_dir / "combined_clean_wearable_full.parquet"
    combined.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

    # Save metadata CSV
    meta_path = args.output_dir / "combined_metadata.csv"
    meta = combined.groupby(["subject_id", "dataset", "site"]).agg(
        n_epochs=("epoch_idx", "count"),
        n_valid=("valid_epoch", "sum"),
        pct_valid=("valid_epoch", "mean"),
    ).reset_index()
    meta["pct_valid"] = (meta["pct_valid"] * 100).round(1)
    meta.to_csv(meta_path, index=False)
    print(f"Metadata: {meta_path}")

    # Save merge report
    report_lines = [
        f"Combined Dataset Merge Report",
        f"{'=' * 60}",
        f"MESA subjects:   {n_mesa}",
        f"STAGES subjects:  {n_stages}",
        f"Total epochs:     {len(combined):,}",
        f"Valid epochs:     {combined.valid_epoch.sum():,}",
        f"Columns:          {len(combined.columns)}",
        f"",
        f"Feature columns: {feature_cols}",
        f"Demographic columns: {demo_cols}",
    ]
    report_path = args.output_dir / "merge_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
