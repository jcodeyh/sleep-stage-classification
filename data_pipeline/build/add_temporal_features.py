#!/usr/bin/env python3
"""
Add temporal features to a clean wearable dataset.

Computes within each subject:
  - Rolling means/stds over 5 and 10 epoch windows for key signals
  - Delta features (change from previous epoch)
  - Time since last wake epoch
  - Epoch position within the night (0-1 normalized)

Works with both MESA and STAGES datasets via CLI arguments.

Usage:
    # MESA (default):
    python scripts/add_temporal_features.py

    # STAGES:
    python scripts/add_temporal_features.py \
        --input datasets/stages/stages_clean_wearable.parquet \
        --output datasets/stages/stages_clean_wearable_temporal.parquet \
        --id-col subject_id
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "mesa" / "mesa_clean_wearable.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "mesa" / "mesa_clean_wearable_temporal.parquet"
DEFAULT_ID_COL = "mesa_id"

# Key signals to compute rolling/delta features on (wearable-collectible only)
ROLLING_COLS = ["hr_mean", "hrv_rmssd", "spo2_mean", "actig_activity"]
WINDOWS = [5, 10]  # in epochs (2.5 min, 5 min)


def add_temporal_features(df: pd.DataFrame, id_col: str = "mesa_id") -> pd.DataFrame:
    """Add all temporal features, computed per-subject."""
    print("Adding temporal features...")

    # Sort by subject and epoch to ensure correct temporal ordering
    df = df.sort_values([id_col, "epoch_idx"]).reset_index(drop=True)

    new_cols = {}

    # ── Rolling features (per subject) ────────────────────────────────────
    # Only compute rolling features for columns that exist in this dataset
    rolling_cols = [c for c in ROLLING_COLS if c in df.columns]
    print(f"  Computing rolling statistics for {len(rolling_cols)} signals...")
    for col in rolling_cols:
        for w in WINDOWS:
            roll = df.groupby(id_col)[col].transform(
                lambda x: x.rolling(w, min_periods=1, center=False).mean()
            )
            new_cols[f"{col}_roll{w}_mean"] = roll

            roll_std = df.groupby(id_col)[col].transform(
                lambda x: x.rolling(w, min_periods=1, center=False).std()
            )
            new_cols[f"{col}_roll{w}_std"] = roll_std

    # ── Delta features (change from previous epoch, per subject) ──────────
    print("  Computing delta features...")
    for col in rolling_cols:
        delta = df.groupby(id_col)[col].transform(lambda x: x.diff())
        new_cols[f"{col}_delta"] = delta

    # ── Time since last wake ──────────────────────────────────────────────
    print("  Computing time since last wake...")
    def time_since_wake(group):
        is_wake = (group["sleep_stage"] == 0).values
        result = np.zeros(len(group), dtype=np.float32)
        counter = 0
        for i in range(len(group)):
            if is_wake[i]:
                counter = 0
            else:
                counter += 1
            result[i] = counter
        return pd.Series(result, index=group.index)

    new_cols["epochs_since_wake"] = df.groupby(id_col).apply(
        time_since_wake, include_groups=False
    ).reset_index(level=0, drop=True)

    # ── Epoch position within night (0.0 = start, 1.0 = end) ─────────────
    print("  Computing night position...")
    new_cols["night_position"] = df.groupby(id_col)["epoch_idx"].transform(
        lambda x: (x - x.min()) / max(x.max() - x.min(), 1)
    )

    # ── Sleep pressure proxy (cumulative non-wake epochs / total so far) ──
    print("  Computing sleep pressure proxy...")
    def sleep_pressure(group):
        is_sleep = (group["sleep_stage"] != 0).astype(float).values
        cumsum = np.cumsum(is_sleep)
        indices = np.arange(1, len(group) + 1, dtype=np.float32)
        return pd.Series(cumsum / indices, index=group.index, dtype=np.float32)

    new_cols["sleep_pressure"] = df.groupby(id_col).apply(
        sleep_pressure, include_groups=False
    ).reset_index(level=0, drop=True)

    # ── Assemble ──────────────────────────────────────────────────────────
    for name, series in new_cols.items():
        df[name] = series.astype(np.float32)

    n_new = len(new_cols)
    print(f"  Added {n_new} temporal features. Total columns: {len(df.columns)}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add temporal features to a clean wearable dataset"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Input parquet file (default: MESA clean wearable)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output parquet file (default: MESA temporal)",
    )
    parser.add_argument(
        "--id-col", type=str, default=DEFAULT_ID_COL,
        help="Subject ID column name (default: mesa_id)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    id_col = args.id_col
    print(f"  {len(df):,} epochs, {df[id_col].nunique()} subjects, {len(df.columns)} columns")

    df = add_temporal_features(df, id_col=id_col)

    # Quick stats on new features
    new_feature_cols = [c for c in df.columns if any(
        c.endswith(s) for s in ["_roll5_mean", "_roll5_std", "_roll10_mean", "_roll10_std",
                                 "_delta", "_since_wake", "_position", "_pressure"]
    ) or c in ["epochs_since_wake", "night_position", "sleep_pressure"]]

    print(f"\nNew temporal features ({len(new_feature_cols)}):")
    for col in sorted(new_feature_cols):
        vals = df[col].dropna()
        print(f"  {col:>35}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"nan={df[col].isna().sum():,}")

    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"  Saved: {size_mb:.1f} MB, {len(df.columns)} columns")
    print("Done!")


if __name__ == "__main__":
    main()
