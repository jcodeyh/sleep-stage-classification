#!/usr/bin/env python3
"""Build a clean wearable-feature dataset from DREAMT (PhysioNet, Empatica E4).

DREAMT is a wearable-only test domain — Empatica E4 PPG/EDA/temp/accel + 1-Hz HR
+ AASM-scored sleep stages from concurrent PSG. There is no SpO2 in the wearable
data; SpO2 columns are emitted as NaN so the schema matches MESA/STAGES and the
cross-domain evaluator can median-impute them at inference time.

The DREAMT 64Hz files contain only Empatica E4 wearable signals (the 100Hz files
also include PSG channels we do not need for the test set). Each row of the
64Hz file is a 1/64-sec sample; lower-rate channels (HR=1Hz, ACC=32Hz,
EDA/TEMP=4Hz) have repeated values in the intermediate rows.

Output schema matches build_stages_dataset.py / build_clean_dataset.py:
    subject_id, epoch_idx, clock_time,
    hr_mean, hr_std, hr_min, hr_max,
    hrv_mean_nn, hrv_sdnn, hrv_rmssd, hrv_pnn50,
    spo2_mean, spo2_std, spo2_min,           (all NaN in DREAMT)
    actig_activity, actig_log_activity, actig_wake, actig_white_light,
    tod_cos, tod_sin,                         (NaN — DREAMT timestamps start at 0)
    sleep_stage, valid_epoch, hr_source

Usage:
    python scripts/build_dreamt_dataset.py --data-dir ./datasets/dreamt
    python scripts/build_dreamt_dataset.py --data-dir ./datasets/dreamt --workers 8
    python scripts/build_dreamt_dataset.py --data-dir ./datasets/dreamt --subjects S002 S003
"""

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EPOCH_SEC = 30
SAMPLE_RATE_HZ = 64           # DREAMT 64Hz file row rate
SAMPLES_PER_EPOCH = EPOCH_SEC * SAMPLE_RATE_HZ  # 1920

# Sleep stage mapping for DREAMT
# DREAMT labels: P (Preparation), W (Wake), N1, N2, N3, R (REM), Missing
STAGE_MAP = {
    "W": 0, "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 5, "REM": 5,
    "P": 6, "Missing": 6,  # Preparation and Missing → unscored
}

# Physiological bounds (consistent with MESA/STAGES)
HR_MIN_BPM = 30.0
HR_MAX_BPM = 200.0

# Empatica E4 accelerometer is in 1/64 g (per DREAMT metadata).
# 1 g (gravity) = 64 in raw units. ENMO subtracts 1g to isolate dynamic motion.
ACC_G_UNIT = 64.0
ACC_RATE_HZ = 32              # E4 accelerometer native rate
ACC_DECIMATION = SAMPLE_RATE_HZ // ACC_RATE_HZ  # 2 — every other row is unique

HR_RATE_HZ = 1                # E4 HR native rate
HR_DECIMATION = SAMPLE_RATE_HZ // HR_RATE_HZ    # 64

# Activity threshold for actig_wake (matches Sadeh-style threshold; arbitrary
# scale, will be z-scored downstream — we only need a rough wake/sleep flag).
ACTIG_WAKE_THRESHOLD = 50.0

USECOLS = ["TIMESTAMP", "HR", "ACC_X", "ACC_Y", "ACC_Z", "Sleep_Stage"]


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: Path) -> list[str]:
    """Return list of DREAMT subject IDs (e.g. 'S002') with a 64Hz CSV file."""
    csv_dir = data_dir / "physionet.org" / "files" / "dreamt" / "2.1.0" / "data_64Hz"
    if not csv_dir.exists():
        logger.error("64Hz CSV dir not found: %s", csv_dir)
        return []

    subjects = []
    for f in sorted(csv_dir.glob("S*_whole_df.csv")):
        if f.stat().st_size > 0:
            sid = f.stem.replace("_whole_df", "")
            subjects.append(sid)
    return subjects


def csv_path_for(data_dir: Path, subject_id: str) -> Path:
    return (data_dir / "physionet.org" / "files" / "dreamt" / "2.1.0"
            / "data_64Hz" / f"{subject_id}_whole_df.csv")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_hr_features(hr_epoch: np.ndarray) -> dict:
    """HR mean/std/min/max from the 1-Hz HR samples in an epoch.

    The 64Hz row layout repeats each HR value 64 times; we deduplicate by
    taking every 64th sample so we work with the ~30 unique HR readings.
    """
    hr_unique = hr_epoch[::HR_DECIMATION]
    hr_valid = hr_unique[(hr_unique >= HR_MIN_BPM) & (hr_unique <= HR_MAX_BPM)
                         & np.isfinite(hr_unique)]
    if len(hr_valid) == 0:
        return {"hr_mean": np.nan, "hr_std": np.nan,
                "hr_min": np.nan, "hr_max": np.nan}
    return {
        "hr_mean": float(np.mean(hr_valid)),
        "hr_std": float(np.std(hr_valid)),
        "hr_min": float(np.min(hr_valid)),
        "hr_max": float(np.max(hr_valid)),
    }


def extract_hrv_features(hr_epoch: np.ndarray) -> dict:
    """Time-domain HRV from HR-derived RR intervals (matches MESA/STAGES approach).

    Converts unique 1-Hz HR samples to RR intervals (RR_ms = 60000 / HR_bpm)
    and computes mean_nn / sdnn / rmssd / pnn50. This is the same recipe
    extract_hrv_epoch uses in build_stages_dataset.py for the DHR channel.
    """
    result = {"hrv_mean_nn": np.nan, "hrv_sdnn": np.nan,
              "hrv_rmssd": np.nan, "hrv_pnn50": np.nan}

    hr_unique = hr_epoch[::HR_DECIMATION]
    hr_valid = hr_unique[(hr_unique >= HR_MIN_BPM) & (hr_unique <= HR_MAX_BPM)
                         & np.isfinite(hr_unique)]
    if len(hr_valid) < 5:
        return result

    rr_ms = 60000.0 / hr_valid
    result["hrv_mean_nn"] = float(np.mean(rr_ms))
    result["hrv_sdnn"] = float(np.std(rr_ms))

    rr_diff = np.diff(rr_ms)
    if len(rr_diff) >= 2:
        result["hrv_rmssd"] = float(np.sqrt(np.mean(rr_diff ** 2)))
        result["hrv_pnn50"] = float(np.mean(np.abs(rr_diff) > 50.0))

    return result


def extract_actig_features(acc_x: np.ndarray, acc_y: np.ndarray,
                           acc_z: np.ndarray) -> dict:
    """ENMO-based activity counts from 32-Hz triaxial accelerometer.

    Empatica E4 ACC is in 1/64 g; gravity ~64 raw. We compute per-sample
    Euclidean Norm Minus One (ENMO = max(|a| - 1g, 0)) which isolates
    dynamic motion, then sum across the epoch. Absolute scale differs from
    MESA actiwatch counts and STAGES Huami activeness, but downstream
    per-subject z-scoring (cross_domain_eval.py) normalizes this away.
    """
    # Decimate to native 32 Hz to avoid double-counting repeated rows.
    ax = acc_x[::ACC_DECIMATION].astype(np.float64)
    ay = acc_y[::ACC_DECIMATION].astype(np.float64)
    az = acc_z[::ACC_DECIMATION].astype(np.float64)
    finite = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    if finite.sum() < 10:
        return {"actig_activity": np.nan, "actig_log_activity": np.nan,
                "actig_wake": np.nan, "actig_white_light": np.nan}

    mag = np.sqrt(ax[finite] ** 2 + ay[finite] ** 2 + az[finite] ** 2)
    enmo = np.maximum(mag - ACC_G_UNIT, 0.0)
    activity = float(np.sum(enmo))
    return {
        "actig_activity": activity,
        "actig_log_activity": float(np.log1p(activity)),
        "actig_wake": 1.0 if activity > ACTIG_WAKE_THRESHOLD else 0.0,
        "actig_white_light": np.nan,  # E4 has no white-light sensor
    }


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(args: tuple) -> pd.DataFrame | None:
    """Process one DREAMT subject. (data_dir, subject_id) tuple for Pool.map."""
    data_dir, subject_id = args
    csv_path = csv_path_for(data_dir, subject_id)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        logger.warning("CSV missing or empty for %s", subject_id)
        return None

    try:
        df_raw = pd.read_csv(csv_path, usecols=USECOLS, low_memory=False)
    except Exception as e:
        logger.warning("CSV read failed for %s: %s", subject_id, e)
        return None

    n_rows = len(df_raw)
    if n_rows < SAMPLES_PER_EPOCH * 60:  # < 30 minutes of data
        logger.warning("Too few rows for %s (%d < %d)", subject_id,
                       n_rows, SAMPLES_PER_EPOCH * 60)
        return None

    n_epochs = n_rows // SAMPLES_PER_EPOCH

    hr = df_raw["HR"].to_numpy(dtype=np.float32, na_value=np.nan)
    ax = df_raw["ACC_X"].to_numpy(dtype=np.float32, na_value=np.nan)
    ay = df_raw["ACC_Y"].to_numpy(dtype=np.float32, na_value=np.nan)
    az = df_raw["ACC_Z"].to_numpy(dtype=np.float32, na_value=np.nan)
    stage_str = df_raw["Sleep_Stage"].astype(str).to_numpy()
    del df_raw

    rows = []
    for ep in range(n_epochs):
        start = ep * SAMPLES_PER_EPOCH
        end = start + SAMPLES_PER_EPOCH

        # Sleep stage: take the most common label in the epoch
        labels_in_epoch = stage_str[start:end]
        # Avoid scipy.stats.mode dependency — pandas value_counts is fine.
        stage_label = pd.Series(labels_in_epoch).mode()
        stage_str_val = stage_label.iloc[0] if len(stage_label) > 0 else "Missing"
        stage = STAGE_MAP.get(str(stage_str_val).strip(), 6)

        row = {"subject_id": subject_id, "epoch_idx": ep, "clock_time": ""}
        row.update(extract_hr_features(hr[start:end]))
        row.update(extract_hrv_features(hr[start:end]))
        row.update(extract_actig_features(ax[start:end], ay[start:end], az[start:end]))

        # SpO2: not collected by Empatica E4 — set NaN (cross_domain_eval
        # imputes with train-set median at inference time).
        row["spo2_mean"] = np.nan
        row["spo2_std"] = np.nan
        row["spo2_min"] = np.nan

        # DREAMT timestamps start at 0 (no wall clock available), so we
        # cannot anchor circadian features. Leave them NaN; the median
        # imputation in cross_domain_eval will fill them.
        row["tod_cos"] = np.nan
        row["tod_sin"] = np.nan

        hr_ok = not np.isnan(row["hr_mean"])
        stage_scored = stage in (0, 1, 2, 3, 5)
        row["valid_epoch"] = bool(hr_ok and stage_scored)
        row["sleep_stage"] = int(stage)
        row["hr_source"] = "wearable_hr_1hz"  # Empatica E4 1-Hz HR estimate

        rows.append(row)

    df = pd.DataFrame(rows)

    n_valid = int(df["valid_epoch"].sum())
    hr_nan = float(df["hr_mean"].isna().mean() * 100)
    logger.info(
        "Subject %s: %d epochs, %d valid (%.0f%%), HR NaN: %.1f%%",
        subject_id, len(df), n_valid,
        n_valid / len(df) * 100 if len(df) > 0 else 0,
        hr_nan,
    )
    return df


# ---------------------------------------------------------------------------
# Metadata summary
# ---------------------------------------------------------------------------

def compute_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = df.groupby("subject_id").agg(
        n_epochs=("epoch_idx", "count"),
        n_valid=("valid_epoch", "sum"),
        pct_valid=("valid_epoch", "mean"),
        hr_mean=("hr_mean", "mean"),
        hr_nan_rate=("hr_mean", lambda x: x.isna().mean()),
        actig_mean=("actig_activity", "mean"),
        n_wake=("sleep_stage", lambda x: (x == 0).sum()),
        n_n1=("sleep_stage", lambda x: (x == 1).sum()),
        n_n2=("sleep_stage", lambda x: (x == 2).sum()),
        n_n3=("sleep_stage", lambda x: (x == 3).sum()),
        n_rem=("sleep_stage", lambda x: (x == 5).sum()),
        n_unscored=("sleep_stage", lambda x: (x == 6).sum()),
    ).reset_index()
    meta["pct_valid"] = (meta["pct_valid"] * 100).round(1)
    meta["hr_nan_rate"] = (meta["hr_nan_rate"] * 100).round(1)
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build wearable-feature parquet from DREAMT 64Hz CSVs"
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="DREAMT data dir (contains physionet.org/files/dreamt/...)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir (default: same as --data-dir)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Process only these subject IDs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just count subjects, don't process")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    subjects = args.subjects or discover_subjects(args.data_dir)

    if not subjects:
        logger.error("No DREAMT subjects found. Did the 64Hz CSVs download?")
        sys.exit(1)

    logger.info("Processing %d DREAMT subjects", len(subjects))

    if args.dry_run:
        for s in subjects[:20]:
            print(f"  {s}")
        if len(subjects) > 20:
            print(f"  ... and {len(subjects) - 20} more")
        return

    t0 = time.time()
    task_args = [(args.data_dir, s) for s in subjects]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(process_subject, task_args)
    else:
        results = [process_subject(a) for a in task_args]

    dfs = [r for r in results if r is not None]
    if not dfs:
        logger.error("No subjects processed successfully")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    elapsed = time.time() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "dreamt_clean_wearable.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    meta = compute_metadata(df)
    meta_path = output_dir / "dreamt_clean_metadata.csv"
    meta.to_csv(meta_path, index=False)

    n_subjects = df["subject_id"].nunique()
    n_valid = int(df["valid_epoch"].sum())
    logger.info(
        "Done in %.1f min. %d subjects, %d epochs, %d valid (%.1f%%)",
        elapsed / 60, n_subjects, len(df), n_valid,
        n_valid / len(df) * 100 if len(df) > 0 else 0,
    )
    logger.info("Parquet saved to: %s (%.1f MB)", parquet_path,
                parquet_path.stat().st_size / 1e6)
    logger.info("Metadata saved to: %s", meta_path)

    scored = df[df["sleep_stage"].isin([0, 1, 2, 3, 5])]
    if len(scored) > 0:
        dist = scored["sleep_stage"].value_counts().sort_index()
        stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}
        logger.info("Stage distribution:")
        for stage, count in dist.items():
            pct = count / len(scored) * 100
            logger.info("  %s: %d (%.1f%%)", stage_names.get(stage, f"?{stage}"), count, pct)


if __name__ == "__main__":
    main()
