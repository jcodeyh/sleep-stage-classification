#!/usr/bin/env python3
"""Build a clean wearable-feature dataset from MESA PSG + actigraphy data.

Produces a single Parquet file with one row per 30-second epoch per subject,
containing only features collectible by consumer wearables (HR, HRV, SpO2,
movement, body position, actigraphy, circadian time) plus expert-scored
sleep stage labels.

Usage:
    python build_clean_dataset.py --data-dir ./datasets/mesa --output-dir ./datasets/mesa
    python build_clean_dataset.py --data-dir ./datasets/mesa --dry-run
    python build_clean_dataset.py --data-dir ./datasets/mesa --subjects 0001 0002
"""

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyedflib
from lxml import etree
from scipy.signal import butter, filtfilt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EDF channel names (canonical)
CHAN_HR = "HR"
CHAN_DHR = "DHR"
CHAN_SPO2 = "SpO2"
CHAN_OXSTATUS = "OxStatus"
CHAN_POS = "Pos"
CHAN_LEG = "Leg"

NEEDED_CHANNELS = [CHAN_HR, CHAN_DHR, CHAN_SPO2, CHAN_OXSTATUS, CHAN_POS, CHAN_LEG]

# Channel label variants -> canonical name
CHANNEL_LABEL_MAP = {
    "HR": CHAN_HR, "hr": CHAN_HR, "Heart Rate": CHAN_HR, "Pulse": CHAN_HR,
    "DHR": CHAN_DHR, "dhr": CHAN_DHR,
    "SpO2": CHAN_SPO2, "spo2": CHAN_SPO2, "SaO2": CHAN_SPO2, "SAO2": CHAN_SPO2, "sao2": CHAN_SPO2,
    "OxStatus": CHAN_OXSTATUS, "oxstatus": CHAN_OXSTATUS, "OXSTATUS": CHAN_OXSTATUS,
    "Pos": CHAN_POS, "pos": CHAN_POS, "Position": CHAN_POS,
    "Leg": CHAN_LEG, "leg": CHAN_LEG, "Leg EMG": CHAN_LEG,
}

# Sleep stage mapping from NSRR annotation XML
STAGE_MAP = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "Stage 4 sleep|4": 3,  # Merge N3/N4 per AASM
    "REM sleep|5": 5,
}

# Physiological bounds
DHR_MIN_BPM = 30.0
DHR_MAX_BPM = 200.0
RR_MIN_S = 60.0 / DHR_MAX_BPM  # 0.3s
RR_MAX_S = 60.0 / DHR_MIN_BPM  # 2.0s
SPO2_MIN_VALID = 50.0
SPO2_MAX_VALID = 100.0

# HRV frequency bands (Hz)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.4)

# Epoch duration
EPOCH_SEC = 30

# Movement threshold (fraction of max absolute amplitude)
LEG_ACTIVE_THRESHOLD_FRAC = 0.1

# Body position labels (MESA convention)
POS_LABELS = ["supine", "left", "right", "prone", "upright"]


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: Path) -> list[str]:
    """Find subject IDs that have EDF + annotation XML + actigraphy CSV."""
    edf_dir = data_dir / "polysomnography" / "edfs"
    annot_dir = data_dir / "polysomnography" / "annotations-events-nsrr"
    actig_dir = data_dir / "actigraphy"

    def extract_ids(directory: Path, pattern: str, strip_suffix: str) -> set[str]:
        ids = set()
        for f in directory.glob(pattern):
            name = f.stem
            # Extract 4-digit ID from filename like mesa-sleep-0001 or mesa-sleep-0001-nsrr
            parts = name.split("-")
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    ids.add(part)
                    break
        return ids

    edf_ids = extract_ids(edf_dir, "mesa-sleep-*.edf", ".edf")
    annot_ids = extract_ids(annot_dir, "mesa-sleep-*-nsrr.xml", "-nsrr.xml")
    actig_ids = extract_ids(actig_dir, "mesa-sleep-*.csv", ".csv")

    common = sorted(edf_ids & annot_ids & actig_ids)
    logger.info(
        "Discovered subjects: %d EDF, %d annotations, %d actigraphy, %d common",
        len(edf_ids), len(annot_ids), len(actig_ids), len(common),
    )
    return common


def resolve_paths(data_dir: Path, subject_id: str) -> tuple[Path, Path, Path]:
    """Return (edf_path, xml_path, csv_path) for a subject."""
    edf_path = data_dir / "polysomnography" / "edfs" / f"mesa-sleep-{subject_id}.edf"
    xml_path = (
        data_dir / "polysomnography" / "annotations-events-nsrr"
        / f"mesa-sleep-{subject_id}-nsrr.xml"
    )
    csv_path = data_dir / "actigraphy" / f"mesa-sleep-{subject_id}.csv"
    return edf_path, xml_path, csv_path


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------

def parse_sleep_stages(xml_path: Path) -> np.ndarray:
    """Parse sleep stage labels from NSRR annotation XML.

    Returns:
        Array of integer stage labels, one per 30-second epoch.
        Unscored epochs are labeled 6.
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    epochs = {}
    for event in root.iter():
        tag = event.tag.split("}")[-1] if "}" in event.tag else event.tag
        if tag != "ScoredEvent":
            continue

        event_data = {}
        for child in event:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            event_data[child_tag] = child.text

        event_type = event_data.get("EventType") or ""
        if "Stages" not in event_type:
            continue

        event_concept = event_data.get("EventConcept", "")
        start_sec = float(event_data.get("Start", 0))
        duration_sec = float(event_data.get("Duration", 30))

        stage_label = STAGE_MAP.get(event_concept)
        if stage_label is None:
            for key, val in STAGE_MAP.items():
                if key.split("|")[0] in event_concept:
                    stage_label = val
                    break
        if stage_label is None:
            stage_label = 6  # unscored

        epoch_start = int(start_sec / EPOCH_SEC)
        n_epochs = max(1, int(duration_sec / EPOCH_SEC))
        for e in range(epoch_start, epoch_start + n_epochs):
            epochs[e] = stage_label

    if not epochs:
        raise ValueError(f"No sleep stage events in {xml_path.name}")

    max_epoch = max(epochs.keys()) + 1
    labels = np.full(max_epoch, 6, dtype=np.int32)
    for idx, stage in epochs.items():
        labels[idx] = stage

    return labels


# ---------------------------------------------------------------------------
# Actigraphy alignment
# ---------------------------------------------------------------------------

ACTIG_MAX_OFFSET_MIN = 30  # Max acceptable actigraphy-PSG alignment offset (minutes)


def align_actigraphy(
    edf_start_sec: float, csv_path: Path, n_psg_epochs: int
) -> pd.DataFrame | None:
    """Align actigraphy CSV to PSG epochs using timestamp matching.

    Args:
        edf_start_sec: PSG start time in seconds since midnight.
        csv_path: Path to actigraphy CSV.
        n_psg_epochs: Number of 30-second PSG epochs.

    Returns:
        DataFrame with one row per PSG epoch containing actigraphy columns,
        or None if alignment fails or offset exceeds ACTIG_MAX_OFFSET_MIN.
    """
    df = pd.read_csv(csv_path)

    # Filter to first PSG night (daybymidnight == 1)
    day1 = df[df["daybymidnight"] == 1].copy()
    if day1.empty:
        # Fallback: use the first day available
        min_day = df["daybymidnight"].min()
        day1 = df[df["daybymidnight"] == min_day].copy()

    if day1.empty:
        return None

    # Parse linetime to seconds since midnight
    def time_to_sec(t: str) -> float:
        parts = t.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return h * 3600 + m * 60 + s

    day1_times = day1["linetime"].apply(time_to_sec).values

    # Find the actigraphy epoch closest to PSG start time
    # Handle midnight wrap: if PSG starts in evening and actigraphy wraps past midnight
    edf_start_mod = edf_start_sec % 86400
    diffs = np.abs(day1_times - edf_start_mod)
    # Also check midnight-wrapped difference
    diffs_wrap = np.minimum(diffs, 86400 - diffs)
    best_idx = np.argmin(diffs_wrap)

    # Check alignment quality
    offset_min = float(diffs_wrap[best_idx]) / 60.0
    subject_hint = csv_path.stem  # e.g. "mesa-sleep-0001"
    logger.info(
        "%s: actigraphy alignment offset = %.1f min", subject_hint, offset_min,
    )
    if offset_min > ACTIG_MAX_OFFSET_MIN:
        logger.warning(
            "%s: actigraphy offset %.1f min exceeds %d min threshold, "
            "discarding actigraphy (subject kept with NaN actigraphy)",
            subject_hint, offset_min, ACTIG_MAX_OFFSET_MIN,
        )
        return None

    # Now extend from this point through the full PSG recording
    # We also need epochs that cross midnight into the next day
    actig_start_line = day1.index[best_idx]

    # Get all actigraphy from the start line through n_psg_epochs
    actig_slice = df.loc[actig_start_line : actig_start_line + n_psg_epochs - 1].copy()
    actig_slice = actig_slice.reset_index(drop=True)

    # Pad with NaN if actigraphy is shorter than PSG
    if len(actig_slice) < n_psg_epochs:
        pad = pd.DataFrame(
            np.nan,
            index=range(len(actig_slice), n_psg_epochs),
            columns=actig_slice.columns,
        )
        actig_slice = pd.concat([actig_slice, pad], ignore_index=True)

    return actig_slice.head(n_psg_epochs)


# ---------------------------------------------------------------------------
# HRV feature extraction from DHR channel
# ---------------------------------------------------------------------------

def extract_hrv_epoch(dhr_epoch: np.ndarray, sample_rate: int = 256) -> dict:
    """Compute HRV features from a single 30-second DHR epoch.

    The DHR channel provides instantaneous heart rate at every sample.
    We convert nonzero values to RR intervals and compute standard HRV metrics.

    Returns dict with keys: hrv_mean_nn, hrv_sdnn, hrv_rmssd, hrv_pnn50, hrv_lf_hf
    """
    result = {
        "hrv_mean_nn": np.nan,
        "hrv_sdnn": np.nan,
        "hrv_rmssd": np.nan,
        "hrv_pnn50": np.nan,
        "hrv_lf_hf": np.nan,
    }

    # Filter to physiological range
    valid_mask = (dhr_epoch >= DHR_MIN_BPM) & (dhr_epoch <= DHR_MAX_BPM)
    dhr_valid = dhr_epoch[valid_mask]

    if len(dhr_valid) < sample_rate * 5:  # Need at least 5 seconds of valid data
        return result

    # Convert HR (bpm) to RR intervals (ms)
    rr_ms = 60000.0 / dhr_valid  # milliseconds

    result["hrv_mean_nn"] = float(np.mean(rr_ms))
    result["hrv_sdnn"] = float(np.std(rr_ms))

    # For RMSSD and pNN50, we need successive differences
    # Subsample to approximate beat-to-beat intervals
    # DHR at 256Hz is oversampled; take values at ~1 per beat
    mean_rr_s = np.mean(rr_ms) / 1000.0
    beat_spacing = max(1, int(sample_rate * mean_rr_s))
    rr_beats = rr_ms[::beat_spacing]

    if len(rr_beats) >= 3:
        rr_diff = np.diff(rr_beats)
        result["hrv_rmssd"] = float(np.sqrt(np.mean(rr_diff ** 2)))
        result["hrv_pnn50"] = float(np.mean(np.abs(rr_diff) > 50.0))

    # LF/HF ratio from FFT on uniformly sampled RR series
    result["hrv_lf_hf"] = _compute_lf_hf_ratio(dhr_valid, sample_rate)

    return result


def _compute_lf_hf_ratio(dhr_valid: np.ndarray, sample_rate: int) -> float:
    """Compute LF/HF power ratio from continuous DHR signal."""
    if len(dhr_valid) < sample_rate * 10:  # Need at least 10 seconds
        return np.nan

    # Convert to RR in seconds, interpolated uniformly at 4 Hz
    rr_s = 60.0 / dhr_valid
    interp_rate = 4.0
    n_orig = len(rr_s)
    t_orig = np.arange(n_orig) / sample_rate
    t_uniform = np.arange(t_orig[0], t_orig[-1], 1.0 / interp_rate)

    if len(t_uniform) < 16:
        return np.nan

    rr_uniform = np.interp(t_uniform, t_orig, rr_s)
    rr_uniform -= np.mean(rr_uniform)

    # FFT
    n = len(rr_uniform)
    freqs = np.fft.rfftfreq(n, d=1.0 / interp_rate)
    power = np.abs(np.fft.rfft(rr_uniform)) ** 2 / n

    lf_mask = (freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])
    hf_mask = (freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])

    lf_power = np.sum(power[lf_mask])
    hf_power = np.sum(power[hf_mask])

    if hf_power < 1e-12:
        return np.nan

    return float(lf_power / hf_power)


def estimate_resp_rate(dhr_valid: np.ndarray, sample_rate: int) -> float:
    """Estimate respiratory rate from DHR via respiratory sinus arrhythmia."""
    if len(dhr_valid) < sample_rate * 10:
        return np.nan

    rr_s = 60.0 / dhr_valid
    interp_rate = 4.0
    t_orig = np.arange(len(rr_s)) / sample_rate
    t_uniform = np.arange(t_orig[0], t_orig[-1], 1.0 / interp_rate)

    if len(t_uniform) < 16:
        return np.nan

    rr_uniform = np.interp(t_uniform, t_orig, rr_s)
    rr_uniform -= np.mean(rr_uniform)

    n = len(rr_uniform)
    freqs = np.fft.rfftfreq(n, d=1.0 / interp_rate)
    power = np.abs(np.fft.rfft(rr_uniform)) ** 2

    # Respiratory band: 0.15-0.5 Hz (9-30 breaths/min)
    resp_mask = (freqs >= 0.15) & (freqs <= 0.5)
    if not np.any(resp_mask):
        return np.nan

    resp_freqs = freqs[resp_mask]
    resp_power = power[resp_mask]
    peak_idx = np.argmax(resp_power)

    return float(resp_freqs[peak_idx] * 60)  # Hz -> breaths/min


# ---------------------------------------------------------------------------
# EDF signal loading
# ---------------------------------------------------------------------------

def load_edf_channels(edf_path: Path) -> dict:
    """Load needed channels from EDF file.

    Returns dict with keys:
        'hr', 'dhr', 'spo2', 'oxstatus', 'pos', 'leg': numpy arrays
        'hr_rate', 'dhr_rate', 'pos_rate', 'leg_rate': sample rates
        'start_time_sec': start time in seconds since midnight
        'duration_sec': recording duration
    """
    f = pyedflib.EdfReader(str(edf_path))
    try:
        n_channels = f.signals_in_file
        labels = [f.getLabel(i).strip() for i in range(n_channels)]

        # Map labels to canonical names
        chan_idx = {}
        for i, label in enumerate(labels):
            canonical = CHANNEL_LABEL_MAP.get(label)
            if canonical and canonical not in chan_idx:
                chan_idx[canonical] = i

        # Read signals
        result = {}
        for chan_name in NEEDED_CHANNELS:
            if chan_name in chan_idx:
                idx = chan_idx[chan_name]
                result[chan_name.lower()] = f.readSignal(idx)
                result[f"{chan_name.lower()}_rate"] = int(f.getSampleFrequency(idx))
            else:
                result[chan_name.lower()] = None
                result[f"{chan_name.lower()}_rate"] = 0

        # Start time
        dt = f.getStartdatetime()
        result["start_time_sec"] = dt.hour * 3600 + dt.minute * 60 + dt.second
        result["duration_sec"] = f.getFileDuration()

        return result
    finally:
        f.close()


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(args: tuple) -> pd.DataFrame | None:
    """Process a single subject. Takes (data_dir, subject_id) tuple for Pool.map."""
    data_dir, subject_id = args
    edf_path, xml_path, csv_path = resolve_paths(data_dir, subject_id)

    try:
        # Load EDF channels
        edf = load_edf_channels(edf_path)
    except Exception as e:
        logger.warning("EDF load failed for %s: %s", subject_id, e)
        return None

    if edf["hr"] is None:
        logger.warning("No HR channel for %s, skipping", subject_id)
        return None

    # Parse sleep stages
    try:
        stage_labels = parse_sleep_stages(xml_path)
    except Exception as e:
        logger.warning("Annotation parse failed for %s: %s", subject_id, e)
        return None

    # Determine number of epochs from EDF duration
    n_epochs_edf = int(edf["duration_sec"] / EPOCH_SEC)
    n_epochs = min(n_epochs_edf, len(stage_labels))

    if n_epochs < 60:  # At least 30 minutes
        logger.warning("Too few epochs for %s (%d), skipping", subject_id, n_epochs)
        return None

    # Align actigraphy
    actig_df = align_actigraphy(edf["start_time_sec"], csv_path, n_epochs)

    # Pre-extract channel arrays and rates
    hr = edf["hr"]
    dhr = edf["dhr"]
    spo2 = edf["spo2"]
    oxstatus = edf["oxstatus"]
    pos = edf["pos"]
    leg = edf["leg"]

    hr_rate = edf["hr_rate"]
    dhr_rate = edf["dhr_rate"]
    pos_rate = edf["pos_rate"]
    leg_rate = edf["leg_rate"]

    start_time_sec = edf["start_time_sec"]

    # Build rows
    rows = []
    for ep in range(n_epochs):
        row = {}
        row["mesa_id"] = int(subject_id)
        row["epoch_idx"] = ep

        # Clock time
        epoch_time_sec = (start_time_sec + ep * EPOCH_SEC) % 86400
        h = int(epoch_time_sec // 3600)
        m = int((epoch_time_sec % 3600) // 60)
        s = int(epoch_time_sec % 60)
        row["clock_time"] = f"{h:02d}:{m:02d}:{s:02d}"

        # --- HR features (1 Hz, 30 samples per epoch) ---
        hr_start = ep * EPOCH_SEC * hr_rate
        hr_end = hr_start + EPOCH_SEC * hr_rate
        if hr is not None and hr_end <= len(hr):
            hr_win = hr[hr_start:hr_end]
            hr_valid = hr_win[(hr_win >= DHR_MIN_BPM) & (hr_win <= DHR_MAX_BPM)]
            if len(hr_valid) > 0:
                row["hr_mean"] = float(np.mean(hr_valid))
                row["hr_std"] = float(np.std(hr_valid))
                row["hr_min"] = float(np.min(hr_valid))
                row["hr_max"] = float(np.max(hr_valid))
            else:
                row["hr_mean"] = row["hr_std"] = row["hr_min"] = row["hr_max"] = np.nan
        else:
            row["hr_mean"] = row["hr_std"] = row["hr_min"] = row["hr_max"] = np.nan

        # --- HRV features (from DHR, 256 Hz) ---
        if dhr is not None and dhr_rate > 0:
            dhr_start = ep * EPOCH_SEC * dhr_rate
            dhr_end = dhr_start + EPOCH_SEC * dhr_rate
            if dhr_end <= len(dhr):
                dhr_win = dhr[dhr_start:dhr_end]
                hrv = extract_hrv_epoch(dhr_win, dhr_rate)
                row.update(hrv)

                # Respiratory rate
                dhr_valid = dhr_win[
                    (dhr_win >= DHR_MIN_BPM) & (dhr_win <= DHR_MAX_BPM)
                ]
                row["resp_rate"] = estimate_resp_rate(dhr_valid, dhr_rate)
            else:
                row.update({
                    "hrv_mean_nn": np.nan, "hrv_sdnn": np.nan,
                    "hrv_rmssd": np.nan, "hrv_pnn50": np.nan,
                    "hrv_lf_hf": np.nan, "resp_rate": np.nan,
                })
        else:
            row.update({
                "hrv_mean_nn": np.nan, "hrv_sdnn": np.nan,
                "hrv_rmssd": np.nan, "hrv_pnn50": np.nan,
                "hrv_lf_hf": np.nan, "resp_rate": np.nan,
            })

        # --- SpO2 features (1 Hz, 30 samples per epoch) ---
        if spo2 is not None:
            spo2_start = ep * EPOCH_SEC  # 1 Hz
            spo2_end = spo2_start + EPOCH_SEC
            if spo2_end <= len(spo2):
                spo2_win = spo2[spo2_start:spo2_end]
                spo2_valid = spo2_win[
                    (spo2_win >= SPO2_MIN_VALID) & (spo2_win <= SPO2_MAX_VALID)
                ]
                if len(spo2_valid) > 0:
                    row["spo2_mean"] = float(np.mean(spo2_valid))
                    row["spo2_std"] = float(np.std(spo2_valid))
                    row["spo2_min"] = float(np.min(spo2_valid))
                else:
                    row["spo2_mean"] = row["spo2_std"] = row["spo2_min"] = np.nan
            else:
                row["spo2_mean"] = row["spo2_std"] = row["spo2_min"] = np.nan
        else:
            row["spo2_mean"] = row["spo2_std"] = row["spo2_min"] = np.nan

        # --- Leg movement features (32 Hz, 960 samples per epoch) ---
        if leg is not None and leg_rate > 0:
            leg_start = ep * EPOCH_SEC * leg_rate
            leg_end = leg_start + EPOCH_SEC * leg_rate
            if leg_end <= len(leg):
                leg_win = np.abs(leg[leg_start:leg_end])
                row["leg_move_mean"] = float(np.mean(leg_win))
                row["leg_move_std"] = float(np.std(leg_win))
                threshold = LEG_ACTIVE_THRESHOLD_FRAC * np.max(leg_win) if np.max(leg_win) > 0 else 0
                row["leg_pct_active"] = float(np.mean(leg_win > threshold))
            else:
                row["leg_move_mean"] = row["leg_move_std"] = row["leg_pct_active"] = np.nan
        else:
            row["leg_move_mean"] = row["leg_move_std"] = row["leg_pct_active"] = np.nan

        # --- Body position (32 Hz, mode per epoch) ---
        if pos is not None and pos_rate > 0:
            pos_start = ep * EPOCH_SEC * pos_rate
            pos_end = pos_start + EPOCH_SEC * pos_rate
            if pos_end <= len(pos):
                pos_win = pos[pos_start:pos_end].astype(int)
                pos_win = np.clip(pos_win, 0, 4)
                counts = np.bincount(pos_win, minlength=5)
                row["pos_dominant"] = int(np.argmax(counts))
                total = len(pos_win)
                for i, label in enumerate(POS_LABELS):
                    row[f"pos_{label}"] = float(counts[i] / total)
            else:
                row["pos_dominant"] = np.nan
                for label in POS_LABELS:
                    row[f"pos_{label}"] = np.nan
        else:
            row["pos_dominant"] = np.nan
            for label in POS_LABELS:
                row[f"pos_{label}"] = np.nan

        # --- Circadian features ---
        hour_frac = epoch_time_sec / 86400.0
        row["tod_cos"] = float(np.cos(2 * np.pi * hour_frac))
        row["tod_sin"] = float(np.sin(2 * np.pi * hour_frac))

        # --- Actigraphy features ---
        if actig_df is not None and ep < len(actig_df):
            actig_row = actig_df.iloc[ep]
            activity = actig_row.get("activity", np.nan)
            row["actig_activity"] = float(activity) if pd.notna(activity) else np.nan
            row["actig_wake"] = float(actig_row.get("wake", np.nan))
            row["actig_white_light"] = float(actig_row.get("whitelight", np.nan))
            row["actig_log_activity"] = (
                float(np.log1p(activity)) if pd.notna(activity) else np.nan
            )
        else:
            row["actig_activity"] = np.nan
            row["actig_wake"] = np.nan
            row["actig_white_light"] = np.nan
            row["actig_log_activity"] = np.nan

        # --- Quality flag ---
        stage = stage_labels[ep] if ep < len(stage_labels) else 6
        hr_ok = not np.isnan(row.get("hr_mean", np.nan))
        spo2_ok = not np.isnan(row.get("spo2_mean", np.nan))
        offwrist = (
            actig_df.iloc[ep].get("offwrist", 0) == 1
            if actig_df is not None and ep < len(actig_df)
            else False
        )
        stage_scored = stage in (0, 1, 2, 3, 5)

        row["valid_epoch"] = bool(hr_ok and spo2_ok and (not offwrist) and stage_scored)
        row["sleep_stage"] = int(stage)

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Subject %s: %d epochs, %d valid (%.0f%%)",
        subject_id,
        len(df),
        df["valid_epoch"].sum(),
        df["valid_epoch"].mean() * 100,
    )
    return df


# ---------------------------------------------------------------------------
# Metadata summary
# ---------------------------------------------------------------------------

def compute_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject summary statistics."""
    meta = df.groupby("mesa_id").agg(
        n_epochs=("epoch_idx", "count"),
        n_valid=("valid_epoch", "sum"),
        pct_valid=("valid_epoch", "mean"),
        hr_mean=("hr_mean", "mean"),
        spo2_mean=("spo2_mean", "mean"),
        n_wake=("sleep_stage", lambda x: (x == 0).sum()),
        n_n1=("sleep_stage", lambda x: (x == 1).sum()),
        n_n2=("sleep_stage", lambda x: (x == 2).sum()),
        n_n3=("sleep_stage", lambda x: (x == 3).sum()),
        n_rem=("sleep_stage", lambda x: (x == 5).sum()),
        n_unscored=("sleep_stage", lambda x: (x == 6).sum()),
    ).reset_index()
    meta["pct_valid"] = (meta["pct_valid"] * 100).round(1)
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build clean wearable-feature dataset from MESA data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./datasets/mesa"),
        help="Path to MESA dataset root (contains polysomnography/ and actigraphy/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for Parquet and metadata CSV. Defaults to data-dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Process only these subject IDs (e.g., 0001 0002).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover subjects, don't process.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir or args.data_dir

    # Discover subjects
    all_subjects = discover_subjects(args.data_dir)
    if args.subjects:
        subjects = [s for s in args.subjects if s in all_subjects]
        logger.info("Filtered to %d requested subjects", len(subjects))
    else:
        subjects = all_subjects

    if args.dry_run:
        logger.info("Dry run: found %d subjects to process", len(subjects))
        return

    if not subjects:
        logger.error("No subjects to process")
        sys.exit(1)

    logger.info("Processing %d subjects with %d workers...", len(subjects), args.workers)
    t0 = time.time()

    # Process subjects
    task_args = [(args.data_dir, sid) for sid in subjects]

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(process_subject, task_args)
    else:
        results = [process_subject(a) for a in task_args]

    # Filter out failed subjects
    dfs = [r for r in results if r is not None]
    n_failed = len(results) - len(dfs)
    if n_failed > 0:
        logger.warning("%d subjects failed processing", n_failed)

    if not dfs:
        logger.error("All subjects failed")
        sys.exit(1)

    # Concatenate
    full_df = pd.concat(dfs, ignore_index=True)
    elapsed = time.time() - t0
    logger.info(
        "Done: %d subjects, %d total epochs in %.1f seconds",
        len(dfs), len(full_df), elapsed,
    )

    # Save Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "mesa_clean_wearable.parquet"
    full_df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved: %s (%.1f MB)", parquet_path, parquet_path.stat().st_size / 1e6)

    # Save metadata
    meta = compute_metadata(full_df)
    meta_path = output_dir / "mesa_clean_metadata.csv"
    meta.to_csv(meta_path, index=False)
    logger.info("Saved: %s", meta_path)

    # Print summary
    scored = full_df[full_df["sleep_stage"].isin([0, 1, 2, 3, 5])]
    logger.info("--- Summary ---")
    logger.info("Subjects: %d", full_df["mesa_id"].nunique())
    logger.info("Total epochs: %d", len(full_df))
    logger.info("Valid epochs: %d (%.1f%%)", full_df["valid_epoch"].sum(), full_df["valid_epoch"].mean() * 100)
    logger.info("Scored epochs: %d", len(scored))
    if len(scored) > 0:
        stage_counts = scored["sleep_stage"].value_counts().sort_index()
        stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}
        for stage, count in stage_counts.items():
            pct = count / len(scored) * 100
            logger.info("  %s: %d (%.1f%%)", stage_names.get(stage, f"?{stage}"), count, pct)


if __name__ == "__main__":
    main()
