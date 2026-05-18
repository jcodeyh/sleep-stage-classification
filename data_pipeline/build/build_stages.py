#!/usr/bin/env python3
"""Build a clean wearable-feature dataset from STAGES PSG + actigraphy data.

Produces a single Parquet file with one row per 30-second epoch per subject,
containing only features collectible by consumer wearables (HR, HRV, SpO2,
actigraphy, circadian time) plus expert-scored PSG sleep stage labels.

This script is the STAGES counterpart to build_clean_dataset.py (MESA).
Key differences from MESA:
  - Expanded channel label map for multi-site PSG heterogeneity
  - HR derivation from ECG/PPG if pre-computed HR/DHR channels are absent
  - Actigraphy alignment for Huami Arc 60-second epochs (windowed aggregation)
  - 17 wearable-collectible features only (no leg EMG, body position, resp_rate, hrv_lf_hf)
  - String subject IDs (site-prefixed, not 4-digit integers)

Usage:
    python scripts/build_stages_dataset.py --data-dir ./datasets/stages --output-dir ./datasets/stages
    python scripts/build_stages_dataset.py --data-dir ./datasets/stages --dry-run
    python scripts/build_stages_dataset.py --data-dir ./datasets/stages --subjects SC-STGS-0001 SC-STGS-0002
    python scripts/build_stages_dataset.py --data-dir ./datasets/stages --workers 16
"""

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, filtfilt, find_peaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCH_SEC = 30

# Mayo site data excluded (known conversion errors)
STAGES_MAYO_SITES = {"mayo"}

# Huami Arc artifact value
HUAMI_ARTIFACT_VALUE = 20

# Sleep stage mapping for STAGES scoring CSV format
# STAGES uses: Wake, Stage1, Stage2, Stage3, REM (plus Awake variant)
STAGE_MAP_CSV = {
    "Wake": 0, "Awake": 0,
    "Stage1": 1, "N1": 1,
    "Stage2": 2, "N2": 2,
    "Stage3": 3, "N3": 3,
    "Stage4": 3,  # Merge N3/N4 per AASM
    "REM": 5, "R": 5,
}

# Movement events — map to unscored
MOVEMENT_EVENTS = {"MT", "mt", "MTa", "mta"}

# Physiological bounds
DHR_MIN_BPM = 30.0
DHR_MAX_BPM = 200.0
SPO2_MIN_VALID = 50.0
SPO2_MAX_VALID = 100.0

# Channel label variants -> canonical name
# Expanded for STAGES multi-site heterogeneity
CHANNEL_LABEL_MAP = {
    # HR (pre-computed)
    "HR": "HR", "hr": "HR", "Heart Rate": "HR", "Pulse": "HR",
    "Pulse Rate": "HR", "PR": "HR",
    # DHR (continuous interpolated HR)
    "DHR": "DHR", "dhr": "DHR",
    # SpO2
    "SpO2": "SpO2", "spo2": "SpO2", "SaO2": "SpO2", "SAO2": "SpO2",
    "sao2": "SpO2", "Spo2": "SpO2", "SPO2": "SpO2", "Sp02": "SpO2",
    # OxStatus
    "OxStatus": "OxStatus", "oxstatus": "OxStatus", "OXSTATUS": "OxStatus",
    "Ox Status": "OxStatus",
    # PPG/Pleth (for HR derivation fallback)
    "Pleth": "Pleth", "pleth": "Pleth", "PLETH": "Pleth",
    "PPG": "Pleth", "ppg": "Pleth", "Plethysmogram": "Pleth",
    # ECG (for HR derivation fallback)
    "ECG": "ECG", "ecg": "ECG", "EKG": "ECG", "ekg": "ECG",
    "ECG1": "ECG", "ECG2": "ECG", "ECG I": "ECG", "ECG II": "ECG",
    "ECG-ECG": "ECG", "EKG-EKG": "ECG",
}

# Channels we try to load (in priority order for HR)
WEARABLE_CHANNELS = ["HR", "DHR", "SpO2", "OxStatus", "Pleth", "ECG"]


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: Path) -> list[str]:
    """Find STAGES subject IDs that have EDF + scoring CSV + (optionally) actigraphy."""
    edf_dir = data_dir / "polysomnography" / "edfs"
    score_dir = data_dir / "polysomnography" / "annotations"
    actig_dir = data_dir / "actigraphy_json"  # Extracted from ZIP

    def extract_ids(directory: Path, suffix: str, strip: str = "") -> set[str]:
        ids = set()
        if not directory.exists():
            return ids
        for f in directory.glob(f"*{suffix}"):
            stem = f.stem
            if strip:
                stem = stem.replace(strip, "")
            # Skip Mayo site files
            if any(site in stem.lower() for site in STAGES_MAYO_SITES):
                continue
            # Skip duplicate variants (e.g., STNF00003_1)
            if "_" in stem and stem.split("_")[-1].isdigit():
                continue
            ids.add(stem)
        return ids

    edf_ids = extract_ids(edf_dir, ".edf")
    score_ids = extract_ids(score_dir, ".csv")

    # Actigraphy: JSON files per subject in subdirectories
    actig_ids = set()
    if actig_dir.exists():
        for subdir in actig_dir.iterdir():
            if subdir.is_dir() and any(subdir.glob("minbymin/*.json")):
                stem = subdir.name
                if any(site in stem.lower() for site in STAGES_MAYO_SITES):
                    continue
                actig_ids.add(stem)

    # Subjects with at least EDF + scoring CSV (actigraphy optional)
    common_psg = sorted(edf_ids & score_ids)
    common_all = sorted(edf_ids & score_ids & actig_ids) if actig_ids else []

    logger.info(
        "Discovered subjects: %d EDF, %d scoring, %d actigraphy, "
        "%d PSG+score, %d PSG+score+actig",
        len(edf_ids), len(score_ids), len(actig_ids),
        len(common_psg), len(common_all),
    )

    # Use subjects with all three if available, otherwise PSG+scoring only
    if common_all:
        return common_all
    return common_psg


def resolve_paths(
    data_dir: Path, subject_id: str
) -> tuple[Path, Path, Path | None]:
    """Return (edf_path, score_csv_path, actig_json_path) for a STAGES subject.

    actig_json_path may be None if actigraphy is not available.
    """
    edf_path = data_dir / "polysomnography" / "edfs" / f"{subject_id}.edf"
    score_path = data_dir / "polysomnography" / "annotations" / f"{subject_id}.csv"

    # Actigraphy JSON: {data_dir}/actigraphy_json/{subject_id}/minbymin/*.json
    actig_json_dir = data_dir / "actigraphy_json" / subject_id / "minbymin"
    actig_path = None
    if actig_json_dir.exists():
        json_files = list(actig_json_dir.glob("*.json"))
        if json_files:
            actig_path = json_files[0]  # Usually one file per subject

    return edf_path, score_path, actig_path


# ---------------------------------------------------------------------------
# Annotation parser (STAGES CSV scoring format)
# ---------------------------------------------------------------------------

def parse_sleep_stages(score_path: Path, start_time_sec: float,
                       duration_sec: float = 0) -> np.ndarray:
    """Parse sleep stage labels from STAGES scoring CSV.

    STAGES scoring CSVs have columns: Start Time, Duration (seconds), Event
    Events are point annotations at 30-second intervals.
    Stage labels: Wake, Stage1, Stage2, Stage3, REM (plus Awake, MT variants).

    start_time_sec: EDF recording start in seconds since midnight (from
        edf["start_time_sec"]).  Used as epoch-0 anchor and to detect
        midnight-crossing recordings (scored events whose clock time is
        numerically less than the EDF start belong to the next calendar day).
    duration_sec: EDF recording duration in seconds.  Used to discard wrapped
        events that map to epoch indices beyond the recording.

    Returns array of stage labels indexed by epoch number relative to EDF start.
    Epochs before the first scored event are filled with 6 (unscored).
    """
    n_epochs_edf = int(duration_sec / EPOCH_SEC) if duration_sec > 0 else 0

    df = pd.read_csv(score_path, on_bad_lines="warn", names=["Start Time", "Duration (seconds)", "Event"], skiprows=1)
    df.columns = df.columns.str.strip()

    # Parse start times to seconds since midnight
    def time_to_sec(t: str) -> float:
        t = t.strip()
        parts = t.split(":")
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2]) if len(parts) > 2 else 0
        return h * 3600 + m * 60 + s

    # Filter to sleep stage events only
    stage_events = []
    for _, row in df.iterrows():
        event = str(row["Event"]).strip()
        stage = STAGE_MAP_CSV.get(event)
        if stage is not None:
            start_sec = time_to_sec(str(row["Start Time"]))
            stage_events.append((start_sec, stage))
        elif event in MOVEMENT_EVENTS:
            start_sec = time_to_sec(str(row["Start Time"]))
            stage_events.append((start_sec, 6))  # Movement → unscored

    if not stage_events:
        raise ValueError(f"No sleep stage events in {score_path.name}")

    # Midnight-wrap: scored events whose clock time is numerically less than
    # the EDF start occurred after midnight on the next calendar day.
    n_wrapped = sum(1 for t, _ in stage_events if t < start_time_sec)
    stage_events = [(t + 86400 if t < start_time_sec else t, stage)
                    for t, stage in stage_events]
    stage_events.sort(key=lambda x: x[0])

    # Convert to epoch indices relative to EDF start
    epochs = {}
    n_discarded_negative = 0
    n_discarded_beyond = 0
    for event_time, stage in stage_events:
        epoch_idx = int(round((event_time - start_time_sec) / EPOCH_SEC))
        if epoch_idx < 0:
            n_discarded_negative += 1
            continue  # scored before EDF start (rare timing offset)
        if n_epochs_edf > 0 and epoch_idx >= n_epochs_edf:
            n_discarded_beyond += 1
            continue  # wrapped event beyond EDF duration (spurious)
        epochs[epoch_idx] = stage

    first_epoch_idx = min(epochs.keys()) if epochs else -1
    logger.debug(
        "%s: start_time_sec=%.0f, midnight-wrapped=%d events, "
        "discarded %d negative + %d beyond-EDF, first scored epoch idx=%d",
        score_path.stem, start_time_sec, n_wrapped,
        n_discarded_negative, n_discarded_beyond, first_epoch_idx,
    )
    if n_discarded_negative > 0 or n_discarded_beyond > 0:
        logger.info(
            "%s: discarded %d pre-start + %d beyond-EDF events "
            "(of %d total stage events, %d midnight-wrapped)",
            score_path.stem, n_discarded_negative, n_discarded_beyond,
            len(stage_events), n_wrapped,
        )

    if not epochs:
        raise ValueError(f"No scored epochs within EDF window in {score_path.name}")

    max_epoch = max(epochs.keys()) + 1
    labels = np.full(max_epoch, 6, dtype=np.int32)
    for idx, stage in epochs.items():
        labels[idx] = stage

    return labels


# ---------------------------------------------------------------------------
# STAGES actigraphy alignment (Huami Arc JSON, 1-minute epochs)
# ---------------------------------------------------------------------------

def align_stages_actigraphy(
    edf_start_dt: "datetime", actig_json_path: Path, n_psg_epochs: int
) -> pd.DataFrame | None:
    """Align STAGES Huami Arc actigraphy (JSON, 1-minute epochs) to 30-sec PSG epochs.

    Huami JSON format: {"items": [{"date": "2018-10-03", "minute": 1303,
        "activeness": 40, "mode": 6, "steps": 0}, ...]}
    Each item is one minute; "minute" is minutes since midnight of "date".
    """
    import json as json_mod
    from datetime import datetime, timedelta

    try:
        with open(actig_json_path) as f:
            data = json_mod.load(f)
        items = data.get("items", [])
    except Exception as e:
        logger.warning("Failed to read actigraphy JSON %s: %s",
                       actig_json_path.name, e)
        return None

    if not items:
        return None

    # Convert each item to (absolute_minute_since_epoch, activeness, steps, mode)
    # where minute = date_minutes_since_epoch + minute_of_day
    actig_records = []
    for item in items:
        try:
            date = datetime.strptime(item["date"], "%Y-%m-%d")
            minute_of_day = int(item["minute"])
            dt = date + timedelta(minutes=minute_of_day)
            ts_min = dt.timestamp() / 60  # absolute minutes
            activeness = float(item.get("activeness", 0))
            steps = float(item.get("steps", 0))
            mode = int(item.get("mode", 0))
            actig_records.append((ts_min, activeness, steps, mode))
        except (ValueError, KeyError):
            continue

    if not actig_records:
        return None

    # Sort by absolute time
    actig_records.sort(key=lambda x: x[0])
    actig_times = np.array([r[0] for r in actig_records])

    # Convert PSG start to absolute minutes using the full EDF recording datetime
    psg_min = edf_start_dt.timestamp() / 60

    # Find closest actigraphy record to PSG start
    diffs = np.abs(actig_times - psg_min)
    best_idx = int(np.argmin(diffs))
    min_diff_hrs = diffs[best_idx] / 60

    if min_diff_hrs > 6:  # Allow up to 6 hours offset for date mismatches
        logger.warning("Actigraphy too far from PSG start (%.1f hrs)", min_diff_hrs)
        return None

    # Build per-PSG-epoch actigraphy features
    rows = []
    for ep in range(n_psg_epochs):
        elapsed_sec = ep * EPOCH_SEC + EPOCH_SEC / 2
        actig_offset = int(elapsed_sec / 60)  # Convert to minute offset
        actig_idx = best_idx + actig_offset

        row = {}
        if 0 <= actig_idx < len(actig_records):
            _, activeness, steps, mode = actig_records[actig_idx]

            # Filter Huami Arc artifact value
            if activeness == HUAMI_ARTIFACT_VALUE:
                activeness = 0.0

            row["actig_activity"] = activeness
            row["actig_log_activity"] = float(np.log1p(activeness))
            # Wake estimation: steps > 0 indicates movement/wake
            row["actig_wake"] = 1.0 if steps > 0 else 0.0
            # No white light sensor in Huami Arc JSON data
            row["actig_white_light"] = np.nan
        else:
            row["actig_activity"] = np.nan
            row["actig_log_activity"] = np.nan
            row["actig_wake"] = np.nan
            row["actig_white_light"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EDF date correction (pyedflib 2-digit year bug)
# ---------------------------------------------------------------------------

def correct_edf_start_date(start_time_sec: int, actig_json_path: Path):
    """Return a corrected start datetime by anchoring the EDF's time-of-day to
    the actigraphy date whose data brackets it.

    pyedflib.getStartdatetime() misinterprets 2-digit EDF years and returns
    1985-01-01 for all STAGES recordings (actual dates: 2018-2019).
    The HH:MM:SS component is correct; only the date is wrong.

    Strategy:
      - Group actigraphy items by date → [min_minute, max_minute] per date
      - EDF start minute-of-day = start_time_sec / 60
      - Accept a date if edf_minute is within [min_minute - 120, max_minute + 120]
        (2-hour tolerance handles alignment gaps at recording boundaries)
      - If multiple candidates, pick the closest span
      - Return None (with a warning) if no candidate found

    Args:
        start_time_sec: seconds since midnight from the EDF header (correct)
        actig_json_path: path to the subject's Huami Arc JSON file

    Returns:
        datetime with the correct date and the original HH:MM:SS, or None
    """
    import json as json_mod
    from datetime import datetime, date as date_type

    TOLERANCE_MIN = 120  # 2-hour window on each side

    try:
        with open(actig_json_path) as f:
            data = json_mod.load(f)
        items = data.get("items", [])
    except Exception as e:
        logger.warning("correct_edf_start_date: failed to read %s: %s",
                       actig_json_path.name, e)
        return None

    if not items:
        return None

    # Group by date → track min/max minute-of-day
    date_spans: dict[date_type, tuple[int, int]] = {}
    for item in items:
        try:
            d = datetime.strptime(item["date"], "%Y-%m-%d").date()
            minute = int(item["minute"])
        except (ValueError, KeyError):
            continue
        if d in date_spans:
            lo, hi = date_spans[d]
            date_spans[d] = (min(lo, minute), max(hi, minute))
        else:
            date_spans[d] = (minute, minute)

    if not date_spans:
        return None

    edf_minute = start_time_sec / 60.0  # e.g. 23:04:44 → 1384.7

    # Find candidate dates whose span brackets edf_minute within tolerance
    best_date = None
    best_dist = float("inf")
    for d, (lo, hi) in date_spans.items():
        if (lo - TOLERANCE_MIN) <= edf_minute <= (hi + TOLERANCE_MIN):
            # Distance = 0 if inside span, else gap to nearest edge
            dist = max(0.0, lo - edf_minute, edf_minute - hi)
            if dist < best_dist:
                best_dist = dist
                best_date = d

    if best_date is None:
        logger.warning(
            "correct_edf_start_date: no actigraphy date brackets EDF start "
            "minute %.1f within %d min tolerance (dates checked: %s)",
            edf_minute, TOLERANCE_MIN,
            ", ".join(str(d) for d in sorted(date_spans)),
        )
        return None

    h = int(start_time_sec // 3600)
    m = int((start_time_sec % 3600) // 60)
    s = int(start_time_sec % 60)
    return datetime(best_date.year, best_date.month, best_date.day, h, m, s)


# ---------------------------------------------------------------------------
# HRV feature extraction (wearable subset — no LF/HF)
# ---------------------------------------------------------------------------

def extract_hrv_epoch(dhr_epoch: np.ndarray, sample_rate: int = 256) -> dict:
    """Compute time-domain HRV features from a DHR epoch.

    Returns dict with keys: hrv_mean_nn, hrv_sdnn, hrv_rmssd, hrv_pnn50
    (No hrv_lf_hf — frequency-domain metric not available on consumer wearables.)
    """
    result = {
        "hrv_mean_nn": np.nan,
        "hrv_sdnn": np.nan,
        "hrv_rmssd": np.nan,
        "hrv_pnn50": np.nan,
    }

    valid_mask = (dhr_epoch >= DHR_MIN_BPM) & (dhr_epoch <= DHR_MAX_BPM)
    dhr_valid = dhr_epoch[valid_mask]

    if len(dhr_valid) < sample_rate * 5:
        return result

    rr_ms = 60000.0 / dhr_valid

    result["hrv_mean_nn"] = float(np.mean(rr_ms))
    result["hrv_sdnn"] = float(np.std(rr_ms))

    mean_rr_s = np.mean(rr_ms) / 1000.0
    beat_spacing = max(1, int(sample_rate * mean_rr_s))
    rr_beats = rr_ms[::beat_spacing]

    if len(rr_beats) >= 3:
        rr_diff = np.diff(rr_beats)
        result["hrv_rmssd"] = float(np.sqrt(np.mean(rr_diff ** 2)))
        result["hrv_pnn50"] = float(np.mean(np.abs(rr_diff) > 50.0))

    return result


def extract_hrv_from_rr(rr_intervals_ms: np.ndarray) -> dict:
    """Compute HRV features from actual RR intervals (from ECG/PPG peak detection).

    This is more accurate than the DHR→RR conversion used in MESA since we have
    true beat-to-beat intervals rather than interpolated continuous HR.
    """
    result = {
        "hrv_mean_nn": np.nan,
        "hrv_sdnn": np.nan,
        "hrv_rmssd": np.nan,
        "hrv_pnn50": np.nan,
    }

    if len(rr_intervals_ms) < 5:
        return result

    # Filter to physiological range (300-2000 ms ≈ 30-200 bpm)
    valid = rr_intervals_ms[(rr_intervals_ms >= 300) & (rr_intervals_ms <= 2000)]
    if len(valid) < 5:
        return result

    result["hrv_mean_nn"] = float(np.mean(valid))
    result["hrv_sdnn"] = float(np.std(valid))

    if len(valid) >= 3:
        rr_diff = np.diff(valid)
        result["hrv_rmssd"] = float(np.sqrt(np.mean(rr_diff ** 2)))
        result["hrv_pnn50"] = float(np.mean(np.abs(rr_diff) > 50.0))

    return result


# ---------------------------------------------------------------------------
# HR derivation from PPG/ECG (fallback when no pre-computed HR channel)
# ---------------------------------------------------------------------------

def derive_hr_from_ppg(ppg_signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Derive HR and RR intervals from PPG/Pleth signal.

    Returns:
        (hr_1hz, rr_intervals_ms) — HR resampled to 1 Hz, and raw RR intervals.
    """
    if len(ppg_signal) < sample_rate * 5:
        return np.array([]), np.array([])

    # Bandpass filter 0.5-8 Hz
    nyq = sample_rate / 2.0
    low = max(0.5 / nyq, 0.01)
    high = min(8.0 / nyq, 0.99)
    if low >= high:
        return np.array([]), np.array([])
    b, a = butter(3, [low, high], btype="band")
    filtered = filtfilt(b, a, ppg_signal)

    # Find systolic peaks
    min_dist = int(sample_rate * 0.3)  # Min 0.3s between beats (200 bpm max)
    peaks, _ = find_peaks(filtered, distance=min_dist, height=np.std(filtered) * 0.3)

    if len(peaks) < 3:
        return np.array([]), np.array([])

    # Compute instantaneous HR from peak-to-peak intervals
    peak_intervals_s = np.diff(peaks) / sample_rate
    # Filter physiological range
    valid_mask = (peak_intervals_s >= 0.3) & (peak_intervals_s <= 2.0)
    peak_intervals_s = peak_intervals_s[valid_mask]

    if len(peak_intervals_s) < 2:
        return np.array([]), np.array([])

    hr_instantaneous = 60.0 / peak_intervals_s
    rr_ms = peak_intervals_s * 1000.0

    # Resample to 1 Hz
    duration_s = len(ppg_signal) / sample_rate
    n_seconds = int(duration_s)
    if n_seconds < 1:
        return np.array([]), rr_ms

    t_beats = (peaks[1:][valid_mask[:len(peaks)-1]] / sample_rate)
    if len(t_beats) < 2:
        return np.array([]), rr_ms
    t_out = np.arange(n_seconds) + 0.5
    t_out = t_out[(t_out >= t_beats[0]) & (t_out <= t_beats[-1])]
    if len(t_out) < 1:
        return np.array([]), rr_ms

    hr_1hz = np.interp(t_out, t_beats, hr_instantaneous[:len(t_beats)])

    return hr_1hz, rr_ms


def derive_hr_from_ecg(ecg_signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Derive HR and RR intervals from ECG signal via R-peak detection.

    Uses a simplified Pan-Tompkins approach.
    """
    if len(ecg_signal) < sample_rate * 5:
        return np.array([]), np.array([])

    # Bandpass filter 5-15 Hz (QRS complex)
    nyq = sample_rate / 2.0
    low = max(5.0 / nyq, 0.01)
    high = min(15.0 / nyq, 0.99)
    if low >= high:
        return np.array([]), np.array([])
    b, a = butter(2, [low, high], btype="band")
    filtered = filtfilt(b, a, ecg_signal)

    # Square the signal
    squared = filtered ** 2

    # Moving average (150 ms window)
    win_size = max(1, int(sample_rate * 0.15))
    ma = np.convolve(squared, np.ones(win_size) / win_size, mode="same")

    # Find R-peaks
    min_dist = int(sample_rate * 0.3)
    threshold = np.mean(ma) + 0.5 * np.std(ma)
    peaks, _ = find_peaks(ma, distance=min_dist, height=threshold)

    if len(peaks) < 3:
        return np.array([]), np.array([])

    # Same as PPG from here
    peak_intervals_s = np.diff(peaks) / sample_rate
    valid_mask = (peak_intervals_s >= 0.3) & (peak_intervals_s <= 2.0)
    peak_intervals_s = peak_intervals_s[valid_mask]

    if len(peak_intervals_s) < 2:
        return np.array([]), np.array([])

    hr_instantaneous = 60.0 / peak_intervals_s
    rr_ms = peak_intervals_s * 1000.0

    duration_s = len(ecg_signal) / sample_rate
    n_seconds = int(duration_s)
    if n_seconds < 1:
        return np.array([]), rr_ms

    t_beats = (peaks[1:][valid_mask[:len(peaks)-1]] / sample_rate)
    if len(t_beats) < 2:
        return np.array([]), rr_ms
    t_out = np.arange(n_seconds) + 0.5
    t_out = t_out[(t_out >= t_beats[0]) & (t_out <= t_beats[-1])]
    if len(t_out) < 1:
        return np.array([]), rr_ms

    hr_1hz = np.interp(t_out, t_beats, hr_instantaneous[:len(t_beats)])

    return hr_1hz, rr_ms


# ---------------------------------------------------------------------------
# EDF signal loading
# ---------------------------------------------------------------------------

def load_edf_channels(edf_path: Path) -> dict:
    """Load wearable-relevant channels from STAGES EDF file.

    Returns dict with signal arrays and metadata. Attempts HR/DHR channels first;
    if absent, falls back to PPG/ECG for HR derivation.
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

        result = {"available_channels": list(chan_idx.keys())}

        # Read each channel
        for chan_name in WEARABLE_CHANNELS:
            if chan_name in chan_idx:
                idx = chan_idx[chan_name]
                result[chan_name.lower()] = f.readSignal(idx)
                result[f"{chan_name.lower()}_rate"] = int(f.getSampleFrequency(idx))
            else:
                result[chan_name.lower()] = None
                result[f"{chan_name.lower()}_rate"] = 0

        # Start time
        dt = f.getStartdatetime()
        result["start_datetime"] = dt
        result["start_time_sec"] = dt.hour * 3600 + dt.minute * 60 + dt.second
        result["duration_sec"] = f.getFileDuration()

        return result
    finally:
        f.close()


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(args: tuple) -> pd.DataFrame | None:
    """Process a single STAGES subject. Takes (data_dir, subject_id) for Pool.map."""
    data_dir, subject_id = args
    edf_path, score_path, actig_path = resolve_paths(data_dir, subject_id)

    # Load EDF channels
    try:
        edf = load_edf_channels(edf_path)
    except Exception as e:
        logger.warning("EDF load failed for %s: %s", subject_id, e)
        return None

    # Correct pyedflib 1985 date bug using actigraphy timestamps
    if actig_path is not None:
        corrected_dt = correct_edf_start_date(edf["start_time_sec"], actig_path)
        if corrected_dt is not None:
            logger.warning(
                "%s: EDF date corrected %s → %s (actigraphy anchor date %s)",
                subject_id,
                edf["start_datetime"].date(),
                corrected_dt.date(),
                corrected_dt.strftime("%Y-%m-%d"),
            )
            edf["start_datetime"] = corrected_dt
            # edf["start_time_sec"] is left unchanged — clock_time math is correct
        else:
            logger.warning(
                "%s: cannot correct EDF date from actigraphy, alignment may be wrong",
                subject_id,
            )

    # Determine HR source
    hr = edf.get("hr")
    hr_rate = edf.get("hr_rate", 0)
    dhr = edf.get("dhr")
    dhr_rate = edf.get("dhr_rate", 0)
    spo2 = edf.get("spo2")
    pleth = edf.get("pleth")
    pleth_rate = edf.get("pleth_rate", 0)
    ecg = edf.get("ecg")
    ecg_rate = edf.get("ecg_rate", 0)

    # HR derivation strategy
    hr_source = "none"
    derived_hr_1hz = None
    derived_rr_ms = None

    if hr is not None and hr_rate > 0:
        hr_source = "hr_channel"
    elif dhr is not None and dhr_rate > 0:
        hr_source = "dhr_channel"
    elif pleth is not None and pleth_rate > 0:
        derived_hr_1hz, derived_rr_ms = derive_hr_from_ppg(pleth, pleth_rate)
        if len(derived_hr_1hz) > 0:
            hr_source = "ppg_derived"
        else:
            logger.debug("PPG derivation failed for %s, trying ECG", subject_id)
    if hr_source == "none" and ecg is not None and ecg_rate > 0:
        derived_hr_1hz, derived_rr_ms = derive_hr_from_ecg(ecg, ecg_rate)
        if len(derived_hr_1hz) > 0:
            hr_source = "ecg_derived"

    if hr_source == "none":
        logger.warning("No HR source for %s (channels: %s), skipping",
                       subject_id, edf.get("available_channels", []))
        return None

    # Parse sleep stages from scoring CSV
    try:
        stage_labels = parse_sleep_stages(score_path, edf["start_time_sec"],
                                         duration_sec=edf["duration_sec"])
    except Exception as e:
        logger.warning("Scoring CSV parse failed for %s: %s", subject_id, e)
        return None

    # Determine number of epochs
    n_epochs_edf = int(edf["duration_sec"] / EPOCH_SEC)
    n_epochs = min(n_epochs_edf, len(stage_labels))

    if n_epochs < 60:  # At least 30 minutes
        logger.warning("Too few epochs for %s (%d), skipping", subject_id, n_epochs)
        return None

    # Align actigraphy
    actig_df = None
    if actig_path is not None:
        actig_df = align_stages_actigraphy(edf["start_datetime"], actig_path, n_epochs)

    start_time_sec = edf["start_time_sec"]

    # Build rows
    rows = []
    for ep in range(n_epochs):
        row = {}
        row["subject_id"] = subject_id
        row["epoch_idx"] = ep

        # Clock time
        epoch_time_sec = (start_time_sec + ep * EPOCH_SEC) % 86400
        h = int(epoch_time_sec // 3600)
        m = int((epoch_time_sec % 3600) // 60)
        s = int(epoch_time_sec % 60)
        row["clock_time"] = f"{h:02d}:{m:02d}:{s:02d}"

        # --- HR features ---
        if hr_source == "hr_channel":
            hr_start = ep * EPOCH_SEC * hr_rate
            hr_end = hr_start + EPOCH_SEC * hr_rate
            if hr_end <= len(hr):
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

        elif hr_source in ("ppg_derived", "ecg_derived"):
            # derived_hr_1hz is at 1 Hz
            hr_start = ep * EPOCH_SEC
            hr_end = hr_start + EPOCH_SEC
            if derived_hr_1hz is not None and hr_end <= len(derived_hr_1hz):
                hr_win = derived_hr_1hz[hr_start:hr_end]
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

        elif hr_source == "dhr_channel":
            # Use DHR for HR features
            dhr_start = ep * EPOCH_SEC * dhr_rate
            dhr_end = dhr_start + EPOCH_SEC * dhr_rate
            if dhr_end <= len(dhr):
                dhr_win = dhr[dhr_start:dhr_end]
                dhr_valid = dhr_win[(dhr_win >= DHR_MIN_BPM) & (dhr_win <= DHR_MAX_BPM)]
                if len(dhr_valid) > 0:
                    row["hr_mean"] = float(np.mean(dhr_valid))
                    row["hr_std"] = float(np.std(dhr_valid))
                    row["hr_min"] = float(np.min(dhr_valid))
                    row["hr_max"] = float(np.max(dhr_valid))
                else:
                    row["hr_mean"] = row["hr_std"] = row["hr_min"] = row["hr_max"] = np.nan
            else:
                row["hr_mean"] = row["hr_std"] = row["hr_min"] = row["hr_max"] = np.nan
        else:
            row["hr_mean"] = row["hr_std"] = row["hr_min"] = row["hr_max"] = np.nan

        # --- HRV features (time-domain only) ---
        if hr_source == "dhr_channel" and dhr is not None and dhr_rate > 0:
            dhr_start = ep * EPOCH_SEC * dhr_rate
            dhr_end = dhr_start + EPOCH_SEC * dhr_rate
            if dhr_end <= len(dhr):
                hrv = extract_hrv_epoch(dhr[dhr_start:dhr_end], dhr_rate)
                row.update(hrv)
            else:
                row.update({"hrv_mean_nn": np.nan, "hrv_sdnn": np.nan,
                            "hrv_rmssd": np.nan, "hrv_pnn50": np.nan})

        elif hr_source in ("ppg_derived", "ecg_derived") and derived_rr_ms is not None:
            # Use actual RR intervals for this epoch
            # Estimate which RR intervals fall in this epoch
            epoch_start_s = ep * EPOCH_SEC
            epoch_end_s = epoch_start_s + EPOCH_SEC
            # Approximate: take a slice of RR intervals
            cumulative_time = np.cumsum(derived_rr_ms) / 1000.0
            mask = (cumulative_time >= epoch_start_s) & (cumulative_time < epoch_end_s)
            epoch_rr = derived_rr_ms[mask]
            hrv = extract_hrv_from_rr(epoch_rr)
            row.update(hrv)

        elif hr_source == "hr_channel" and hr is not None and hr_rate > 0:
            # Derive HRV from HR channel (convert to RR like MESA does with DHR)
            hr_start = ep * EPOCH_SEC * hr_rate
            hr_end = hr_start + EPOCH_SEC * hr_rate
            if hr_end <= len(hr):
                hrv = extract_hrv_epoch(hr[hr_start:hr_end], hr_rate)
                row.update(hrv)
            else:
                row.update({"hrv_mean_nn": np.nan, "hrv_sdnn": np.nan,
                            "hrv_rmssd": np.nan, "hrv_pnn50": np.nan})
        else:
            row.update({"hrv_mean_nn": np.nan, "hrv_sdnn": np.nan,
                        "hrv_rmssd": np.nan, "hrv_pnn50": np.nan})

        # --- SpO2 features ---
        if spo2 is not None:
            spo2_rate = edf.get("spo2_rate", 1)
            spo2_start = ep * EPOCH_SEC * spo2_rate
            spo2_end = spo2_start + EPOCH_SEC * spo2_rate
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

        # --- Circadian features ---
        hour_frac = epoch_time_sec / 86400.0
        row["tod_cos"] = float(np.cos(2 * np.pi * hour_frac))
        row["tod_sin"] = float(np.sin(2 * np.pi * hour_frac))

        # --- Actigraphy features ---
        if actig_df is not None and ep < len(actig_df):
            actig_row = actig_df.iloc[ep]
            row["actig_activity"] = actig_row.get("actig_activity", np.nan)
            row["actig_wake"] = actig_row.get("actig_wake", np.nan)
            row["actig_white_light"] = actig_row.get("actig_white_light", np.nan)
            row["actig_log_activity"] = actig_row.get("actig_log_activity", np.nan)
        else:
            row["actig_activity"] = np.nan
            row["actig_wake"] = np.nan
            row["actig_white_light"] = np.nan
            row["actig_log_activity"] = np.nan

        # --- Quality flag ---
        stage = stage_labels[ep] if ep < len(stage_labels) else 6
        hr_ok = not np.isnan(row.get("hr_mean", np.nan))
        spo2_ok = not np.isnan(row.get("spo2_mean", np.nan))
        stage_scored = stage in (0, 1, 2, 3, 5)

        row["valid_epoch"] = bool(hr_ok and spo2_ok and stage_scored)
        row["sleep_stage"] = int(stage)
        row["hr_source"] = hr_source

        rows.append(row)

    df = pd.DataFrame(rows)

    # Per-subject diagnostic logging
    n_total = len(df)
    n_valid = df["valid_epoch"].sum()
    hr_nan_rate = df["hr_mean"].isna().mean() * 100
    hrv_nan_rate = df["hrv_rmssd"].isna().mean() * 100
    actig_coverage = df["actig_activity"].notna().mean() * 100

    logger.info(
        "Subject %s [%s]: %d epochs, %d valid (%.0f%%), HR source: %s, "
        "HR NaN: %.1f%%, HRV NaN: %.1f%%, actig: %.1f%%",
        subject_id, hr_source, n_total, n_valid,
        n_valid / n_total * 100 if n_total > 0 else 0,
        hr_source, hr_nan_rate, hrv_nan_rate, actig_coverage,
    )

    # Flag subjects with high R-peak failure (ECG/PPG-derived HR)
    if hr_source in ("ecg_derived", "ppg_derived") and hr_nan_rate > 50:
        logger.warning(
            "%s: high R-peak failure rate — %.1f%% HR NaN epochs (source: %s)",
            subject_id, hr_nan_rate, hr_source,
        )

    return df


# ---------------------------------------------------------------------------
# Metadata summary
# ---------------------------------------------------------------------------

def compute_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject summary statistics."""
    meta = df.groupby("subject_id").agg(
        n_epochs=("epoch_idx", "count"),
        n_valid=("valid_epoch", "sum"),
        pct_valid=("valid_epoch", "mean"),
        hr_mean=("hr_mean", "mean"),
        hr_nan_rate=("hr_mean", lambda x: x.isna().mean()),
        spo2_mean=("spo2_mean", "mean"),
        actig_coverage=("actig_activity", lambda x: x.notna().mean()),
        hr_source=("hr_source", "first"),
        n_wake=("sleep_stage", lambda x: (x == 0).sum()),
        n_n1=("sleep_stage", lambda x: (x == 1).sum()),
        n_n2=("sleep_stage", lambda x: (x == 2).sum()),
        n_n3=("sleep_stage", lambda x: (x == 3).sum()),
        n_rem=("sleep_stage", lambda x: (x == 5).sum()),
        n_unscored=("sleep_stage", lambda x: (x == 6).sum()),
    ).reset_index()
    meta["pct_valid"] = (meta["pct_valid"] * 100).round(1)
    meta["hr_nan_rate"] = (meta["hr_nan_rate"] * 100).round(1)
    meta["actig_coverage"] = (meta["actig_coverage"] * 100).round(1)

    # Extract site prefix from subject_id (e.g. "STNF00001" → "STNF")
    meta["site"] = meta["subject_id"].str.extract(r"^([A-Z]{4})")

    return meta


def log_site_qc(meta: pd.DataFrame) -> None:
    """Log per-site quality summary from subject metadata."""
    if "site" not in meta.columns or meta["site"].isna().all():
        return

    logger.info("--- Site-level QC ---")
    for site, grp in meta.groupby("site"):
        n_subj = len(grp)
        mean_valid = grp["pct_valid"].mean()
        mean_hr_nan = grp["hr_nan_rate"].mean()
        mean_actig = grp["actig_coverage"].mean()

        # HR source distribution
        hr_dist = grp["hr_source"].value_counts()
        hr_dist_str = ", ".join(f"{src}={cnt}" for src, cnt in hr_dist.items())

        logger.info(
            "  %s: %d subjects, valid=%.1f%%, HR NaN=%.1f%%, "
            "actig=%.1f%%, HR sources: %s",
            site, n_subj, mean_valid, mean_hr_nan, mean_actig, hr_dist_str,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build clean wearable-feature dataset from STAGES PSG + actigraphy"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="STAGES data directory (containing polysomnography/ and actigraphy/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: same as data-dir)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Process only these subject IDs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just count subjects, don't process",
    )

    args = parser.parse_args()
    output_dir = args.output_dir or args.data_dir

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects(args.data_dir)

    if not subjects:
        logger.error("No valid subjects found in %s", args.data_dir)
        sys.exit(1)

    logger.info("Processing %d STAGES subjects", len(subjects))

    if args.dry_run:
        for s in subjects[:20]:
            print(f"  {s}")
        if len(subjects) > 20:
            print(f"  ... and {len(subjects) - 20} more")
        return

    # Process subjects
    t0 = time.time()
    task_args = [(args.data_dir, s) for s in subjects]

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(process_subject, task_args)
    else:
        results = [process_subject(a) for a in task_args]

    # Combine results
    dfs = [r for r in results if r is not None]
    if not dfs:
        logger.error("No subjects processed successfully")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    elapsed = time.time() - t0

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "stages_clean_wearable.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    # Metadata
    meta = compute_metadata(df)
    meta_path = output_dir / "stages_clean_metadata.csv"
    meta.to_csv(meta_path, index=False)
    log_site_qc(meta)

    # Summary
    n_subjects = df["subject_id"].nunique()
    n_valid = df["valid_epoch"].sum()
    logger.info(
        "Done in %.1f min. %d subjects, %d epochs, %d valid (%.1f%%)",
        elapsed / 60, n_subjects, len(df), n_valid,
        n_valid / len(df) * 100 if len(df) > 0 else 0,
    )
    logger.info("Parquet saved to: %s (%.1f MB)", parquet_path,
                parquet_path.stat().st_size / 1e6)
    logger.info("Metadata saved to: %s", meta_path)

    # Stage distribution
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
