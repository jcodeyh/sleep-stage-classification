"""FS-WEARABLE-EXT: Extended wearable feature set for sleep staging.

Extends the built-in FS-W-PA-14 (14 features) with additional circadian,
actigraphy, and HRV features for a total of ~20 features.

Additional features over FS-W-PA-14:
  - Circadian: tod_cos, tod_sin (time-of-day sinusoidal encoding)
  - Actigraphy: act_mean, act_std, act_pct_zero, act_log_mean (from CSV files)
  - Extended HRV: hrv_rmssd, hrv_pnn50

This feature set requires both PSG EDF files (for PPG/SpO2) AND actigraphy CSV
files. It generates per-subject HDF5 files compatible with SleepKit's H5Dataloader.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Channel name variants to canonical names
CHANNEL_MAP = {
    "pleth": "pleth", "Pleth": "pleth", "PPG": "pleth", "ppg": "pleth", "PLETH": "pleth",
    "spo2": "spo2", "SpO2": "spo2", "SaO2": "spo2", "sao2": "spo2", "SAO2": "spo2",
    "hr": "hr", "HR": "hr", "Heart Rate": "hr", "Pulse": "hr",
    "oxstatus": "oxstatus", "OxStatus": "oxstatus", "OXSTATUS": "oxstatus",
}

# Sleep stage mapping from annotation events
STAGE_MAP = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "Stage 4 sleep|4": 4,
    "REM sleep|5": 5,
}

# PPG bandpass filter range (Hz)
PPG_LOWCUT = 0.5
PPG_HIGHCUT = 3.0

# RR interval valid range (seconds)
RR_MIN = 0.3
RR_MAX = 2.0

# HRV frequency bands (Hz)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.4)


class FSWearableExt:
    """Extended wearable feature set: FS-W-PA-14 + circadian + actigraphy + extended HRV."""

    @staticmethod
    def name() -> str:
        return "FS-WEARABLE-EXT"

    @staticmethod
    def feature_names() -> list[str]:
        return [
            # FS-W-PA-14 base features (14)
            "hr_bpm",
            "hrv_td_mean_nn",
            "hrv_td_sd_nn",
            "hrv_td_median_nn",
            "hrv_fd_lfhf_ratio",
            "spo2_mu",
            "spo2_std",
            "spo2_med",
            "mov_mu",
            "mov_std",
            "mov_med",
            "rsp_bpm",
            "spo2_qos",
            "hrv_qos",
            # Extended HRV (2)
            "hrv_rmssd",
            "hrv_pnn50",
            # Circadian (2)
            "tod_cos",
            "tod_sin",
            # Actigraphy from CSV (4)
            "act_mean",
            "act_std",
            "act_pct_zero",
            "act_log_mean",
        ]

    @staticmethod
    def generate_subject_features(subject_id: str, ds_name: str, params) -> None:
        """Generate extended wearable features for a single subject.

        Args:
            subject_id: Subject identifier string.
            ds_name: Dataset name ("mesa" or "stages").
            params: SleepKit TaskParams with feature configuration.
        """
        import sleepkit as sk

        sample_rate = int(params.feature.sampling_rate)  # e.g., 64 Hz
        win_size = int(params.feature.frame_size)  # e.g., 3840 samples (60s at 64Hz)
        ovl_size = win_size // 2
        win_sec = win_size / sample_rate  # Window duration in seconds

        # Get dataset-specific params
        ds_params = {}
        for ds in params.datasets:
            if ds.name == ds_name:
                ds_params = ds.params
                break

        ds_path = Path(ds_params.get("path", f"./datasets/{ds_name}"))
        save_path = Path(params.feature.save_path) / ds_name
        save_path.mkdir(parents=True, exist_ok=True)

        output_file = save_path / f"{subject_id}.h5"
        if output_file.exists():
            logger.debug("Already exists, skipping: %s", output_file)
            return

        # Load PSG signals
        try:
            signals, channel_map, psg_start_time, psg_sample_rate = _load_psg_signals(
                ds_path, subject_id, sample_rate
            )
        except Exception as e:
            logger.warning("Failed to load PSG for %s/%s: %s", ds_name, subject_id, e)
            return

        if "pleth" not in signals:
            logger.warning("No Pleth channel for %s/%s, skipping", ds_name, subject_id)
            return

        # Load sleep stage labels from annotations
        try:
            stage_labels_30s, annot_start_time = _load_annotations(
                ds_path, subject_id
            )
        except Exception as e:
            logger.warning(
                "Failed to load annotations for %s/%s: %s", ds_name, subject_id, e
            )
            return

        # Load actigraphy from CSV
        try:
            actig_data, actig_epoch_sec = _load_actigraphy(
                ds_path, subject_id, ds_name
            )
        except Exception as e:
            logger.warning(
                "Failed to load actigraphy for %s/%s: %s", ds_name, subject_id, e
            )
            actig_data = None
            actig_epoch_sec = 30

        # Compute number of windows
        pleth = signals["pleth"]
        n_samples = len(pleth)
        num_windows = max(0, (n_samples - win_size) // ovl_size)

        if num_windows == 0:
            logger.warning("No windows for %s/%s (too short)", ds_name, subject_id)
            return

        n_features = len(FSWearableExt.feature_names())
        features = np.full((num_windows, n_features), np.nan, dtype=np.float32)
        labels = np.zeros(num_windows, dtype=np.int32)
        mask = np.zeros(num_windows, dtype=np.int32)

        spo2 = signals.get("spo2")
        oxstatus = signals.get("oxstatus")

        for w in range(num_windows):
            start = w * ovl_size
            end = start + win_size
            center_sample = start + win_size // 2
            center_time_sec = center_sample / sample_rate

            # Extract signal windows
            pleth_win = pleth[start:end]
            spo2_win = spo2[start:end] if spo2 is not None else None
            oxstatus_win = oxstatus[start:end] if oxstatus is not None else None

            # --- Base FS-W-PA-14 features ---

            # HRV features from PPG
            rr_intervals = _detect_rr_intervals(pleth_win, sample_rate)
            valid_rr = rr_intervals[
                (rr_intervals >= RR_MIN) & (rr_intervals <= RR_MAX)
            ]

            if len(valid_rr) >= 3:
                hr_bpm = 60.0 / np.mean(valid_rr)
                mean_nn = np.mean(valid_rr) * 1000  # ms
                sd_nn = np.std(valid_rr) * 1000
                median_nn = np.median(valid_rr) * 1000
                lfhf = _compute_lfhf_ratio(valid_rr)
                rsp_bpm = _estimate_respiratory_rate(valid_rr)
                hrv_qos = 1.0 - (len(rr_intervals) - len(valid_rr)) / max(
                    len(rr_intervals), 1
                )

                # Extended HRV
                rr_diff = np.diff(valid_rr) * 1000  # ms
                rmssd = np.sqrt(np.mean(rr_diff**2)) if len(rr_diff) > 0 else 0.0
                pnn50 = (
                    np.mean(np.abs(rr_diff) > 50) if len(rr_diff) > 0 else 0.0
                )
            else:
                hr_bpm = np.nan
                mean_nn = sd_nn = median_nn = np.nan
                lfhf = rsp_bpm = np.nan
                hrv_qos = 0.0
                rmssd = pnn50 = np.nan

            # SpO2 features
            if spo2_win is not None:
                spo2_clean = spo2_win[
                    (spo2_win >= 50) & (spo2_win <= 100)
                ]
                if len(spo2_clean) > 0:
                    spo2_mu = np.mean(spo2_clean)
                    spo2_std = np.std(spo2_clean)
                    spo2_med = np.median(spo2_clean)
                else:
                    spo2_mu = spo2_std = spo2_med = np.nan
                spo2_qos = (
                    np.mean(oxstatus_win < 2.8)
                    if oxstatus_win is not None
                    else 1.0
                )
            else:
                spo2_mu = spo2_std = spo2_med = np.nan
                spo2_qos = 0.0

            # Movement features (use high-frequency PPG content as proxy)
            mov_signal = _extract_movement(pleth_win, sample_rate)
            mov_mu = np.mean(mov_signal)
            mov_std = np.std(mov_signal)
            mov_med = np.median(mov_signal)

            # --- Circadian features ---
            # Time of day as sinusoidal encoding (assumes recording starts ~evening)
            if psg_start_time is not None:
                # psg_start_time is in seconds since midnight
                current_time_sec = (psg_start_time + center_time_sec) % 86400
            else:
                current_time_sec = center_time_sec % 86400

            hour_frac = current_time_sec / 86400.0
            tod_cos = np.cos(2 * np.pi * hour_frac)
            tod_sin = np.sin(2 * np.pi * hour_frac)

            # --- Actigraphy CSV features ---
            if actig_data is not None:
                # Map PSG window center to actigraphy epoch
                actig_epoch_idx = int(center_time_sec / actig_epoch_sec)
                # Use a window of actigraphy epochs around center
                actig_half_win = max(1, int(win_sec / actig_epoch_sec / 2))
                actig_start = max(0, actig_epoch_idx - actig_half_win)
                actig_end = min(len(actig_data), actig_epoch_idx + actig_half_win + 1)
                actig_win = actig_data[actig_start:actig_end]

                if len(actig_win) > 0:
                    act_mean = np.mean(actig_win)
                    act_std = np.std(actig_win)
                    act_pct_zero = np.mean(actig_win == 0)
                    act_log_mean = np.log1p(np.mean(actig_win))
                else:
                    act_mean = act_std = act_pct_zero = act_log_mean = np.nan
            else:
                act_mean = act_std = act_pct_zero = act_log_mean = np.nan

            # Assemble feature vector
            features[w] = [
                hr_bpm, mean_nn, sd_nn, median_nn, lfhf,
                spo2_mu, spo2_std, spo2_med,
                mov_mu, mov_std, mov_med,
                rsp_bpm, spo2_qos, hrv_qos,
                rmssd, pnn50,
                tod_cos, tod_sin,
                act_mean, act_std, act_pct_zero, act_log_mean,
            ]

            # Assign label from 30-sec epoch annotations
            # Each window of win_sec seconds with 50% overlap → center epoch
            epoch_idx_30s = int(center_time_sec / 30)
            if 0 <= epoch_idx_30s < len(stage_labels_30s):
                labels[w] = stage_labels_30s[epoch_idx_30s]
            else:
                labels[w] = 6  # noise/unknown

            # Mask: valid if we have enough good RR intervals and no NaN in key features
            mask[w] = 1 if (len(valid_rr) >= 3 and not np.any(np.isnan(features[w, :14]))) else 0

        # Save HDF5
        with h5py.File(str(output_file), "w") as h5:
            h5.create_dataset(
                "features", data=features, compression="gzip", compression_opts=6
            )
            h5.create_dataset(
                "stage_labels", data=labels, compression="gzip", compression_opts=6
            )
            h5.create_dataset(
                "mask", data=mask, compression="gzip", compression_opts=6
            )

        valid_pct = np.mean(mask) * 100
        logger.info(
            "Generated %s/%s: %d windows, %.1f%% valid",
            ds_name,
            subject_id,
            num_windows,
            valid_pct,
        )


# =============================================================================
# Signal loading helpers
# =============================================================================


def _load_psg_signals(
    ds_path: Path, subject_id: str, target_rate: int
) -> tuple[dict[str, np.ndarray], dict[str, int], float | None, int]:
    """Load wearable-type signals from PSG EDF file.

    Returns:
        signals: Dict mapping canonical name -> signal array (resampled to target_rate)
        channel_map: Dict mapping canonical name -> original channel index
        start_time_sec: Recording start time in seconds since midnight, or None
        original_rate: Original sampling rate
    """
    import pyedflib

    edf_dir = ds_path / "polysomnography" / "edfs"
    edf_files = list(edf_dir.glob(f"*{subject_id}*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF file for {subject_id} in {edf_dir}")

    f = pyedflib.EdfReader(str(edf_files[0]))
    try:
        n_channels = f.signals_in_file
        channel_labels = [f.getLabel(i) for i in range(n_channels)]

        # Map channels
        channel_map = {}
        for i, label in enumerate(channel_labels):
            canonical = CHANNEL_MAP.get(label.strip())
            if canonical and canonical not in channel_map:
                channel_map[canonical] = i

        # Get start time
        start_datetime = f.getStartdatetime()
        start_time_sec = (
            start_datetime.hour * 3600
            + start_datetime.minute * 60
            + start_datetime.second
        )

        # Read signals
        signals = {}
        original_rate = target_rate
        for canonical, idx in channel_map.items():
            sig_rate = int(f.getSampleFrequency(idx))
            data = f.readSignal(idx)

            # Resample to target rate if needed
            if sig_rate != target_rate:
                n_target = int(len(data) * target_rate / sig_rate)
                data = np.interp(
                    np.linspace(0, len(data) - 1, n_target),
                    np.arange(len(data)),
                    data,
                )
            if canonical == "pleth":
                original_rate = sig_rate

            signals[canonical] = data.astype(np.float32)

        return signals, channel_map, start_time_sec, original_rate
    finally:
        f.close()


def _load_annotations(
    ds_path: Path, subject_id: str
) -> tuple[np.ndarray, float | None]:
    """Load sleep stage annotations from XML file.

    Returns:
        labels: Array of integer stage labels, one per 30-sec epoch
        start_time: Annotation start time in seconds since midnight, or None
    """
    from lxml import etree

    annot_dir = ds_path / "polysomnography" / "annotations-events-nsrr"
    xml_files = list(annot_dir.glob(f"*{subject_id}*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No annotation file for {subject_id}")

    tree = etree.parse(str(xml_files[0]))
    root = tree.getroot()

    # Parse scored events for sleep stages
    epochs = {}  # epoch_index -> stage_label

    for event in root.iter():
        tag = event.tag.split("}")[-1] if "}" in event.tag else event.tag
        if tag != "ScoredEvent":
            continue

        event_data = {}
        for child in event:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            event_data[child_tag] = child.text

        event_type = event_data.get("EventType", "")
        event_concept = event_data.get("EventConcept", "")

        if "Stages" not in event_type:
            continue

        start_sec = float(event_data.get("Start", 0))
        duration_sec = float(event_data.get("Duration", 30))

        # Map event concept to stage label
        stage_label = STAGE_MAP.get(event_concept)
        if stage_label is None:
            # Try partial match
            for key, val in STAGE_MAP.items():
                if key.split("|")[0] in event_concept:
                    stage_label = val
                    break
        if stage_label is None:
            stage_label = 6  # noise/unknown

        # Assign to 30-sec epochs covered by this event
        epoch_start = int(start_sec / 30)
        n_epochs = max(1, int(duration_sec / 30))
        for e in range(epoch_start, epoch_start + n_epochs):
            epochs[e] = stage_label

    if not epochs:
        raise ValueError(f"No sleep stage events found for {subject_id}")

    max_epoch = max(epochs.keys()) + 1
    labels = np.full(max_epoch, 6, dtype=np.int32)  # default: noise
    for idx, stage in epochs.items():
        labels[idx] = stage

    return labels, None


def _load_actigraphy(
    ds_path: Path, subject_id: str, ds_name: str
) -> tuple[np.ndarray, int]:
    """Load actigraphy activity counts from CSV.

    Returns:
        activity: Array of activity count values
        epoch_sec: Epoch duration in seconds (30 for MESA, 60 for STAGES)
    """
    import pandas as pd

    actig_dir = ds_path / "actigraphy"
    csv_files = list(actig_dir.glob(f"*{subject_id}*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No actigraphy CSV for {subject_id}")

    df = pd.read_csv(csv_files[0])

    # Find activity column
    activity_col = None
    for col_name in ["activity", "Activity", "ACTIVITY"]:
        if col_name in df.columns:
            activity_col = col_name
            break

    if activity_col is None:
        raise ValueError(f"No activity column in {csv_files[0].name}")

    activity = df[activity_col].values.astype(np.float32)

    # Filter Huami Arc artifacts for STAGES
    if ds_name == "stages":
        activity[activity == 20] = 0  # Known artifact value

    epoch_sec = 60 if ds_name == "stages" else 30

    return activity, epoch_sec


# =============================================================================
# Signal processing helpers
# =============================================================================


def _detect_rr_intervals(pleth: np.ndarray, sample_rate: int) -> np.ndarray:
    """Detect RR intervals from PPG signal using peak detection."""
    from scipy.signal import butter, filtfilt, find_peaks

    # Bandpass filter the PPG signal
    nyq = sample_rate / 2
    low = PPG_LOWCUT / nyq
    high = min(PPG_HIGHCUT / nyq, 0.99)
    b, a = butter(3, [low, high], btype="band")
    filtered = filtfilt(b, a, pleth)

    # Detect peaks (PPG pulses)
    min_distance = int(sample_rate * RR_MIN)
    peaks, _ = find_peaks(filtered, distance=min_distance, prominence=0.1 * np.std(filtered))

    if len(peaks) < 2:
        return np.array([])

    # Compute RR intervals in seconds
    rr = np.diff(peaks) / sample_rate
    return rr.astype(np.float32)


def _compute_lfhf_ratio(rr_intervals: np.ndarray) -> float:
    """Compute LF/HF power ratio from RR intervals using FFT."""
    if len(rr_intervals) < 8:
        return np.nan

    # Interpolate RR intervals to uniform 4 Hz sampling
    rr_times = np.cumsum(rr_intervals)
    interp_rate = 4.0
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / interp_rate)
    rr_uniform = np.interp(t_uniform, rr_times, rr_intervals)

    # Remove mean
    rr_uniform -= np.mean(rr_uniform)

    # FFT
    n = len(rr_uniform)
    freqs = np.fft.rfftfreq(n, d=1.0 / interp_rate)
    power = np.abs(np.fft.rfft(rr_uniform)) ** 2 / n

    # Compute band powers
    lf_mask = (freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])
    hf_mask = (freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])

    lf_power = np.sum(power[lf_mask])
    hf_power = np.sum(power[hf_mask])

    if hf_power < 1e-10:
        return np.nan

    return float(lf_power / hf_power)


def _estimate_respiratory_rate(rr_intervals: np.ndarray) -> float:
    """Estimate respiratory rate from RR interval variability (RIFV method)."""
    if len(rr_intervals) < 8:
        return np.nan

    # Interpolate to uniform sampling
    rr_times = np.cumsum(rr_intervals)
    interp_rate = 4.0
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / interp_rate)
    rr_uniform = np.interp(t_uniform, rr_times, rr_intervals)

    # Remove mean and apply FFT
    rr_uniform -= np.mean(rr_uniform)
    n = len(rr_uniform)
    freqs = np.fft.rfftfreq(n, d=1.0 / interp_rate)
    power = np.abs(np.fft.rfft(rr_uniform)) ** 2

    # Find peak in respiratory band (0.15-0.5 Hz = 9-30 breaths/min)
    resp_mask = (freqs >= 0.15) & (freqs <= 0.5)
    if not np.any(resp_mask):
        return np.nan

    resp_freqs = freqs[resp_mask]
    resp_power = power[resp_mask]
    peak_idx = np.argmax(resp_power)

    return float(resp_freqs[peak_idx] * 60)  # Convert Hz to breaths/min


def _extract_movement(pleth: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract movement proxy from high-frequency PPG content."""
    from scipy.signal import butter, filtfilt

    # High-pass filter to get motion artifact (>3 Hz)
    nyq = sample_rate / 2
    high = min(3.0 / nyq, 0.99)
    b, a = butter(3, high, btype="high")
    motion = filtfilt(b, a, pleth)

    # Return envelope (absolute value)
    return np.abs(motion).astype(np.float32)
