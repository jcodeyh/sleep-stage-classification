#!/usr/bin/env python3
"""Validate downloaded STAGES and MESA data quality.

Checks file integrity, signal quality, and temporal alignment before feature
extraction. Produces a per-subject quality report and a filtered subject list.

Usage:
    python scripts/validate_data.py \
        --datasets-dir ./datasets \
        --output ./logs/data_quality_report.csv \
        --valid-subjects-output ./logs/valid_subjects.txt
"""

import argparse
import csv
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Known bad data
STAGES_MAYO_SITES = {"mayo"}
MIN_RECORDING_HOURS = 2.0
MIN_OVERLAP_HOURS = 4.0
MIN_ACTIG_NONZERO_FRAC = 0.01  # At least 1% non-zero epochs
SPO2_VALID_RANGE = (50, 100)
STAGES_HUAMI_ARTIFACT_VALUE = 20  # Known artifact in Huami Arc actigraphy

# Channel name mappings for cross-site heterogeneity in STAGES
CHANNEL_MAP = {
    # SpO2 variants
    "spo2": "spo2",
    "sao2": "spo2",
    "SAO2": "spo2",
    "SpO2": "spo2",
    "SaO2": "spo2",
    # Pleth/PPG variants
    "pleth": "pleth",
    "Pleth": "pleth",
    "PPG": "pleth",
    "ppg": "pleth",
    "PLETH": "pleth",
    # Heart rate variants
    "hr": "hr",
    "HR": "hr",
    "Heart Rate": "hr",
    "Pulse": "hr",
    # OxStatus variants
    "oxstatus": "oxstatus",
    "OxStatus": "oxstatus",
    "OXSTATUS": "oxstatus",
}


@dataclass
class SubjectReport:
    subject_id: str = ""
    dataset: str = ""
    has_actigraphy: bool = False
    has_psg: bool = False
    has_annotations: bool = False
    has_pleth: bool = False
    has_spo2: bool = False
    has_hr: bool = False
    psg_duration_hours: float = 0.0
    actig_duration_hours: float = 0.0
    overlap_hours: float = 0.0
    spo2_in_range: bool = True
    actig_nonzero_frac: float = 0.0
    pleth_quality: str = "unknown"
    annotation_has_stages: bool = False
    is_mayo_site: bool = False
    is_duplicate: bool = False
    quality_score: float = 0.0
    include: bool = False
    exclusion_reason: str = ""
    unmapped_channels: str = ""


def validate_edf(edf_path: Path, report: SubjectReport) -> dict[str, int]:
    """Validate an EDF file and return channel name -> index mapping."""
    try:
        import pyedflib

        f = pyedflib.EdfReader(str(edf_path))
    except Exception as e:
        report.exclusion_reason = f"EDF read error: {e}"
        return {}

    try:
        n_channels = f.signals_in_file
        channel_labels = [f.getLabel(i) for i in range(n_channels)]

        # Map channel names to canonical names
        mapped = {}
        unmapped = []
        for i, label in enumerate(channel_labels):
            canonical = CHANNEL_MAP.get(label.strip())
            if canonical:
                mapped[canonical] = i
            else:
                unmapped.append(label.strip())

        report.has_pleth = "pleth" in mapped
        report.has_spo2 = "spo2" in mapped
        report.has_hr = "hr" in mapped

        if unmapped:
            report.unmapped_channels = "; ".join(unmapped[:10])

        # Check recording duration
        duration_sec = f.getFileDuration()
        report.psg_duration_hours = duration_sec / 3600.0

        # Validate Pleth signal quality (check for flat/clipped)
        if report.has_pleth:
            pleth_idx = mapped["pleth"]
            sample_rate = f.getSampleFrequency(pleth_idx)
            # Read first 60 seconds for quick quality check
            n_samples = min(int(sample_rate * 60), f.getNSamples()[pleth_idx])
            if n_samples > 0:
                pleth_data = f.readSignal(pleth_idx, 0, n_samples)
                if np.std(pleth_data) < 1e-6:
                    report.pleth_quality = "flat"
                elif np.mean(pleth_data == np.max(pleth_data)) > 0.5:
                    report.pleth_quality = "clipped"
                else:
                    report.pleth_quality = "ok"

        # Validate SpO2 range
        if report.has_spo2:
            spo2_idx = mapped["spo2"]
            sample_rate = f.getSampleFrequency(spo2_idx)
            n_samples = min(int(sample_rate * 60), f.getNSamples()[spo2_idx])
            if n_samples > 0:
                spo2_data = f.readSignal(spo2_idx, 0, n_samples)
                valid_mask = (spo2_data >= SPO2_VALID_RANGE[0]) & (
                    spo2_data <= SPO2_VALID_RANGE[1]
                )
                report.spo2_in_range = np.mean(valid_mask) > 0.5

        return mapped
    finally:
        f.close()


def validate_actigraphy(actig_path: Path, dataset: str, report: SubjectReport):
    """Validate an actigraphy file (CSV for MESA, JSON for STAGES)."""
    if dataset == "stages":
        # STAGES: Huami Arc JSON format
        # {"items": [{"date": "...", "minute": N, "activeness": N, "steps": N, "mode": N}, ...]}
        import json
        try:
            with open(actig_path) as f:
                data = json.load(f)
            items = data.get("items", [])
        except Exception as e:
            report.exclusion_reason = f"Actigraphy JSON read error: {e}"
            return

        report.has_actigraphy = True
        activity = np.array([float(item.get("activeness", 0)) for item in items])
        n_records = len(items)
    else:
        # MESA: Actiwatch CSV format
        import pandas as pd
        try:
            df = pd.read_csv(actig_path)
        except Exception as e:
            report.exclusion_reason = f"Actigraphy CSV read error: {e}"
            return

        report.has_actigraphy = True

        activity_col = None
        for col_name in ["activity", "Activity", "ACTIVITY"]:
            if col_name in df.columns:
                activity_col = col_name
                break

        if activity_col is None:
            logger.warning(
                "No activity column found in %s. Columns: %s",
                actig_path.name,
                list(df.columns),
            )
            return

        activity = df[activity_col].values
        n_records = len(df)

    # Filter Huami Arc "20" artifact values for STAGES
    if dataset == "stages":
        artifact_mask = activity == STAGES_HUAMI_ARTIFACT_VALUE
        if np.sum(artifact_mask) > 0:
            logger.debug(
                "%s: Filtered %d Huami Arc artifact values",
                report.subject_id,
                np.sum(artifact_mask),
            )
            activity = activity[~artifact_mask]

    # Check non-zero fraction
    if len(activity) > 0:
        report.actig_nonzero_frac = float(np.mean(activity > 0))

    # Estimate duration: MESA=30-sec epochs, STAGES JSON=60-sec (1-min) epochs
    epoch_sec = 60 if dataset == "stages" else 30
    report.actig_duration_hours = n_records * epoch_sec / 3600.0


def validate_annotations(annot_path: Path, report: SubjectReport):
    """Validate annotation XML for sleep stage events."""
    try:
        from lxml import etree

        tree = etree.parse(str(annot_path))
        root = tree.getroot()
    except Exception as e:
        report.exclusion_reason = f"Annotation XML parse error: {e}"
        return

    report.has_annotations = True

    # Look for ScoredEvents with sleep stages
    stage_events = []
    ns = {"ns": root.nsmap.get(None, "")}

    # Try with namespace
    events = root.findall(".//ScoredEvent", ns) if ns.get("ns") else []
    if not events:
        events = root.findall(".//{*}ScoredEvent")
    if not events:
        events = root.findall(".//ScoredEvent")

    for event in events:
        event_type = None
        for child in event:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "EventType" and "Stages" in (child.text or ""):
                event_type = "stage"
            elif tag == "EventConcept" and child.text:
                text = child.text
                if any(
                    s in text
                    for s in [
                        "Wake",
                        "Stage 1",
                        "Stage 2",
                        "Stage 3",
                        "Stage 4",
                        "REM",
                    ]
                ):
                    event_type = "stage"
        if event_type == "stage":
            stage_events.append(event)

    report.annotation_has_stages = len(stage_events) > 0

    if not stage_events:
        report.exclusion_reason = "No sleep stage events in annotation XML"


def compute_quality_score(report: SubjectReport) -> float:
    """Compute a 0-1 quality score for inclusion decision."""
    score = 0.0

    # Essential signals (each worth 0.2)
    if report.has_pleth and report.pleth_quality == "ok":
        score += 0.2
    if report.has_spo2 and report.spo2_in_range:
        score += 0.2
    if report.has_actigraphy and report.actig_nonzero_frac >= MIN_ACTIG_NONZERO_FRAC:
        score += 0.2
    if report.annotation_has_stages:
        score += 0.2

    # Duration/overlap bonus (worth 0.2)
    if report.psg_duration_hours >= MIN_RECORDING_HOURS:
        score += 0.1
    if report.actig_duration_hours >= MIN_RECORDING_HOURS:
        score += 0.1

    return round(score, 2)


def validate_subject(
    subject_id: str,
    dataset: str,
    datasets_dir: Path,
) -> SubjectReport:
    """Validate all data for a single subject."""
    report = SubjectReport(subject_id=subject_id, dataset=dataset)

    # Check for Mayo site exclusion (STAGES only)
    if dataset == "stages":
        if any(site in subject_id.lower() for site in STAGES_MAYO_SITES):
            report.is_mayo_site = True
            report.exclusion_reason = "Mayo clinic site (known conversion errors)"
            return report

    ds_dir = datasets_dir / dataset

    # Find and validate EDF
    edf_dir = ds_dir / "polysomnography" / "edfs"
    edf_files = list(edf_dir.glob(f"*{subject_id}*.edf")) if edf_dir.exists() else []
    if edf_files:
        report.has_psg = True
        validate_edf(edf_files[0], report)
    else:
        report.has_psg = False

    # Find and validate actigraphy
    if dataset == "stages":
        # STAGES: Huami Arc JSON in actigraphy_json/{subject_id}/minbymin/
        actig_json_dir = ds_dir / "actigraphy_json" / subject_id / "minbymin"
        json_files = list(actig_json_dir.glob("*.json")) if actig_json_dir.exists() else []
        if json_files:
            validate_actigraphy(json_files[0], dataset, report)
        else:
            report.has_actigraphy = False
    else:
        actig_dir = ds_dir / "actigraphy"
        actig_files = (
            list(actig_dir.glob(f"*{subject_id}*.csv")) if actig_dir.exists() else []
        )
        if actig_files:
            validate_actigraphy(actig_files[0], dataset, report)
        else:
            report.has_actigraphy = False

    # Find and validate annotations
    annot_dir = ds_dir / "polysomnography" / "annotations-events-nsrr"
    annot_files = (
        list(annot_dir.glob(f"*{subject_id}*.xml")) if annot_dir.exists() else []
    )
    if annot_files:
        validate_annotations(annot_files[0], report)
    else:
        report.has_annotations = False

    # Compute quality score
    report.quality_score = compute_quality_score(report)

    # Determine inclusion
    if not report.exclusion_reason:
        if not report.has_psg:
            report.exclusion_reason = "Missing PSG EDF file"
        elif not report.has_actigraphy:
            report.exclusion_reason = "Missing actigraphy CSV"
        elif not report.has_annotations:
            report.exclusion_reason = "Missing annotation XML"
        elif not report.annotation_has_stages:
            report.exclusion_reason = "No sleep stage events in annotations"
        elif report.psg_duration_hours < MIN_RECORDING_HOURS:
            report.exclusion_reason = (
                f"PSG too short ({report.psg_duration_hours:.1f}h < {MIN_RECORDING_HOURS}h)"
            )
        elif not report.has_pleth:
            report.exclusion_reason = "Missing Pleth/PPG channel in EDF"
        elif report.pleth_quality != "ok":
            report.exclusion_reason = f"Pleth signal quality: {report.pleth_quality}"
        elif report.is_mayo_site:
            report.exclusion_reason = "Mayo clinic site excluded"

    report.include = report.exclusion_reason == ""

    return report


def discover_subjects(datasets_dir: Path) -> list[tuple[str, str]]:
    """Discover all subjects from downloaded data files."""
    subjects = []

    for dataset in ("mesa", "stages"):
        ds_dir = datasets_dir / dataset

        # Find subjects from annotation files (most reliable)
        annot_dir = ds_dir / "polysomnography" / "annotations-events-nsrr"
        if annot_dir.exists():
            for xml_file in annot_dir.glob("*.xml"):
                name = xml_file.stem
                # Clean up common suffixes
                name = name.replace("-nsrr", "").replace("_nsrr", "")
                subjects.append((name, dataset))

        # Also check actigraphy dir for subjects without annotations
        actig_dir = ds_dir / "actigraphy"
        if actig_dir.exists():
            existing_ids = {s[0] for s in subjects if s[1] == dataset}
            for csv_file in actig_dir.glob("*.csv"):
                name = csv_file.stem
                name = name.replace("-actigraphy", "").replace("_actigraphy", "")
                if name not in existing_ids:
                    subjects.append((name, dataset))

    return subjects


def main():
    parser = argparse.ArgumentParser(description="Validate STAGES/MESA data quality")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        required=True,
        help="Base directory with mesa/ and stages/ subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for quality report",
    )
    parser.add_argument(
        "--valid-subjects-output",
        type=Path,
        default=None,
        help="Output text file with valid subject IDs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Discover all subjects
    subjects = discover_subjects(args.datasets_dir)
    logger.info("Discovered %d subjects across all datasets", len(subjects))

    # Validate each subject
    reports = []
    for i, (subject_id, dataset) in enumerate(subjects):
        if (i + 1) % 50 == 0:
            logger.info("Validating subject %d/%d...", i + 1, len(subjects))
        report = validate_subject(subject_id, dataset, args.datasets_dir)
        reports.append(report)

    # Write CSV report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SubjectReport.__dataclass_fields__.keys())

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow(asdict(report))

    # Count results
    included = [r for r in reports if r.include]
    excluded = [r for r in reports if not r.include]
    mesa_included = [r for r in included if r.dataset == "mesa"]
    stages_included = [r for r in included if r.dataset == "stages"]

    # Write valid subjects file
    if args.valid_subjects_output:
        args.valid_subjects_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.valid_subjects_output, "w") as f:
            for r in included:
                f.write(f"{r.dataset}/{r.subject_id}\n")

    # Write summary
    summary_path = args.output.parent / "valid_subjects_summary.txt"
    exclusion_reasons = {}
    for r in excluded:
        reason = r.exclusion_reason or "Unknown"
        exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    summary = (
        f"Data Validation Summary\n"
        f"{'=' * 50}\n"
        f"Total subjects discovered: {len(reports)}\n"
        f"  MESA:   {len([r for r in reports if r.dataset == 'mesa'])}\n"
        f"  STAGES: {len([r for r in reports if r.dataset == 'stages'])}\n"
        f"\n"
        f"Included (passed validation): {len(included)}\n"
        f"  MESA:   {len(mesa_included)}\n"
        f"  STAGES: {len(stages_included)}\n"
        f"\n"
        f"Excluded: {len(excluded)}\n"
    )
    for reason, count in sorted(exclusion_reasons.items(), key=lambda x: -x[1]):
        summary += f"  {reason}: {count}\n"

    summary_path.write_text(summary)
    logger.info("\n%s", summary)
    logger.info("Quality report written to: %s", args.output)


if __name__ == "__main__":
    main()
