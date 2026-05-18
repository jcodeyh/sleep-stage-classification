#!/usr/bin/env python3
"""Probe STAGES EDF files to discover channel names and sample rates.

Scans a directory of EDF files, extracts all channel labels and sample rates,
and produces a frequency table showing which channels appear across subjects.
This is essential for building the STAGES channel label map before processing.

Usage:
    python scripts/probe_stages_channels.py --edf-dir ./datasets/stages/polysomnography/edfs
    python scripts/probe_stages_channels.py --edf-dir ./datasets/stages/polysomnography/edfs --max-files 50
"""

import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pyedflib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def probe_edf(edf_path: Path) -> list[dict]:
    """Extract channel labels and sample rates from a single EDF file.

    Returns list of dicts with keys: label, sample_rate, physical_dim, n_samples.
    """
    channels = []
    try:
        reader = pyedflib.EdfReader(str(edf_path))
        n_channels = reader.signals_in_file
        for i in range(n_channels):
            channels.append({
                "label": reader.getLabel(i).strip(),
                "sample_rate": int(reader.getSampleFrequency(i)),
                "physical_dim": reader.getPhysicalDimension(i).strip(),
                "n_samples": reader.getNSamples()[i],
            })
        reader.close()
    except Exception as e:
        logger.warning("Failed to read %s: %s", edf_path.name, e)
    return channels


def extract_site(filename: str) -> str:
    """Attempt to extract site prefix from STAGES filename.

    STAGES filenames may follow patterns like:
      STAGES_stanford_12345.edf
      SC-STGS-0001.edf
    Returns the best guess at a site identifier.
    """
    stem = Path(filename).stem.lower()
    # Common STAGES site prefixes
    known_sites = [
        "stanford", "mayo", "medsleep", "stlukes", "st_lukes",
        "geisinger", "bogansleep", "bogan",
    ]
    for site in known_sites:
        if site in stem:
            return site
    # Fall back to first segment before a digit
    parts = stem.replace("-", "_").split("_")
    for part in parts:
        if not part.isdigit() and len(part) > 1:
            return part
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Probe STAGES EDF files for channel names and sample rates"
    )
    parser.add_argument(
        "--edf-dir",
        type=Path,
        required=True,
        help="Directory containing EDF files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./logs/stages_channel_survey.csv"),
        help="Output CSV file for channel survey",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of EDFs to probe (0 = all)",
    )
    args = parser.parse_args()

    edf_files = sorted(args.edf_dir.glob("*.edf"))
    if not edf_files:
        logger.error("No EDF files found in %s", args.edf_dir)
        return

    if args.max_files > 0:
        edf_files = edf_files[: args.max_files]

    logger.info("Probing %d EDF files in %s", len(edf_files), args.edf_dir)

    # Collect data
    channel_counts: Counter = Counter()
    channel_rates: defaultdict = defaultdict(Counter)
    channel_dims: defaultdict = defaultdict(Counter)
    site_channels: defaultdict = defaultdict(lambda: Counter())
    per_file_rows: list[dict] = []
    total_files = 0
    failed_files = 0

    for edf_path in edf_files:
        channels = probe_edf(edf_path)
        if not channels:
            failed_files += 1
            continue
        total_files += 1
        site = extract_site(edf_path.name)

        for ch in channels:
            label = ch["label"]
            channel_counts[label] += 1
            channel_rates[label][ch["sample_rate"]] += 1
            channel_dims[label][ch["physical_dim"]] += 1
            site_channels[site][label] += 1
            per_file_rows.append({
                "file": edf_path.name,
                "site": site,
                "label": label,
                "sample_rate": ch["sample_rate"],
                "physical_dim": ch["physical_dim"],
                "n_samples": ch["n_samples"],
            })

    # Write per-file detail CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "site", "label", "sample_rate", "physical_dim", "n_samples"]
        )
        writer.writeheader()
        writer.writerows(per_file_rows)

    # Write summary
    summary_path = args.output.with_suffix(".summary.txt")
    lines = [
        f"STAGES EDF Channel Survey",
        f"{'=' * 60}",
        f"Files probed: {total_files} (failed: {failed_files})",
        f"Unique channel labels: {len(channel_counts)}",
        f"",
        f"{'Channel Label':<30} {'Count':>6} {'%':>6}  {'Sample Rates':>20}  {'Units'}",
        f"{'-' * 30} {'-' * 6} {'-' * 6}  {'-' * 20}  {'-' * 10}",
    ]

    for label, count in channel_counts.most_common():
        pct = 100.0 * count / total_files
        rates = ", ".join(f"{r}Hz({c})" for r, c in channel_rates[label].most_common())
        dims = ", ".join(f"{d}({c})" for d, c in channel_dims[label].most_common(3))
        lines.append(f"{label:<30} {count:>6} {pct:>5.1f}%  {rates:>20}  {dims}")

    lines.extend([
        "",
        "Channels by site:",
        f"{'-' * 60}",
    ])
    for site in sorted(site_channels.keys()):
        n_files_site = sum(1 for r in per_file_rows if r["site"] == site) // max(
            1, len(set(r["label"] for r in per_file_rows if r["site"] == site))
        )
        labels = [lbl for lbl, _ in site_channels[site].most_common(20)]
        lines.append(f"  {site} (~{n_files_site} files): {', '.join(labels)}")

    # Wearable-relevant channel detection
    lines.extend([
        "",
        "Wearable-relevant channel detection:",
        f"{'-' * 60}",
    ])
    hr_labels = [l for l in channel_counts if l.lower() in (
        "hr", "heart rate", "pulse", "pulse rate", "pr"
    )]
    dhr_labels = [l for l in channel_counts if l.lower() in ("dhr",)]
    spo2_labels = [l for l in channel_counts if l.lower() in (
        "spo2", "sao2", "sp02",
    )]
    ecg_labels = [l for l in channel_counts if "ecg" in l.lower() or "ekg" in l.lower()]
    ppg_labels = [l for l in channel_counts if l.lower() in (
        "pleth", "ppg", "pulse",
    )]

    lines.append(f"  HR channels found: {hr_labels or 'NONE'}")
    lines.append(f"  DHR channels found: {dhr_labels or 'NONE'}")
    lines.append(f"  SpO2 channels found: {spo2_labels or 'NONE'}")
    lines.append(f"  ECG channels found: {ecg_labels or 'NONE (may need PPG fallback)'}")
    lines.append(f"  PPG/Pleth channels found: {ppg_labels or 'NONE'}")

    if not hr_labels and not dhr_labels:
        lines.append("  WARNING: No pre-computed HR/DHR channels found!")
        if ecg_labels:
            lines.append("  -> Can derive HR from ECG via R-peak detection")
        elif ppg_labels:
            lines.append("  -> Can derive HR from PPG/Pleth via peak detection")
        else:
            lines.append("  -> NO cardiac signal found — HR features will be NaN")

    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)

    print(summary_text)
    logger.info("Detail CSV written to: %s", args.output)
    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
