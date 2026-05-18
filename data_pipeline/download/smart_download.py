#!/usr/bin/env python3
"""Smart selective download for STAGES and MESA datasets from NSRR.

Downloads only subjects with overlapping actigraphy + PSG data, in stages:
  1. metadata   - Discover valid subjects from NSRR API (~5 MB)
  2. actigraphy - Actigraphy CSVs/ZIPs + annotation XMLs + demographics (~5-20 GB)
  3. edfs       - PSG EDF files for valid subjects (~100-500 GB)

STAGES data on NSRR is organized differently from MESA:
  - EDFs are under original/STAGES PSGs/{SITE_CODE}/ (13 sites)
  - Actigraphy is a single ZIP: original/De-identified Data/Actigraphy Final to Post.zip
  - Demographics: datasets/stages-dataset-0.3.0.csv
  - Annotations: may be in polysomnography/annotations-events-nsrr/ or EDF+ embedded

Usage:
    # Stage 1: metadata (run on login node)
    python scripts/smart_download.py --stage metadata --dataset stages

    # Stage 2: actigraphy + annotations + demographics (PBS job)
    python scripts/smart_download.py --stage actigraphy --dataset stages

    # Stage 3: PSG EDFs (PBS overnight job)
    python scripts/smart_download.py --stage edfs --dataset stages

Requires NSRR_TOKEN environment variable (get from https://sleepdata.org/account).
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

NSRR_API_BASE = "https://sleepdata.org/api/v1/"

# Known bad data to exclude
STAGES_MAYO_SITES = {"mayo"}  # Known conversion errors, removed June 2023
STAGES_DUPLICATE_SUBJECTS = set()  # Populated from metadata if found

# STAGES site codes — EDFs live under original/STAGES PSGs/{SITE_CODE}/
STAGES_SITE_CODES = [
    "BOGN", "GSBB", "GSDV", "GSLH", "GSSA", "GSSW",
    "MSMI", "MSNF", "MSQW", "MSTH", "MSTR", "STLK", "STNF",
]

# MESA actigraphy file pattern: mesa-sleep-{id}-actigraphy.csv
# MESA annotation pattern: mesa-sleep-{id}-nsrr.xml
# STAGES EDFs: original/STAGES PSGs/{SITE}/{subject_id}.edf
# STAGES actigraphy: single ZIP at original/De-identified Data/Actigraphy Final to Post.zip
# STAGES demographics: datasets/stages-dataset-0.3.0.csv


def get_nsrr_token() -> str:
    """Get NSRR API token from environment or file."""
    token = os.environ.get("NSRR_TOKEN", "")
    if not token:
        token_file = Path.home() / ".nsrr_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        raise RuntimeError(
            "NSRR_TOKEN not set. Export it or create .nsrr_token file.\n"
            "Get your token from: https://sleepdata.org/account"
        )
    return token


def nsrr_api_get(endpoint: str, token: str, params: dict | None = None) -> dict:
    """Make authenticated GET request to NSRR API."""
    url = urljoin(NSRR_API_BASE, endpoint)
    headers = {"Authorization": f"Token token={token}"}
    params = params or {}
    params["auth_token"] = token

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                logger.warning("API request failed (attempt %d/3): %s", attempt + 1, e)
                time.sleep(5 * (attempt + 1))
            else:
                raise


def nsrr_download_file(
    dataset: str, file_path: str, dest_path: Path, token: str
) -> bool:
    """Download a single file from NSRR."""
    url = f"https://sleepdata.org/datasets/{dataset}/files/{file_path}"
    params = {"auth_token": token}

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded and non-empty
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.debug("Already exists, skipping: %s", dest_path)
        return True

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, stream=True, timeout=300)
            resp.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.debug("Downloaded: %s", dest_path.name)
            return True
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                logger.warning(
                    "Download failed (attempt %d/3) for %s: %s",
                    attempt + 1,
                    file_path,
                    e,
                )
                time.sleep(5 * (attempt + 1))
            else:
                logger.error("Failed to download %s after 3 attempts: %s", file_path, e)
                return False


def list_nsrr_files(dataset: str, folder: str, token: str) -> list[dict]:
    """List files in an NSRR dataset folder."""
    endpoint = f"datasets/{dataset}/files.json"
    params = {"path": folder}
    data = nsrr_api_get(endpoint, token, params)
    return data if isinstance(data, list) else data.get("files", [])


# =============================================================================
# Stage 1: Metadata download
# =============================================================================


def download_metadata(
    datasets_dir: Path,
    log_dir: Path,
    token: str,
    dataset: str = "both",
    max_subjects: int = 0,
):
    """Download dataset metadata and identify valid subjects."""
    log_dir.mkdir(parents=True, exist_ok=True)

    valid_mesa: set[str] = set()
    valid_stages: set[str] = set()

    if dataset in ("mesa", "both"):
        valid_mesa = identify_mesa_subjects(datasets_dir, token)
        if max_subjects > 0:
            valid_mesa = set(sorted(valid_mesa)[:max_subjects])
        mesa_file = log_dir / "valid_subjects_mesa.txt"
        mesa_file.write_text("\n".join(sorted(valid_mesa)) + "\n")

    if dataset in ("stages", "both"):
        valid_stages = identify_stages_subjects(datasets_dir, token)
        if max_subjects > 0:
            valid_stages = set(sorted(valid_stages)[:max_subjects])
        stages_file = log_dir / "valid_subjects_stages.txt"
        stages_file.write_text("\n".join(sorted(valid_stages)) + "\n")

    # Summary
    lines = [f"Valid Subjects Summary", f"{'=' * 40}"]
    if dataset in ("mesa", "both"):
        lines.append(f"MESA:   {len(valid_mesa)} subjects with actigraphy + PSG annotations")
    if dataset in ("stages", "both"):
        lines.append(f"STAGES: {len(valid_stages)} subjects with actigraphy + PSG annotations")
    lines.append(f"Total:  {len(valid_mesa) + len(valid_stages)} subjects")
    if max_subjects > 0:
        lines.append(f"(Limited to {max_subjects} subjects per dataset)")

    summary = "\n".join(lines) + "\n"
    summary_file = log_dir / "valid_subjects_summary.txt"
    summary_file.write_text(summary)

    print(summary)
    return valid_mesa, valid_stages


def identify_mesa_subjects(datasets_dir: Path, token: str) -> set[str]:
    """Identify MESA subjects that have both actigraphy and PSG annotations."""
    logger.info("Identifying valid MESA subjects...")

    # List actigraphy files
    actig_files = list_nsrr_files("mesa", "actigraphy", token)
    actig_subjects = set()
    for f in actig_files:
        name = f.get("file_name", "") if isinstance(f, dict) else str(f)
        # Pattern: mesa-sleep-NNNN.csv (files are in the actigraphy/ folder)
        if name.endswith(".csv"):
            parts = name.replace(".csv", "").split("-")
            for p in parts:
                if p.isdigit() and len(p) == 4:
                    actig_subjects.add(p)
                    break

    # List annotation files
    annot_files = list_nsrr_files(
        "mesa", "polysomnography/annotations-events-nsrr", token
    )
    annot_subjects = set()
    for f in annot_files:
        name = f.get("file_name", "") if isinstance(f, dict) else str(f)
        if name.endswith("-nsrr.xml"):
            parts = name.replace("-nsrr.xml", "").split("-")
            for p in parts:
                if p.isdigit() and len(p) == 4:
                    annot_subjects.add(p)
                    break

    valid = actig_subjects & annot_subjects
    logger.info(
        "MESA: %d actigraphy, %d annotations, %d overlap",
        len(actig_subjects),
        len(annot_subjects),
        len(valid),
    )
    return valid


def identify_stages_subjects(datasets_dir: Path, token: str) -> set[str]:
    """Identify STAGES subjects from EDF listings across site folders.

    STAGES data on NSRR is organized by site under original/STAGES PSGs/{SITE}/.
    Actigraphy is in a single ZIP, so we identify subjects from EDFs and attempt
    to also check annotations if available.
    """
    logger.info("Identifying valid STAGES subjects from EDF listings...")

    edf_subjects = set()

    # List EDFs across all site folders
    for site in STAGES_SITE_CODES:
        folder = f"original/STAGES PSGs/{site}"
        try:
            files = list_nsrr_files("stages", folder, token)
            site_count = 0
            for f in files:
                name = f.get("file_name", "") if isinstance(f, dict) else str(f)
                if name.endswith(".edf"):
                    subj_id = name.replace(".edf", "")
                    # Skip known duplicates (e.g., STNF00003_1)
                    if "_" in subj_id and subj_id.split("_")[-1].isdigit():
                        logger.debug("Skipping duplicate variant: %s", name)
                        continue
                    edf_subjects.add(subj_id)
                    site_count += 1
            logger.info("  Site %s: %d EDF subjects", site, site_count)
        except Exception as e:
            logger.warning("  Failed to list site %s: %s", site, e)

    # Try to find annotations (NSRR may provide these in a standard path)
    annot_subjects = set()
    try:
        annot_files = list_nsrr_files(
            "stages", "polysomnography/annotations-events-nsrr", token
        )
        for f in annot_files:
            name = f.get("file_name", "") if isinstance(f, dict) else str(f)
            if name.endswith(".xml"):
                base = name.replace(".xml", "").replace("-nsrr", "").replace("_nsrr", "")
                annot_subjects.add(base)
        logger.info("  Found %d annotation XMLs in standard NSRR path", len(annot_subjects))
    except Exception:
        logger.info("  No annotations-events-nsrr folder found (may be EDF+ embedded)")

    # If annotations exist, intersect with EDFs; otherwise use all EDFs
    # (sleep stages may be embedded in EDF+ format)
    if annot_subjects:
        valid = edf_subjects & annot_subjects
        logger.info(
            "STAGES: %d EDFs, %d annotations, %d overlap",
            len(edf_subjects), len(annot_subjects), len(valid),
        )
    else:
        valid = edf_subjects
        logger.info(
            "STAGES: %d EDF subjects (annotations may be EDF+ embedded)",
            len(valid),
        )

    return valid


# =============================================================================
# Stage 2: Actigraphy + Annotations download
# =============================================================================


def download_actigraphy_and_labels(
    datasets_dir: Path,
    log_dir: Path,
    token: str,
    max_workers: int = 4,
    dataset: str = "both",
    max_subjects: int = 0,
):
    """Download actigraphy CSVs and annotation XMLs for valid subjects."""
    mesa_subjects: list[str] = []
    stages_subjects: list[str] = []
    if dataset in ("mesa", "both"):
        mesa_subjects = _load_subject_list(log_dir / "valid_subjects_mesa.txt")
        if max_subjects > 0:
            mesa_subjects = mesa_subjects[:max_subjects]
    if dataset in ("stages", "both"):
        stages_subjects = _load_subject_list(log_dir / "valid_subjects_stages.txt")
        if max_subjects > 0:
            stages_subjects = stages_subjects[:max_subjects]

    tasks = []

    # MESA actigraphy + annotations
    for subj_id in mesa_subjects:
        actig_remote = f"actigraphy/mesa-sleep-{subj_id}.csv"
        actig_local = datasets_dir / "mesa" / "actigraphy" / f"mesa-sleep-{subj_id}.csv"
        tasks.append(("mesa", actig_remote, actig_local))

        annot_remote = f"polysomnography/annotations-events-nsrr/mesa-sleep-{subj_id}-nsrr.xml"
        annot_local = (
            datasets_dir
            / "mesa"
            / "polysomnography"
            / "annotations-events-nsrr"
            / f"mesa-sleep-{subj_id}-nsrr.xml"
        )
        tasks.append(("mesa", annot_remote, annot_local))

    # STAGES: download actigraphy ZIP + demographics + annotation XMLs (if available)
    if stages_subjects:
        # Actigraphy: single ZIP file for all subjects
        actig_zip_remote = "original/De-identified Data/Actigraphy Final to Post.zip"
        actig_zip_local = (
            datasets_dir / "stages" / "original" / "Actigraphy Final to Post.zip"
        )
        tasks.append(("stages", actig_zip_remote, actig_zip_local))

        # Demographics CSVs
        for demo_file in [
            "datasets/stages-dataset-0.3.0.csv",
            "datasets/stages-harmonized-dataset-0.3.0.csv",
        ]:
            demo_local = datasets_dir / "stages" / demo_file.replace("/", os.sep)
            tasks.append(("stages", demo_file, demo_local))

        # Note: STAGES scoring annotations are CSVs next to EDFs, downloaded in --stage edfs

    logger.info("Downloading %d files (actigraphy + annotations + demographics)...", len(tasks))
    _parallel_download(tasks, token, max_workers)

    # Extract STAGES actigraphy ZIP if downloaded
    if stages_subjects:
        actig_zip = datasets_dir / "stages" / "original" / "Actigraphy Final to Post.zip"
        actig_out = datasets_dir / "stages" / "actigraphy_json"
        if actig_zip.exists():
            _extract_stages_actigraphy(actig_zip, actig_out)
        else:
            logger.warning("STAGES actigraphy ZIP not found at %s", actig_zip)


def _extract_stages_actigraphy(zip_path: Path, output_dir: Path):
    """Extract STAGES actigraphy ZIP preserving per-subject JSON structure.

    ZIP contains: Actigraphy/Final to Post/{SUBJECT_ID}/minbymin/min_{dates}.json
    Extracts to: output_dir/{SUBJECT_ID}/minbymin/*.json

    Uses system unzip because the ZIP uses Deflate64 compression (type 9)
    which Python's zipfile module does not support.
    """
    import subprocess

    logger.info("Extracting STAGES actigraphy from %s ...", zip_path.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use system unzip (handles Deflate64)
    result = subprocess.run(
        ["unzip", "-o", "-q", str(zip_path), "-d", str(output_dir)],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        logger.error("unzip failed: %s", result.stderr[:500])
        return

    # Move files from nested structure to flat per-subject directories
    nested = output_dir / "Actigraphy" / "Final to Post"
    if nested.exists():
        subject_count = 0
        already_count = 0
        for subdir in sorted(nested.iterdir()):
            if subdir.is_dir():
                target = output_dir / subdir.name
                if not target.exists():
                    subdir.rename(target)
                    subject_count += 1
                else:
                    already_count += 1
        # Clean up empty nested directories
        import shutil
        actig_top = output_dir / "Actigraphy"
        if actig_top.exists():
            shutil.rmtree(actig_top, ignore_errors=True)
        logger.info("Extracted %d subject actigraphy directories to %s (%d already existed)",
                     subject_count, output_dir, already_count)
    else:
        logger.warning("Expected nested structure not found in ZIP")


# =============================================================================
# Stage 3: EDF download
# =============================================================================


def download_edfs(
    datasets_dir: Path,
    log_dir: Path,
    token: str,
    max_workers: int = 4,
    dataset: str = "both",
    max_subjects: int = 0,
):
    """Download PSG EDF files for valid subjects."""
    mesa_subjects: list[str] = []
    stages_subjects: list[str] = []
    if dataset in ("mesa", "both"):
        mesa_subjects = _load_subject_list(log_dir / "valid_subjects_mesa.txt")
        if max_subjects > 0:
            mesa_subjects = mesa_subjects[:max_subjects]
    if dataset in ("stages", "both"):
        stages_subjects = _load_subject_list(log_dir / "valid_subjects_stages.txt")
        if max_subjects > 0:
            stages_subjects = stages_subjects[:max_subjects]

    tasks = []

    # MESA EDFs
    for subj_id in mesa_subjects:
        edf_remote = f"polysomnography/edfs/mesa-sleep-{subj_id}.edf"
        edf_local = (
            datasets_dir / "mesa" / "polysomnography" / "edfs" / f"mesa-sleep-{subj_id}.edf"
        )
        tasks.append(("mesa", edf_remote, edf_local))

    # STAGES EDFs + scoring CSVs — organized by site under original/STAGES PSGs/{SITE}/
    for subj_id in stages_subjects:
        # Extract site code from subject ID (e.g., "STNF" from "STNF00001")
        site = ""
        for sc in STAGES_SITE_CODES:
            if subj_id.startswith(sc):
                site = sc
                break
        if not site:
            logger.warning("Cannot determine site for subject %s, skipping EDF", subj_id)
            continue

        # EDF file — store locally in flat edfs/ directory
        edf_remote = f"original/STAGES PSGs/{site}/{subj_id}.edf"
        edf_local = datasets_dir / "stages" / "polysomnography" / "edfs" / f"{subj_id}.edf"
        tasks.append(("stages", edf_remote, edf_local))

        # Scoring CSV — sleep stage annotations are CSV files alongside EDFs
        score_remote = f"original/STAGES PSGs/{site}/{subj_id}.csv"
        score_local = (
            datasets_dir / "stages" / "polysomnography" / "annotations" / f"{subj_id}.csv"
        )
        tasks.append(("stages", score_remote, score_local))

    logger.info("Downloading %d EDF files...", len(tasks))
    _parallel_download(tasks, token, max_workers)


# =============================================================================
# Helpers
# =============================================================================


def _load_subject_list(path: Path) -> list[str]:
    """Load subject IDs from a text file (one per line)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Subject list not found: {path}\n"
            f"Run --stage metadata first to generate subject lists."
        )
    subjects = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    logger.info("Loaded %d subjects from %s", len(subjects), path.name)
    return subjects


def _parallel_download(
    tasks: list[tuple[str, str, Path]], token: str, max_workers: int
):
    """Download files in parallel with progress reporting."""
    total = len(tasks)
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(nsrr_download_file, ds, remote, local, token): (
                ds,
                remote,
                local,
            )
            for ds, remote, local in tasks
        }

        for future in as_completed(futures):
            ds, remote, local = futures[future]
            try:
                success = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error("Exception downloading %s: %s", remote, e)
                failed += 1

            if (completed + failed) % 50 == 0:
                logger.info(
                    "Progress: %d/%d completed, %d failed",
                    completed,
                    total,
                    failed,
                )

    logger.info(
        "Download complete: %d/%d successful, %d failed", completed, total, failed
    )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Smart selective download for STAGES/MESA from NSRR"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["metadata", "actigraphy", "edfs", "all"],
        help="Download stage: metadata, actigraphy (+ annotations), edfs, or all",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("./datasets"),
        help="Base directory for downloaded datasets",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs"),
        help="Directory for logs and subject lists",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel download threads",
    )
    parser.add_argument(
        "--dataset",
        choices=["mesa", "stages", "both"],
        default="both",
        help="Which dataset to download (default: both)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="Limit to N subjects per dataset (0 = all, useful for testing)",
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
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    token = get_nsrr_token()
    args.datasets_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("metadata", "all"):
        download_metadata(
            args.datasets_dir, args.log_dir, token,
            dataset=args.dataset, max_subjects=args.max_subjects,
        )

    if args.stage in ("actigraphy", "all"):
        download_actigraphy_and_labels(
            args.datasets_dir, args.log_dir, token, args.max_workers,
            dataset=args.dataset, max_subjects=args.max_subjects,
        )

    if args.stage in ("edfs", "all"):
        download_edfs(
            args.datasets_dir, args.log_dir, token, args.max_workers,
            dataset=args.dataset, max_subjects=args.max_subjects,
        )


if __name__ == "__main__":
    main()
