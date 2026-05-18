#!/usr/bin/env python3
"""Repair corrupted actigraphy JSON files from the STAGES Huami Arc ZIP.

Most files in 'Actigraphy Final to Post.zip' were stored as raw HTTP responses
(headers + JSON body) instead of pure JSON. This script:
  1. Reads each JSON file from the ZIP using system unzip (Deflate64)
  2. Strips HTTP headers if present (everything up to and including \r\n\r\n)
  3. Writes the clean JSON to datasets/stages/actigraphy_json/{subject_id}/minbymin/

Files with empty payloads ({"items":[]}) are written but flagged in the report.

Usage:
    python scripts/repair_actigraphy_zip.py
    python scripts/repair_actigraphy_zip.py --dry-run
    python scripts/repair_actigraphy_zip.py --zip path/to/zip --out-dir path/to/out
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_ZIP = "datasets/stages/original/Actigraphy Final to Post.zip"
DEFAULT_OUT = "datasets/stages/actigraphy_json"


def list_zip_json_files(zip_path: str) -> list[tuple[str, int]]:
    """Return list of (internal_path, uncompressed_size) for all .json entries."""
    result = subprocess.run(["unzip", "-v", zip_path], capture_output=True, text=True)
    if result.returncode not in (0, 1):  # 1 = warnings only
        raise RuntimeError(f"unzip -v failed (rc={result.returncode}): {result.stderr[:200]}")

    crc_re = re.compile(
        r"^\s*(\d+)\s+\S+\s+\d+\s+\S+\s+\S+\s+\S+\s+[0-9a-f]{8}\s+(.+)$"
    )
    entries = []
    for line in result.stdout.splitlines():
        if ".json" not in line:
            continue
        m = crc_re.match(line)
        if m:
            entries.append((m.group(2), int(m.group(1))))
    return entries


def extract_file(zip_path: str, internal_path: str, dest_dir: str) -> Path:
    """Extract a single file from the ZIP into dest_dir, return its path."""
    result = subprocess.run(
        ["unzip", "-o", zip_path, internal_path, "-d", dest_dir],
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"unzip failed (rc={result.returncode}) for {internal_path}: "
            f"{result.stderr[:200]}"
        )
    return Path(dest_dir) / internal_path


def strip_http_headers(raw: bytes) -> bytes:
    """Return only the body after HTTP headers, or raw unchanged if not HTTP."""
    if not raw.startswith(b"HTTP/"):
        return raw
    sep = raw.find(b"\r\n\r\n")
    if sep == -1:
        raise ValueError("HTTP response has no header/body separator (\\r\\n\\r\\n)")
    return raw[sep + 4:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=DEFAULT_ZIP, help="Path to source ZIP")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Output directory root")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't write")
    args = parser.parse_args()

    zip_path = str(Path(args.zip).resolve())
    out_root = Path(args.out_dir)

    if not Path(zip_path).exists():
        logger.error("ZIP not found: %s", zip_path)
        sys.exit(1)

    logger.info("Listing ZIP contents: %s", zip_path)
    entries = list_zip_json_files(zip_path)
    logger.info("Found %d JSON files in ZIP", len(entries))

    # Parse subject IDs from internal paths:
    # Actigraphy/Final to Post/{SUBJECT_ID}/minbymin/min_*.json
    subject_re = re.compile(
        r"^Actigraphy/Final to Post/([^/]+)/minbymin/(.+\.json)$"
    )
    valid_entries = []
    for ipath, size in entries:
        m = subject_re.match(ipath)
        if m:
            valid_entries.append((ipath, size, m.group(1), m.group(2)))
        else:
            logger.warning("Skipping unrecognised path: %s", ipath)

    logger.info("%d files match expected path pattern", len(valid_entries))

    counters = {"repaired": 0, "already_valid": 0, "empty": 0, "failed": 0}
    failures = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (ipath, size, subject_id, filename) in enumerate(valid_entries, 1):
            if i % 200 == 0:
                logger.info("Progress: %d/%d", i, len(valid_entries))

            out_path = out_root / subject_id / "minbymin" / filename

            try:
                extracted = extract_file(zip_path, ipath, tmpdir)
                raw = extracted.read_bytes()
            except Exception as e:
                logger.error("Extract failed for %s: %s", subject_id, e)
                counters["failed"] += 1
                failures.append((subject_id, str(e)))
                continue

            was_http = raw.startswith(b"HTTP/")
            try:
                body = strip_http_headers(raw)
            except ValueError as e:
                logger.error("Header strip failed for %s: %s", subject_id, e)
                counters["failed"] += 1
                failures.append((subject_id, str(e)))
                continue

            # Validate JSON
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON body for %s: %s", subject_id, e)
                counters["failed"] += 1
                failures.append((subject_id, f"JSON decode: {e}"))
                continue

            items = parsed.get("items", [])
            if len(items) == 0:
                counters["empty"] += 1

            if was_http:
                counters["repaired"] += 1
            else:
                counters["already_valid"] += 1

            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(body)

    # Report
    print("\n=== Repair Summary ===")
    print(f"  Repaired (HTTP stripped):  {counters['repaired']}")
    print(f"  Already valid (no HTTP):   {counters['already_valid']}")
    print(f"  Empty payload (items=[]):  {counters['empty']}  (written but no data)")
    print(f"  Failed:                    {counters['failed']}")
    if failures:
        print("\nFailures:")
        for subject, reason in failures:
            print(f"  {subject}: {reason}")
    if args.dry_run:
        print("\n[DRY RUN — no files written]")
    else:
        print(f"\nOutput written to: {out_root}")


if __name__ == "__main__":
    main()
