#!/usr/bin/env python3
"""
Download the MESA sleep dataset CSV from NSRR and extract demographic/clinical variables.

Usage:
    python3 scripts/download_demographics.py
"""

import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MESA_DIR = PROJECT_ROOT / "datasets" / "mesa"
TOKEN_FILE = Path.home() / ".nsrr_token"
NSRR_BASE = "https://sleepdata.org"


def get_token():
    """Read NSRR token from file."""
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    token = os.environ.get("NSRR_TOKEN")
    if token:
        return token
    print("ERROR: No NSRR token found. Create ~/.nsrr_token or set NSRR_TOKEN env var.")
    sys.exit(1)


def list_dataset_files(token):
    """List files in mesa/datasets/ on NSRR."""
    url = f"{NSRR_BASE}/api/v1/datasets/mesa/files.json"
    params = {"auth_token": token, "path": "datasets"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def download_file(dataset, file_path, dest, token):
    """Download a single file from NSRR."""
    url = f"{NSRR_BASE}/datasets/{dataset}/files/{file_path}"
    params = {"auth_token": token}
    print(f"  Downloading {file_path}...")
    resp = requests.get(url, params=params, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_kb = dest.stat().st_size / 1024
    print(f"  Saved: {dest} ({size_kb:.0f} KB)")


def main():
    token = get_token()

    # List available dataset files
    print("Listing MESA dataset files on NSRR...")
    files = list_dataset_files(token)

    csv_files = [f for f in files if f.get("file_name", "").endswith(".csv")]
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f['file_name']:50} {f.get('file_size', '?')} bytes")

    # Download all CSVs in the datasets folder
    for f in csv_files:
        file_name = f["file_name"]
        file_path = f"datasets/{file_name}"
        dest = MESA_DIR / file_name
        if dest.exists():
            print(f"  Already exists: {dest}")
            continue
        download_file("mesa", file_path, dest, token)

    # Also check for data dictionary
    all_files = requests.get(
        f"{NSRR_BASE}/api/v1/datasets/mesa/files.json",
        params={"auth_token": token, "path": ""},
    ).json()
    for f in all_files:
        name = f.get("file_name", "")
        if "variable" in name.lower() or "dictionary" in name.lower():
            dest = MESA_DIR / name
            if not dest.exists():
                download_file("mesa", name, dest, token)

    print("\nDone! Check datasets/mesa/ for downloaded files.")


if __name__ == "__main__":
    main()
